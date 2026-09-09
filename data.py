# everything about the data: reading the hyperspectral cubes, deriving severity
# labels, the plant-level split, preprocessing, the frozen DINOv2 encoder, the
# per-frame lesion graph, and the feature cache.
#
# uses two temporal campaigns pooled together:
#   March 2021  (Chantecler, 13 plants)  and  July 2022  (Gala, 10 temporal plants,
#   water-stress arm dropped). same camera / bands, different variety + inoculum,
#   so severity is an AREA FRACTION (lesion pixels / plant pixels) to be comparable.
#
# run once:  python data.py       (writes the split + builds the cache)

import datetime
import hashlib
import json
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from torch.utils.data import Dataset

from config import CONFIG

GEO_DIM = 8
KNN_K = 4
LESION_PATCH_FRAC = 0.25
DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"
DINO_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# per campaign: which plant numbers are inoculated / control. everything else
# (water stress, one-off acquisitions) is left out of the temporal dataset.
CAMPAIGNS = {
    "march": {
        "root": "Data _March2021/Data _March2021",
        "plant_re": r"plant (.+)$",
        "inoculated": {"1", "2", "3", "4", "5", "6", "7"},
        "control": {"8", "9", "10", "11", "12", "13"},
        "no_severity": {"5"},          # inoculated but lesions never locatable
    },
    "july": {
        "root": "data_July2022",
        "plant_re": r"plant_(.+)$",
        "inoculated": {"1", "2.1", "3"},
        "control": {"4", "5", "6", "7", "8", "9", "10"},
        "no_severity": set(),
    },
}
# fraction-of-plant-area cut points for the ordinal severity level
FRAC_BINS = (0.002, 0.01, 0.04)


# ---------- reading the ENVI cubes ----------

def read_hdr(hdr_path):
    text = Path(hdr_path).read_text(encoding="utf-8", errors="replace")

    def get(key, cast):
        m = re.search(r"^" + re.escape(key) + r"\s*=\s*(.+)$", text, re.MULTILINE)
        return cast(m.group(1).strip()) if m else None

    wl = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, re.DOTALL)
    return {
        "samples": get("samples", int), "lines": get("lines", int), "bands": get("bands", int),
        "data_type": get("data type", int), "byte_order": get("byte order", int),
        "interleave": (get("interleave", str) or "").lower(),
        "acquisition_date": get("acquisition date", str),
        "wavelengths": [float(x) for x in wl.group(1).split(",") if x.strip()] if wl else [],
    }


def acquisition_date(hdr):
    raw = hdr.get("acquisition_date")
    if not raw:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            pass
    return None


def read_cube(dat_path, hdr=None):
    dat_path = Path(dat_path)
    hdr = hdr or read_hdr(dat_path.with_suffix(".hdr"))
    assert hdr["interleave"] == "bil" and hdr["data_type"] == 4, "only BIL float32 supported"
    dt = np.dtype("<f4" if hdr["byte_order"] == 0 else ">f4")
    lines, samples, bands = hdr["lines"], hdr["samples"], hdr["bands"]
    raw = np.fromfile(dat_path, dtype=dt)
    if raw.size != lines * samples * bands:
        raise ValueError("%s: wrong size" % dat_path.name)
    cube = raw.reshape(lines, bands, samples).transpose(0, 2, 1)   # BIL -> (lines, samples, bands)
    return np.ascontiguousarray(cube, dtype=np.float32)


def read_bands(dat_path, band_indices, hdr=None):
    return read_cube(dat_path, hdr)[:, :, list(band_indices)]


# ---------- scan the folders ----------

Timepoint = namedtuple("Timepoint", "day dat count")


def _pos_count(plant_dir):
    pos = next((f for f in plant_dir.iterdir() if f.name.startswith("pos_")), None)
    return sum(1 for line in open(pos, encoding="utf-8") if line.strip()) if pos else None


def scan_campaign(name):
    # returns {"<name>_<plant>": [Timepoint, ...]} for the inoculated + control plants
    spec = CAMPAIGNS[name]
    root = Path(spec["root"])
    keep = spec["inoculated"] | spec["control"]
    index = {}
    for day_dir in sorted(root.iterdir()):
        m = re.match(r"day_(\d+)$", day_dir.name)
        if not (day_dir.is_dir() and m):
            continue
        day = int(m.group(1))
        for plant_dir in day_dir.iterdir():
            pm = re.match(spec["plant_re"], plant_dir.name)
            if not (plant_dir.is_dir() and pm and pm.group(1) in keep):
                continue
            dat = next((f for f in plant_dir.iterdir()
                        if re.match(r"REFLECTANCE_\d+\.dat$", f.name)), None)
            if dat is None:
                continue
            # only trust lesion counts for inoculated plants (controls have stray files)
            count = _pos_count(plant_dir) if pm.group(1) in spec["inoculated"] else None
            index.setdefault("%s_%s" % (name, pm.group(1)), []).append(Timepoint(day, dat, count))
    for k in index:
        index[k].sort(key=lambda t: t.day)
    return index


def scan_index():
    idx = {}
    for name in CAMPAIGNS:
        idx.update(scan_campaign(name))
    return idx


def plant_status(key):
    name, pid = key.split("_", 1)
    return "infected" if pid in CAMPAIGNS[name]["inoculated"] else "control"


def severity_eval_ok(key):
    name, pid = key.split("_", 1)
    return pid not in CAMPAIGNS[name]["no_severity"]


# ---------- severity: cumulative-max lesion-area fraction, then bins ----------

def running_max(xs):
    r, out = 0.0, []
    for x in xs:
        if x is not None:
            r = max(r, x)
        out.append(r)
    return out


def levels_from_fracs(fracs, bins=FRAC_BINS):
    return [int(sum(v >= b for b in bins)) for v in running_max(fracs)]


# ---------- plant-level split ----------

def make_split(seed):
    import random
    rng = random.Random(seed)
    groups = {}
    for key in scan_index():
        name = key.split("_", 1)[0]
        groups.setdefault((name, plant_status(key)), []).append(key)
    # (n_test, n_val) per (campaign, status); the rest -> train
    plan = {("march", "infected"): (2, 1), ("march", "control"): (2, 1),
            ("july", "infected"): (1, 1), ("july", "control"): (1, 1)}
    train, val, test = [], [], []
    for g, members in groups.items():
        members = [m for m in members if severity_eval_ok(m) or plant_status(m) == "control"]
        # keep the no-severity plants (march_5) out of val/test
        forced_train = [m for m in scan_index() if m.split("_")[0] == g[0]
                        and plant_status(m) == g[1] and not severity_eval_ok(m)]
        pool = [m for m in members if m not in forced_train]
        rng.shuffle(pool)
        nt, nv = plan.get(g, (0, 0))
        test += pool[:nt]
        val += pool[nt:nt + nv]
        train += pool[nt + nv:] + forced_train
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_split(seed=0):
    split = make_split(seed)
    Path(CONFIG["split_file"]).write_text(json.dumps({"seed": seed, **split}, indent=2))
    for k in ("train", "val", "test"):
        print("%-5s (%2d): %s" % (k, len(split[k]), split[k]))
    return split


def load_split():
    return json.loads(Path(CONFIG["split_file"]).read_text())


# ---------- preprocessing: line frames up with day 1 ----------

def align_frames(frames):
    from skimage.registration import phase_cross_correlation

    def plant_mask(g):
        return g > max(0.03, 0.5 * g.mean())

    ref = frames[0].mean(0).numpy()
    ref_m = ref * plant_mask(ref)
    out = [frames[0]]
    for t in range(1, frames.shape[0]):
        mov = frames[t].mean(0).numpy()
        s, _, _ = phase_cross_correlation(ref_m, mov * plant_mask(mov), upsample_factor=1)
        dy = int(np.clip(round(s[0]), -40, 40))
        dx = int(np.clip(round(s[1]), -40, 40))
        f = torch.zeros_like(frames[t])
        h, w = f.shape[-2:]
        f[:, max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
            frames[t][:, max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
        out.append(f)
    return torch.stack(out)


# ---------- frozen DINOv2 encoder ----------

class DinoV2Encoder:
    def __init__(self):
        import timm
        import torch.nn.functional as F
        self.F = F
        self.model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0,
                                       img_size=DINO_SIZE, dynamic_img_size=True).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.dev)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(self.dev)
        self.n_prefix = self.model.num_prefix_tokens

    @torch.no_grad()
    def encode(self, frames):
        x = frames.to(self.dev).clamp(0, 1)
        x = self.F.interpolate(x, size=DINO_SIZE, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        tok = self.model.forward_features(x)
        return tok[:, self.n_prefix:, :].cpu()          # drop CLS


# ---------- Stage 3: per-frame lesion graph + plant-area count ----------

def masks_and_area(frames, grid):
    # plant = bright pixels; lesion = browning index over Otsu, inside the plant.
    # also return the plant-pixel count per frame (for the area fraction).
    t, c, h, w = frames.shape
    lesion_grid = torch.zeros(t, grid, grid, dtype=torch.bool)
    plant_px = torch.zeros(t)
    for i in range(t):
        f = frames[i].numpy()
        gray = f.mean(0)
        plant = gray > max(0.03, 0.5 * gray.mean())
        plant_px[i] = float(plant.sum())
        browning = (f[0] - f[1]) / (f[0] + f[1] + 1e-6)
        vals = browning[plant]
        if vals.size < 50:
            continue
        try:
            thr = threshold_otsu(vals)
        except ValueError:
            continue
        les = (plant & (browning > thr)).reshape(grid, h // grid, grid, w // grid).mean((1, 3))
        pl = plant.reshape(grid, h // grid, grid, w // grid).mean((1, 3))
        lesion_grid[i] = torch.from_numpy((les > LESION_PATCH_FRAC) & (pl > 0.5))
    return lesion_grid, plant_px


def frame_graph(tokens, mask):
    # tokens: (P, D) with P = grid*grid ; mask: (grid, grid) bool.
    # one node per connected lesion blob. no lesion -> a single zero "null" node.
    grid = mask.shape[-1]
    d = tokens.shape[-1]
    tok_grid = tokens.reshape(grid, grid, d)
    lab = label(mask.numpy().astype(np.uint8), connectivity=2)

    feats, centroids = [], []
    for prop in regionprops(lab):
        cells = torch.from_numpy(lab == prop.label)
        minr, minc, maxr, maxc = prop.bbox
        try:
            ecc, sol = prop.eccentricity, prop.solidity
        except Exception:
            ecc, sol = 0.0, 1.0
        geo = [prop.area / grid ** 2, prop.centroid[0] / grid, prop.centroid[1] / grid,
               (maxr - minr) / grid, (maxc - minc) / grid, prop.extent, ecc, sol]
        feats.append(torch.cat([tok_grid[cells].mean(0), torch.tensor(geo, dtype=tokens.dtype)]))
        centroids.append(prop.centroid)

    if not feats:
        return torch.zeros(1, d + GEO_DIM, dtype=tokens.dtype), torch.tensor([[0], [0]])

    x = torch.stack(feats)
    n = x.shape[0]
    c = torch.tensor(centroids, dtype=torch.float32)
    dist = torch.cdist(c, c).fill_diagonal_(float("inf"))
    src, dst = [], []
    k = min(KNN_K, n - 1)
    if k > 0:
        for i, nbrs in enumerate(dist.topk(k, largest=False).indices):
            for j in nbrs.tolist():
                src += [i, j]
                dst += [j, i]
    src += list(range(n))
    dst += list(range(n))
    return x, torch.tensor([src, dst])


# ---------- feature cache ----------

def cache_key(cfg):
    payload = {"model": DINO_MODEL, "bands": cfg["band_indices"], "align": cfg["align"],
               "clip": cfg["clip"], "grid": cfg["patch_grid"], "frac": LESION_PATCH_FRAC,
               "campaigns": sorted(CAMPAIGNS)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cache_dir(cfg):
    return Path(cfg["cache_dir"]) / cache_key(cfg)


def build_cache(cfg):
    out = cache_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    index = scan_index()
    encoder = None
    for key in sorted(index):
        dst = out / (key + ".pt")
        if dst.exists():
            continue
        tps = index[key]
        frames = torch.stack([torch.from_numpy(read_bands(t.dat, cfg["band_indices"]))
                              .permute(2, 0, 1).clamp(max=cfg["clip"]) for t in tps])
        if cfg["align"]:
            frames = align_frames(frames)
        lesion_grid, plant_px = masks_and_area(frames, cfg["patch_grid"])
        if encoder is None:
            encoder = DinoV2Encoder()
        tokens = encoder.encode(frames).float()
        graphs = [frame_graph(tokens[i], lesion_grid[i]) for i in range(len(tps))]
        torch.save({"gx": [g[0].half() for g in graphs], "gei": [g[1] for g in graphs],
                    "days": [t.day for t in tps], "plant_px": plant_px}, dst)
        print("cached %-12s %2d frames  nodes %s" % (key, len(tps), [g[0].shape[0] for g in graphs]))
    return out


# ---------- datasets ----------

def _fake_graph(g, node_dim, max_nodes=5):
    n = int(torch.randint(1, max_nodes + 1, (1,), generator=g).item())
    x = torch.randn(n, node_dim, generator=g)
    src = list(range(n - 1)) + list(range(1, n)) + list(range(n))
    dst = list(range(1, n)) + list(range(n - 1)) + list(range(n))
    return x, torch.tensor([src, dst])


class SyntheticSequences(Dataset):
    def __init__(self, n, cfg, seed):
        g = torch.Generator().manual_seed(seed)
        node_dim = cfg["feature_dim"] + GEO_DIM
        t = 8
        self.items = []
        for _ in range(n):
            graphs = [_fake_graph(g, node_dim) for _ in range(t)]
            self.items.append({
                "gx": [a for a, _ in graphs], "gei": [b for _, b in graphs],
                "delta_t": torch.ones(t), "n_days": t,
                "y_disease": torch.randint(0, 2, (1,), generator=g)[0],
                "y_level": torch.randint(0, cfg["k_levels"], (1,), generator=g)[0],
                "y_level_seq": torch.randint(0, cfg["k_levels"], (t,), generator=g),
                "severity_eval": True, "plant_id": -1,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class CachedSequences(Dataset):
    def __init__(self, cfg, plant_keys):
        self.dir = cache_dir(cfg)
        if not self.dir.is_dir():
            raise FileNotFoundError("no cache at %s - run `python data.py`" % self.dir)
        self.keys = list(plant_keys)
        self.bins = cfg["severity_bins"]
        self.index = scan_index()
        self.key_id = {k: i for i, k in enumerate(sorted(self.index))}   # int id for logging

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        blob = torch.load(self.dir / (key + ".pt"))
        tps = self.index[key]
        plant_px = blob["plant_px"].clamp(min=1.0)
        fracs = [None if t.count is None else t.count / float(plant_px[j])
                 for j, t in enumerate(tps)]
        levels = levels_from_fracs(fracs, self.bins)
        days = [t.day for t in tps]
        delta = [0] + [days[k] - days[k - 1] for k in range(1, len(days))]
        return {
            "gx": [x.float() for x in blob["gx"]], "gei": blob["gei"],
            "delta_t": torch.tensor([max(1.0, float(x)) for x in delta]),
            "n_days": len(tps), "plant_id": self.key_id[key],
            "y_disease": torch.tensor(1 if plant_status(key) == "infected" else 0),
            "y_level": torch.tensor(levels[-1]),
            "y_level_seq": torch.tensor(levels),
            "severity_eval": severity_eval_ok(key),
        }


def collate(batch):
    t_max = max(x["n_days"] for x in batch)
    b = len(batch)
    delta_t = torch.zeros(b, t_max)
    frame_mask = torch.zeros(b, t_max, dtype=torch.bool)
    level_seq = torch.full((b, t_max), -1)
    for i, x in enumerate(batch):
        n = x["n_days"]
        delta_t[i, :n] = x["delta_t"]
        frame_mask[i, :n] = True
        level_seq[i, :n] = x["y_level_seq"]
    return {
        "gx": [x["gx"] for x in batch], "gei": [x["gei"] for x in batch],
        "delta_t": delta_t, "frame_mask": frame_mask, "y_level_seq": level_seq,
        "n_days": torch.tensor([x["n_days"] for x in batch]),
        "plant_id": torch.tensor([x["plant_id"] for x in batch]),
        "y_disease": torch.stack([x["y_disease"] for x in batch]),
        "y_level": torch.stack([x["y_level"] for x in batch]),
        "severity_eval": torch.tensor([bool(x["severity_eval"]) for x in batch]),
    }


def build_dataset(cfg, split):
    if cfg["source"] == "synthetic":
        seed = {"train": 0, "val": 1, "test": 2}[split]
        return SyntheticSequences(20 if split == "train" else 8, cfg, seed)
    return CachedSequences(cfg, load_split()[split])


if __name__ == "__main__":
    cfg = dict(CONFIG)
    write_split(seed=0)
    build_cache(cfg)
