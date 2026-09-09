# training, evaluation, and the quantum-on/off ablation.
#   python train.py --seed 0 --march
#   python train.py --ablation --seeds 12

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import config
from data import build_dataset, collate
from model import QOrdNet, _fit_line, compute_loss, dial_to_level, mae_levels, predicted_levels, qwk


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def loader(dataset, cfg, seed, shuffle):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle,
                      generator=g, collate_fn=collate)


def train_one_epoch(model, dl, opt):
    model.train()
    total = 0.0
    for batch in dl:
        opt.zero_grad()
        loss = compute_loss(model(batch), batch, model.head_kind)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(len(dl), 1)


@torch.no_grad()
def evaluate(model, dl, k):
    model.eval()
    sev_p, sev_t, dis_p, dis_t = [], [], [], []
    prog_p, prog_t = [], []
    per_plant_sev = {}
    for batch in dl:
        out = model(batch)
        keep = batch["severity_eval"]
        lv = predicted_levels(out, k)
        for i in range(len(keep)):
            if bool(keep[i]):
                per_plant_sev[int(batch["plant_id"][i])] = abs(int(lv[i]) - int(batch["y_level"][i]))
        sev_p.append(lv[keep])
        sev_t.append(batch["y_level"][keep])
        dis_p.append(out["disease_logits"].argmax(-1))
        dis_t.append(batch["y_disease"])
        if model.head_kind != "flat":
            fv = out["frame_vecs"]
            for i in range(fv.shape[0]):
                n = int(batch["frame_mask"][i].sum())
                if not bool(keep[i]) or n < 3:
                    continue
                # fit the dial trajectory on all but the last frame, predict the last
                uv = model.dial_traj(fv[i], n)
                elapsed = torch.cumsum(batch["delta_t"][i, :n], 0)
                coef = _fit_line(uv[:-1], elapsed[:-1])
                pred_uv = torch.tensor([elapsed[-1], 1.0]) @ coef
                prog_p.append(int(dial_to_level(pred_uv.unsqueeze(0), k)[0]))
                prog_t.append(int(batch["y_level_seq"][i, n - 1]))

    sev_p, sev_t = torch.cat(sev_p), torch.cat(sev_t)
    dis_p, dis_t = torch.cat(dis_p), torch.cat(dis_t)
    return {
        "severity_mae": mae_levels(sev_p, sev_t),
        "severity_qwk": qwk(sev_p, sev_t, k),
        "disease_acc": float((dis_p == dis_t).float().mean()),
        "progression_mae": mae_levels(torch.tensor(prog_p), torch.tensor(prog_t)) if prog_p else float("nan"),
        "per_plant_sev": per_plant_sev,
    }


def run(cfg, seed, tag="run", runs_root="runs"):
    set_seed(seed)
    train_dl = loader(build_dataset(cfg, "train"), cfg, seed, True)
    val_dl = loader(build_dataset(cfg, "val"), cfg, seed, False)
    test_dl = loader(build_dataset(cfg, "test"), cfg, seed, False)

    model = QOrdNet(cfg)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    losses = [train_one_epoch(model, train_dl, opt) for _ in range(cfg["epochs"])]
    val_m = evaluate(model, val_dl, cfg["k_levels"])
    test_m = evaluate(model, test_dl, cfg["k_levels"])

    d = Path(runs_root) / tag / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"val": val_m, "test": test_m, "train_loss": losses}, indent=2))
    print("%s seed=%d  val mae=%.3f qwk=%.3f | test mae=%.3f qwk=%.3f prog=%.3f" % (
        tag, seed, val_m["severity_mae"], val_m["severity_qwk"],
        test_m["severity_mae"], test_m["severity_qwk"], test_m["progression_mae"]))
    return val_m, test_m


# ---------- small-n stats ----------

def mean_sd(xs):
    xs = list(xs)
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def bootstrap_ci(xs, n_boot=10000, seed=0):
    xs = list(xs)
    if len(xs) < 2:
        return (xs[0], xs[0], xs[0]) if xs else (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(statistics.fmean(xs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return statistics.fmean(xs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def cohens_d(diffs):
    m, s = mean_sd(diffs)
    return m / s if s > 0 else 0.0


def ablation(seeds, epochs):
    # same seeds, quantum on vs off, everything else identical
    results = {"on": defaultdict(list), "off": defaultdict(list)}
    plant_sev = {"on": defaultdict(list), "off": defaultdict(list)}
    for seed in seeds:
        for tag, use_q in (("on", True), ("off", False)):
            cfg = config(source="march", use_quantum=use_q)
            if epochs:
                cfg["epochs"] = epochs
            _, test_m = run(cfg, seed, tag="quantum_" + tag, runs_root="runs/ablation")
            results[tag]["mae"].append(test_m["severity_mae"])
            results[tag]["qwk"].append(test_m["severity_qwk"])
            if test_m["progression_mae"] == test_m["progression_mae"]:   # not nan
                results[tag]["prog"].append(test_m["progression_mae"])
            for p, e in test_m["per_plant_sev"].items():
                plant_sev[tag][p].append(e)

    ex = {p: statistics.fmean(v) for p, v in plant_sev["on"].items()}
    ey = {p: statistics.fmean(v) for p, v in plant_sev["off"].items()}
    plants = sorted(set(ex) & set(ey))
    diffs = [ex[p] - ey[p] for p in plants]
    m, lo, hi = bootstrap_ci(diffs)

    summary = {
        "seeds": list(seeds),
        "quantum_on": {k: mean_sd(v) for k, v in results["on"].items()},
        "quantum_off": {k: mean_sd(v) for k, v in results["off"].items()},
        "contrast_mae_on_minus_off": {"mean_diff": m, "ci95": [lo, hi], "cohens_d": cohens_d(diffs)},
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== quantum ablation (%d seeds) ===" % len(seeds))
    print("  quantum ON : MAE %.2f+-%.2f  QWK %.2f+-%.2f" % (*summary["quantum_on"]["mae"], *summary["quantum_on"]["qwk"]))
    print("  quantum OFF: MAE %.2f+-%.2f  QWK %.2f+-%.2f" % (*summary["quantum_off"]["mae"], *summary["quantum_off"]["qwk"]))
    c = summary["contrast_mae_on_minus_off"]
    print("  on - off MAE: %.3f  CI [%.3f, %.3f]  d=%.2f  (negative = quantum helps)" % (
        c["mean_diff"], c["ci95"][0], c["ci95"][1], c["cohens_d"]))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--march", action="store_true", help="use real data")
    ap.add_argument("--no-quantum", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()

    if args.ablation:
        ablation(range(args.seeds), args.epochs)
    else:
        cfg = config(source="march" if args.march else "synthetic", use_quantum=not args.no_quantum)
        if args.epochs:
            cfg["epochs"] = args.epochs
        run(cfg, args.seed)


if __name__ == "__main__":
    main()
