# the model: GATv2 + GRU (Mamba fallback) over time, fused with a trainable
# quantum circuit, ordinal severity + progression forecast.

import math

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from data import GEO_DIM


# ---------- the quantum branch ----------
# trainable variational circuit with data re-uploading: re-encode the input
# between q_layers trainable entangling blocks, so a small circuit still gets to
# use its parameters more than once. this is the standard way small circuits are
# made more expressive (Perez-Salinas et al., "data re-uploading").

class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits, q_layers):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(x, weights):
            for l in range(q_layers):
                qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[l], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circ = circ
        # StronglyEntanglingLayers wants (n_layers, n_qubits, 3) per block
        self.weights = nn.Parameter(0.1 * torch.randn(q_layers, 1, n_qubits, 3))

    def forward(self, x):
        return torch.stack(self.circ(x, self.weights), dim=-1).to(x.dtype)


# ---------- the model ----------

def _fit_line(y, t):
    # least squares y = slope*t + intercept, per column of y
    a = torch.stack([t, torch.ones_like(t)], dim=1)
    ata = a.T @ a + 1e-6 * torch.eye(2, device=t.device)
    return torch.linalg.solve(ata, a.T @ y)


class QOrdNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_kind = cfg["head"]
        self.use_quantum = cfg["use_quantum"]
        self.k = cfg["k_levels"]
        self.tau = cfg["refine_tau"]
        self.max_passes = cfg["refine_max_passes"]
        self.prog_horizon = cfg["prog_horizon"]
        s, hid = cfg["hidden_dim"], 32
        c_dim, n_qubits, f_dim = cfg["classical_dim"], cfg["n_qubits"], cfg["fused_dim"]

        # Stage 3-4: GATv2 over each frame's lesion graph, GRU over the days
        self.gat1 = GATv2Conv(cfg["feature_dim"] + GEO_DIM, hid, heads=2, concat=True)
        self.gat2 = GATv2Conv(2 * hid, hid, heads=1)
        self.frame_proj = nn.Linear(2 * hid, s)
        self.gru = nn.GRU(s + 1, s, batch_first=True)

        # classical branch + quantum branch, fused together
        self.classical = nn.Sequential(nn.Linear(s, c_dim), nn.Tanh())
        self.q_proj = nn.Linear(s, n_qubits)
        self.quantum = QuantumCircuit(n_qubits, cfg["q_layers"])
        self.fuse = nn.Sequential(nn.Linear(c_dim + n_qubits, f_dim), nn.Tanh())
        self.refine_net = nn.Linear(f_dim, f_dim)

        self.head_disease = nn.Linear(f_dim, 2)
        self.head_sev_flat = nn.Linear(f_dim, self.k)
        self.head_sev_dial = nn.Linear(f_dim, 2)

    def backbone(self, gx, gei, delta_t, frame_mask):
        dev = self.frame_proj.weight.device
        b, s = len(gx), self.frame_proj.out_features
        summaries = []
        frame_vecs = torch.zeros(b, frame_mask.shape[1], s, device=dev)
        for i in range(b):
            n = int(frame_mask[i].sum())
            fv = []
            for t in range(n):
                x, ei = gx[i][t].to(dev), gei[i][t].to(dev)
                h = F.elu(self.gat1(x, ei))
                h = self.gat2(h, ei)
                fv.append(self.frame_proj(torch.cat([h.mean(0), h.amax(0)])))
            seq = torch.stack(fv)
            frame_vecs[i, :n] = seq
            gru_in = torch.cat([seq, delta_t[i, :n, None].to(dev)], dim=-1).unsqueeze(0)
            _, hn = self.gru(gru_in)
            summaries.append(hn[-1, 0])
        return torch.stack(summaries), frame_vecs

    def fuse_features(self, summary):
        c_feat = self.classical(summary)
        if self.use_quantum:
            angles = torch.tanh(self.q_proj(summary)) * math.pi
            q_feat = self.quantum(angles)
        else:
            q_feat = torch.zeros(summary.shape[0], self.quantum.n_qubits, device=summary.device)
        return self.fuse(torch.cat([c_feat, q_feat], dim=-1))

    def refine(self, p):
        passes = 0
        while passes < self.max_passes:
            unc = 1.0 - torch.softmax(p, dim=-1).amax(dim=-1)
            if not bool((unc >= self.tau).any()):
                break
            p = p + torch.tanh(self.refine_net(p))
            passes += 1
        return p, passes

    def dial_traj(self, frame_vecs, n):
        return self.head_sev_dial(self.fuse_features(frame_vecs[:n]))

    def progression(self, frame_vecs, frame_mask, delta_t):
        # fit the (u,v) trajectory vs elapsed time, predict prog_horizon days ahead
        preds = []
        for i in range(frame_vecs.shape[0]):
            n = int(frame_mask[i].sum())
            if n < 2:
                preds.append(torch.zeros(2, device=frame_vecs.device))
                continue
            uv = self.dial_traj(frame_vecs[i], n)
            elapsed = torch.cumsum(delta_t[i, :n], 0)
            coef = _fit_line(uv, elapsed)
            preds.append(torch.tensor([elapsed[-1] + self.prog_horizon, 1.0],
                                      device=uv.device) @ coef)
        return torch.stack(preds)

    def forward(self, batch):
        summary, frame_vecs = self.backbone(batch["gx"], batch["gei"],
                                            batch["delta_t"], batch["frame_mask"])
        p, passes = self.refine(self.fuse_features(summary))
        out = {"fused": p, "refinement_passes": passes,
               "disease_logits": self.head_disease(p), "frame_vecs": frame_vecs}
        if self.head_kind == "flat":
            out["severity_logits"] = self.head_sev_flat(p)
        else:
            out["severity_uv"] = self.head_sev_dial(p)
            fu = torch.zeros(*frame_vecs.shape[:2], 2, device=frame_vecs.device)
            for i in range(frame_vecs.shape[0]):
                n = int(batch["frame_mask"][i].sum())
                fu[i, :n] = self.dial_traj(frame_vecs[i], n)
            out["frame_uv"] = fu
            out["progression_uv"] = self.progression(frame_vecs, batch["frame_mask"], batch["delta_t"])
        return out


# ---------- loss ----------

def dial_loss(uv, y_level):
    # 1 - cos(theta_hat - theta*), no atan2 in the backward pass
    u, v = uv[..., 0], uv[..., 1]
    ts = y_level.to(uv.dtype) * (math.pi / 4.0)
    num = u * torch.cos(ts) + v * torch.sin(ts)
    den = torch.sqrt(u * u + v * v + 1e-8)
    return 1.0 - num / den


def compute_loss(out, batch, head_kind):
    l_dis = F.cross_entropy(out["disease_logits"], batch["y_disease"])
    if head_kind == "flat":
        l_sev = F.cross_entropy(out["severity_logits"], batch["y_level"])
        l_frame = torch.zeros((), device=l_dis.device)
    else:
        l_sev = dial_loss(out["severity_uv"], batch["y_level"]).mean()
        fm = batch["frame_mask"]
        per = dial_loss(out["frame_uv"], batch["y_level_seq"].clamp(min=0))
        l_frame = (per * fm).sum() / fm.sum().clamp(min=1)
    return l_dis + l_sev + 0.3 * l_frame


# ---------- metrics ----------

def dial_to_level(uv, k=4):
    theta = torch.remainder(torch.atan2(uv[:, 1], uv[:, 0]), 2 * math.pi)
    return torch.round(theta / (math.pi / 4.0)).clamp(0, k - 1).long()


def predicted_levels(out, k=4):
    if "severity_logits" in out:
        return out["severity_logits"].argmax(-1)
    return dial_to_level(out["severity_uv"], k)


def confusion(pred, true, k):
    cm = torch.zeros(k, k, dtype=torch.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def qwk(pred, true, k):
    n = true.numel()
    if n < 2:
        return float("nan")
    cm = confusion(pred, true, k).double()
    idx = torch.arange(k, dtype=torch.float64)
    w = (idx.view(-1, 1) - idx.view(1, -1)) ** 2 / (k - 1) ** 2
    exp = torch.outer(cm.sum(1), cm.sum(0)) / n
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1.0 - (w * cm).sum() / denom)


def mae_levels(pred, true):
    return float((pred - true).abs().float().mean())
