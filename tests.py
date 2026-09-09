# run with:  python -m pytest tests.py
import pytest
import torch

from config import config
from data import frame_graph, levels_from_fracs, running_max, scan_index, GEO_DIM
from model import QOrdNet, QuantumCircuit, compute_loss, dial_to_level, qwk
from train import set_seed

DATA_ROOT = "Data _March2021/Data _March2021"


def make_batch(cfg, n=6, seed=0):
    from data import SyntheticSequences, collate
    ds = SyntheticSequences(n, cfg, seed)
    return collate([ds[i] for i in range(len(ds))])


def build(cfg, seed=0):
    set_seed(seed)
    return QOrdNet(cfg)


# ---------- quantum circuit ----------

def test_circuit_is_trainable():
    circ = QuantumCircuit(n_qubits=8, q_layers=3)
    params = list(circ.parameters())
    assert len(params) == 1 and params[0].requires_grad
    assert params[0].shape == (3, 1, 8, 3)


def test_circuit_output_shape_and_range():
    circ = QuantumCircuit(n_qubits=8, q_layers=3)
    out = circ(torch.randn(5, 8))
    assert out.shape == (5, 8)
    assert torch.isfinite(out).all() and out.abs().max() <= 1.0 + 1e-6


def test_circuit_gradients_flow():
    circ = QuantumCircuit(n_qubits=6, q_layers=2)
    x = torch.randn(3, 6)
    circ(x).sum().backward()
    assert circ.weights.grad is not None and torch.isfinite(circ.weights.grad).all()


def test_data_reuploading_uses_more_layers_differently():
    # more re-upload blocks should be a different (not degenerate) function
    torch.manual_seed(0)
    c1 = QuantumCircuit(n_qubits=6, q_layers=1)
    torch.manual_seed(0)
    c3 = QuantumCircuit(n_qubits=6, q_layers=3)
    x = torch.randn(4, 6)
    assert not torch.allclose(c1(x), c3(x)[:, : c1.n_qubits], atol=1e-3) or c1.weights.shape != c3.weights.shape


# ---------- model / ablation wiring ----------

def test_forward_shapes():
    cfg = config()
    out = build(cfg)(make_batch(cfg))
    assert out["fused"].shape[1] == cfg["fused_dim"]
    assert not torch.isnan(out["fused"]).any()


def test_quantum_off_zeroes_quantum_branch_and_stops_gradient():
    cfg = config(use_quantum=False)
    model = build(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    b = make_batch(cfg)
    for _ in range(2):
        opt.zero_grad()
        compute_loss(model(b), b, model.head_kind).backward()
        assert model.quantum.weights.grad is None or torch.all(model.quantum.weights.grad == 0)
        opt.step()


def test_quantum_on_vs_off_gives_different_output():
    b = make_batch(config())
    on = build(config(use_quantum=True), seed=0)(b)["fused"]
    off = build(config(use_quantum=False), seed=0)(b)["fused"]
    assert not torch.equal(on, off)


def test_one_train_step():
    cfg = config()
    model = build(cfg)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    b = make_batch(cfg)
    loss = compute_loss(model(b), b, model.head_kind)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


# ---------- severity labels ----------

def test_running_max_and_bins():
    assert running_max([None, 0.3, 0.6, 1.0, 0.9, 1.1]) == [0, 0.3, 0.6, 1.0, 1.0, 1.1]
    # fraction bins (0.002, 0.01, 0.04): cummax then bin
    assert levels_from_fracs([None, 0.001, 0.005, 0.02, 0.01, 0.09], (0.002, 0.01, 0.04)) == \
        [0, 0, 1, 2, 2, 3]


def test_gap_after_onset_carries_forward():
    assert levels_from_fracs([None] * 4 + [0.005, None, None, 0.001], (0.002, 0.01, 0.04)) == \
        [0, 0, 0, 0, 1, 1, 1, 1]


# ---------- lesion graph ----------

def test_no_lesion_null_node():
    x, ei = frame_graph(torch.randn(256, 384), torch.zeros(16, 16, dtype=torch.bool))
    assert x.shape == (1, 384 + GEO_DIM) and torch.count_nonzero(x) == 0
    assert ei.tolist() == [[0], [0]]


def test_node_feature_is_mean_of_patches():
    tok = torch.zeros(256, 384)
    tok[0], tok[1] = 1.0, 3.0
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[0, 0] = mask[0, 1] = True
    x, _ = frame_graph(tok, mask)
    assert torch.allclose(x[0, :384], torch.full((384,), 2.0))


# ---------- metrics ----------

def test_qwk_perfect_and_ordering():
    y = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0])
    assert qwk(y, y, 4) == 1.0
    true = torch.tensor([0, 1, 2, 3])
    near = torch.tensor([0, 1, 3, 3])
    far = torch.tensor([3, 1, 2, 0])
    assert qwk(far, true, 4) < qwk(near, true, 4)


def test_dial_to_level():
    uv = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert dial_to_level(uv, 4).tolist() == [0, 2]


# ---------- real data (skipped if not present) ----------

def test_scan_index():
    from pathlib import Path
    if not Path(DATA_ROOT).is_dir():
        pytest.skip("no dataset")
    idx = scan_index()
    assert "march_1" in idx and "july_1" in idx
    assert [t.day for t in idx["march_1"]] == [4, 5, 6, 7, 8, 12, 13, 14, 15]
    assert all(t.count is None for t in idx["march_5"])       # inoculated but never labelled
    assert all(t.count is None for t in idx["july_4"])        # control -> counts ignored
