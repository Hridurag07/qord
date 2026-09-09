# settings for the Mamba(GRU)+quantum severity model.
# use_quantum=False turns the quantum branch off (the ablation).

CONFIG = {
    "data_root": "Data _March2021/Data _March2021",
    "split_file": "plant_split.json",
    "cache_dir": "cache",
    "source": "synthetic",          # synthetic | march

    "band_indices": [70, 53, 19],
    "severity_bins": [0.001, 0.003, 0.02],   # lesion area fraction cut points
    "align": True,
    "clip": 1.0,

    "feature_dim": 384,
    "patch_grid": 16,
    "hidden_dim": 512,              # GRU (Mamba fallback) hidden size
    "classical_dim": 32,            # classical branch width
    "fused_dim": 16,                # what the heads read

    # quantum branch
    "use_quantum": True,
    "n_qubits": 8,
    "q_layers": 3,                  # data re-uploading blocks (each = re-encode + a trainable entangling layer)

    "head": "ordinal",             # ordinal | flat
    "prog_horizon": 3,
    "refine_tau": 0.15,
    "refine_max_passes": 5,

    "epochs": 40,
    "batch_size": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "k_levels": 4,
}


def config(**overrides):
    cfg = dict(CONFIG)
    cfg.update(overrides)
    return cfg
