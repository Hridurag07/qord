# QOrd-Net

Checking whether the quantum layer in hybrid quantum-classical plant-disease
models actually helps or just adds parameters. Apple fire-blight severity
(ordinal 0-3) + forecasting severity forward in time.

Notes + results: `docs/notes.md`, `docs/results.md`.

## files

- `config.py` - settings + the 5 arm definitions
- `data.py` - read cubes, severity labels, plant split, preprocess, DINOv2, lesion graph, cache
- `model.py` - quantum circuit, QOrdNet, loss, metrics
- `train.py` - train / evaluate / the sweep
- `tests.py`

## run

```
qordnet-env/Scripts/python.exe -m pytest tests.py
qordnet-env/Scripts/python.exe data.py                       # write split + build DINOv2 cache (once)
qordnet-env/Scripts/python.exe train.py --arm e --seed 0 --march
qordnet-env/Scripts/python.exe train.py --sweep --seeds 12
```

## the 5 arms

One model, `arm_config(name)` changes only `head`, `alpha`, `use_entanglement`.

| arm | head | alpha | entanglement |
|---|---|---|---|
| a | flat | 0 | - |
| b | ordinal | 0 | - |
| c | flat | 0.1 | on |
| d | ordinal | 0.1 | off |
| e | ordinal | 0.1 | on |

alpha=0 -> the model is exactly the classical one. d = e with the CNOT chain removed.

## pipeline

align frames -> frozen DINOv2 patch tokens -> per-frame lesion graph (Otsu browning
mask -> blobs -> nodes) -> GATv2 -> GRU over days -> classical prototype + 6-qubit
circuit -> fusion p = p_c + alpha*q -> refinement -> heads (disease / severity /
progression line-fit).
