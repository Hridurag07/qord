# QOrd-Net

Does the quantum layer in a hybrid quantum-classical plant-disease model actually
contribute anything, or is it just extra parameters?

Published hybrid models (CNN -> squeeze to qubits -> variational circuit -> softmax)
report 98-99% accuracy on plant disease, but none of them compare against the same
model with the quantum branch switched off. This repo builds one such model
properly and runs that comparison, on apple fire blight severity from temporal
hyperspectral imaging.

Severity is treated as an **ordinal** quantity (a 0-3 "dial", not flat classes) and
is also **forecast forward in time**, which the reviewed literature doesn't do.

**Result: on this data, the quantum branch makes no measurable difference.**
See [docs/results.md](docs/results.md) for the numbers, CIs and caveats.

## findings

Quantum on vs off, everything else identical, 12 paired seeds, 23 plants,
6 held-out test plants:

|             | severity MAE | severity QWK |
|-------------|--------------|--------------|
| quantum ON  | 1.42 +- 0.75 |  0.01 +- 0.48 |
| quantum OFF | 1.38 +- 0.46 | -0.12 +- 0.22 |

Per-plant paired difference (on - off) in MAE: **+0.042**, 95% bootstrap CI
**[-0.222, 0.250]**, Cohen's d = 0.12. The CI crosses zero.

Two caveats, both expanded in [docs/results.md](docs/results.md):

- The base model doesn't predict severity well to begin with (QWK near 0, MAE ~1.4
  on a 0-3 scale), so there isn't much headroom for a quantum branch to improve.
- An earlier 13-plant version showed the trainable circuit actively *hurting*
  (d = +1.4, CI excluding zero) - consistent with overfitting extra trainable
  parameters on 7 training sequences. Pooling to 23 plants removed the harm but
  did not turn it into a benefit.

## data

Gaci et al. 2023, *Data in Brief* 50:109532 - apple trees inoculated with fire
blight (*Erwinia amylovora*), imaged with a Specim IQ (512x512, 204 bands,
397-1004 nm), already reflectance-calibrated by the camera.

- Dataset DOI: [10.57745/R6AMN3](https://doi.org/10.57745/R6AMN3) (INRAE /
  Recherche Data Gouv, Licence Ouverte / etalab 2.0)
- Article DOI: [10.1016/j.dib.2023.109532](https://doi.org/10.1016/j.dib.2023.109532)

Two temporal campaigns are pooled (23 plants):

| campaign | variety | plants used | notes |
|---|---|---|---|
| March 2021 | Chantecler | 13 (7 inoculated / 6 control) | days 4-15 post-inoculation, gapped |
| July 2022 | Gala | 10 (3 inoculated / 7 control) | water-stress arm and the short leaves dropped |
| May 2022 (orchard) | - | not used | single visit, non-temporal |

Pooling is a known limitation: the campaigns differ in variety, inoculum and
camera distance. Sequences are variable-length and have real gaps, so the model
consumes actual elapsed dt from the `.hdr` acquisition dates, not a step index.

**Severity labels are derived, not provided.** Severity = lesion pixels / plant
pixels (area fraction, so the two campaigns are comparable), taken as a running
maximum over time, then binned at `(0.001, 0.003, 0.02)` into levels 0-3. Controls
are forced to 0. Plant `march_5` is inoculated but never labelled in the source
data, so it is forced into train and excluded from severity scoring rather than
dropped.

Splits are **by plant, never by frame** - consecutive days of the same plant are
near-duplicates, and a frame-level split would leak test into train. The split is
stratified by (campaign, status) and written once to
[plant_split.json](plant_split.json): train 13 / val 4 / test 6.

The raw cubes (~17 GB) are not in this repo. Download them from the DOI above and
point `data_root` in [config.py](config.py) at them.

## pipeline

```
align frames (phase cross-correlation)
  -> frozen DINOv2 patch tokens        (per frame, cached to disk)
  -> per-frame lesion graph            (Otsu browning mask -> blobs -> nodes, k-NN edges)
  -> GATv2 x2 -> mean+max pool
  -> GRU over days (Mamba fallback), fed elapsed dt
  -> classical branch + quantum branch
  -> fuse -> uncertainty-guided refinement
  -> heads: disease / severity dial / progression
```

**Quantum branch** - PennyLane `default.qubit`, torch interface, backprop.
8 qubits, 3 data-reuploading blocks; each block re-encodes the input with
`AngleEmbedding` (Y-rotations) then applies a trainable `StronglyEntanglingLayers`.
Readout is a Pauli-Z expectation per qubit. `use_quantum=False` zeroes the branch
and stops gradient through it - that's the ablation, and it has its own test.

**Severity head (the dial)** - anchor angles theta_l = l*pi/4 for l in {0,1,2,3};
the head emits (u,v) and the prediction is the nearest anchor to atan2(v,u). Loss
is 1 - cos(theta_hat - theta*), implemented as
1 - (u*cos theta* + v*sin theta*)/sqrt(u^2+v^2) so atan2 never appears in the
backward pass.

**Progression head** - fits a line to the dial trajectory over days 1..T and
extrapolates to T+k.

## setup

Python 3.11. The versions in [requirements.txt](requirements.txt) are what is
installed in `qordnet-env` (torch 2.11 + cu126).

```bash
python -m venv qordnet-env
qordnet-env/Scripts/pip install -r requirements.txt   # Linux/macOS: qordnet-env/bin/pip
```

## run

```bash
python -m pytest tests.py            # synthetic data, no dataset needed

python data.py                       # write split + build the DINOv2 cache (once, ~25 min)

python train.py --seed 0 --march     # single run on real data
python train.py --seed 0 --march --no-quantum
python train.py --ablation --seeds 12    # the paired quantum on/off comparison
```

Without `--march` everything runs on synthetic tensors of the right shape, which is
useful for checking the plumbing before committing to the cache build. Runs are
written to `runs/`; the ablation also writes `runs/ablation_summary.json`.

## repo layout

| file | what's in it |
|---|---|
| [config.py](config.py) | one `CONFIG` dict + `config(**overrides)` |
| [data.py](data.py) | ENVI reader, campaign index, severity labels, split, preprocessing, DINOv2, lesion graph, cache |
| [model.py](model.py) | quantum circuit, `QOrdNet`, losses, metrics |
| [train.py](train.py) | train / evaluate / the ablation + bootstrap CIs |
| [tests.py](tests.py) | circuit trainability and gradients, quantum-off really is off, null-lesion node, severity binning, QWK, dial decoding |
| [docs/notes.md](docs/notes.md) | working notes, status, open questions |
| [docs/results.md](docs/results.md) | results, limits, bottom line |

Raw data, the feature cache, `runs/` and the venv are gitignored.

## statistics

Small n, so no default parametric tests. Seeds are **paired across arms** (the
identical seed set for quantum on and off), differences are taken per plant, and
CIs are bootstrapped over plant-level paired differences rather than run through a
t-test. Cohen's d is reported alongside, not instead.

## references

- Bowles, Ahmed & Schuld (2024), *Better than classical?* - [arXiv:2403.07059](https://arxiv.org/abs/2403.07059) - the benchmarking critique this audit is modelled on
- QAttn-CNN - [Sci. Rep. s41598-025-31122-x](https://www.nature.com/articles/s41598-025-31122-x)
- QMSAN - [Neural Networks S0893608025000024](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000024) - ordinal/positional encoding via fixed R_x rotations
- HQC-CNN - [IEEE 11325703](https://ieeexplore.ieee.org/iel8/11325112/11325114/11325703.pdf) - the standard quantum plant-disease recipe
- Wang, Sun & Wang (2017) - [PMC5516765](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5516765/) - severity-as-ordinal framing (cited for framing; that dataset is not used here)
- DINOv2 - [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | GATv2 - [arXiv:2105.14491](https://arxiv.org/abs/2105.14491) | Mamba - [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | CORAL ordinal regression - [arXiv:1901.07884](https://arxiv.org/abs/1901.07884)

## status

Research code for a faculty-supervised project - not a package, not stable.
Open items are tracked at the bottom of [docs/notes.md](docs/notes.md). The main
two: the progression eval currently predicts a plateaued last step that
persistence already nails, and the severity range is thin (only one test plant
reaches level 3).
