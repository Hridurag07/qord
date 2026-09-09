# QOrd-Net — Project Instructions for Claude Code

## Who this is for
College student (CS/AI&ML), building this for a faculty-supervised research paper +
implementation. Not deeply versed in QML or deep learning — explain non-obvious choices
in plain terms in commit messages / comments, but keep the actual code and math technically
correct and defensible to a faculty reviewer. Always include paper links when citing a method.

## The project in one paragraph
A controlled audit of whether the quantum layer in hybrid quantum-classical plant-disease
models contributes anything beyond its parameter count, applied to apple leaf disease severity
(treated as an ordinal "dial" position, not flat classes) and severity *progression* forecasting
over time. The contribution is the audit + the progression head, not a new architecture.

## Core novelty claim
Multiple published papers apply quantum layers to plant disease (CNN → squeeze to qubits →
variational circuit → softmax), all ~98-99% accuracy, none with a parameter-matched classical
baseline, none testing with entanglement removed. Bowles, Ahmed & Schuld (arXiv:2403.07059)
found classical models generally match/beat quantum ones across 160 benchmark datasets, and
that removing entanglement often doesn't hurt — nobody has run that check in agriculture. This
paper does. Severity is additionally modelled as an ordinal, sequence-aware quantity (forecastable
forward in time), which no reviewed paper does either.

## IMPORTANT — what "success" means here (read before writing eval code)
Faculty advisor wants a positive result, not a null result. Given the small dataset (below),
the entanglement ablation (Arm E vs D) may be underpowered and inconclusive either way — that's
expected, not a bug. **The paper's headline positive claim is the progression-forecast head**
(no reviewed paper forecasts severity forward in time — a demonstrated capability, not a
p-value). The entanglement ablation is reported as an honest secondary finding whatever it shows.
Do not let this framing pressure the ablation's implementation or analysis — the audit must stay
methodologically clean (see "Statistics" below). If a change to the ablation analysis would only
make sense because it improves the odds of a significant result, stop and flag it instead of
making it.

## Core source papers (always cite with links; do not add more without asking)
- QAttn-CNN — https://www.nature.com/articles/s41598-025-31122-x — CNN + quantum attention
  block template; the paper faculty cited as an example of real novelty.
- QMSAN — https://www.sciencedirect.com/science/article/abs/pii/S0893608025000024 — encodes
  order via fixed, non-trainable single-qubit R_x rotations (sinusoidal schedule adapted from
  classical Transformer positional encoding). Transplanted from word position to severity order.
  Ordinal gate angles are indexed by WIRE (qubit index), not by severity label — severity is the
  model's output, so conditioning gates on it would be label leakage. Open item: verify QMSAN's
  Fig. 5 "ring configuration" before finalizing ring vs open CNOT chain (see Open Items).
- Wang, Sun & Wang (2017) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5516765/ — problem
  framing + apple black rot severity dataset (PlantVillage-derived). NOTE: this dataset is
  NOT being used as the primary audit dataset (see Dataset below) — cite for framing only.
- HQC-CNN — https://ieeexplore.ieee.org/iel8/11325112/11325114/11325703.pdf — standard quantum
  plant-disease recipe, used as Arm C's reference. NOTE: HQC-CNN's circuit is variational
  (trainable); our circuit as specified is fixed/non-trainable (see Quantum circuit below) —
  Arm C reproduces a simplified version of the published recipe, state this explicitly in the
  paper, don't imply exact replication.
- Bowles, Ahmed & Schuld — https://arxiv.org/abs/2403.07059 — the benchmarking critique
  motivating the audit framing.
- Component citations (not core novelty papers, cite when implementing that piece): Oquab et
  al. 2023 DINOv2 (arXiv:2304.07193), Brody/Alon/Yahav 2022 GATv2 (arXiv:2105.14491), Gu & Dao
  2023 Mamba (arXiv:2312.00752), Cao/Mirjalili/Raschka 2019 CORAL ordinal regression
  (arXiv:1901.07884 — cited as the classical alternative we compare against, not dismissed).

## Dataset — CONFIRMED, do not substitute without asking
Gaci et al. 2023, Data in Brief 50:109532, article DOI 10.1016/j.dib.2023.109532.
Dataset itself: INRAE / Recherche Data Gouv, DOI 10.57745/R6AMN3 (V2), Licence Ouverte /
etalab 2.0 (CC-BY-compatible). Camera: Specim IQ, 512×512, 204 bands.

The full published dataset has THREE groups; we have downloaded only the first:
- **March 2021 (lab)** — the ONLY group in hand. 13 bench-grafted "Chantecler" plants:
  7 inoculated with fire blight (Erwinia amylovora, strain CFBP3472), 6 control. This is the
  primary and (currently) sole audit dataset.
- July 2022 (lab) — 3 inoculated + water-stress + control, temporal over ~23 days. NOT
  downloaded. Would be needed for any n≈23 pooling (see Open Items) — treat as unavailable
  until explicitly fetched.
- May 2022 (orchard) — single-day field acquisitions, 9 symptomatic + 6 symptom-free trees.
  NOT downloaded. Field validation only if ever fetched; not part of sequence training/eval.

### March 2021 group — actual structure (verified against the archive + paper §2.1, Table 1)
Local path (note the space and the double nesting):
`Data _March2021/Data _March2021/day_<D>/plant <N>/`
- `day_<D>` = **days after inoculation** (biological clock, NOT calendar date, NOT step index).
- 13 plants, run as 3 biological sub-trials staggered ~1 week apart (so calendar dates in the
  .hdr files span late March into April — this is one continuous experiment, "the April cohort
  is just the March one extended", not a separate group):
  - **SS1**: inoculated {1,2,3,4} + control {8,9}. Acquired days 4–15, **missing days 9–11**
    → 9 timepoints/plant. Naked-eye symptoms from ~day 5.
  - **SS2**: inoculated {5,6,7} + control {10,11}. Acquired days 1–15, complete → 15
    timepoints/plant. Naked-eye symptoms from ~day 7.
  - **SS3**: control {12,13}. Acquired days 1–15, **missing days 3, 12, 13** → 12 timepoints.
- **Infected = plants {1,2,3,4,5,6,7}; control = {8,9,10,11,12,13}.** (Resolves the earlier
  "is plant 5 infected?" question: yes, inoculated — it just has no locatable lesion pixels,
  see below.)

### Files per `plant <N>` folder (paper §2; 153 folders total)
- `REFLECTANCE_<id>.dat` — ENVI binary hyperspectral cube. **BIL interleave**, `data type = 4`
  (float32), `byte order = 0` (little-endian), `header offset = 0`, 512 samples × 512 lines ×
  204 bands (each file exactly 213,909,504 bytes). Values are **already reflectance-calibrated
  by the camera** (Spectralon SRS-50 in scene, correction pre-applied) — range ≈ 0–1 with a
  few specular pixels up to ~1.9.
- `REFLECTANCE_<id>.hdr` — ENVI text header. Per-band wavelengths listed explicitly:
  **397.32 – 1003.58 nm**, 204 bands, ~3 nm spacing (authoritative; paper prose variously says
  "397–966" / "400–1000"). Also carries `acquisition date` (DD-MM-YYYY) and `tint`.
- `<id>.png` — 512×512 RGB preview render (from `default bands = {70,53,19}`). Not for modelling.
- `pos_p<N>.csv` — **only for symptomatic plants, only from symptom onset onward, not every
  symptomatic day.** No header; two columns = (vertical row index, horizontal col index) of
  each symptomatic pixel, 0–511. Sparse coordinate list, NOT a raster mask. Pixel counts are
  small (tens–hundreds) and grow over time — this growth IS the severity-progression signal.
  Present for plants {1,2,3,4,6,7}. **Plant 5: inoculated but NO csv on any day** (paper: "it
  was difficult to locate the symptomatic pixels ... the position of these pixels is not
  provided"). Days 1–4: no csv for anyone.

### Severity ground truth — DERIVED, not provided
The dataset has **no severity grades**. Decision (confirmed): define the ordinal severity
level ℓ∈{0,1,2,3} from the growth of the symptomatic-pixel count over time (per plant,
per timepoint). Open sub-decisions before implementing the head: exact bin thresholds;
whether to normalise pixel count by leaf area (needs a leaf mask, not provided) or use raw
counts at fixed camera distance (50 cm, constant for March); label for inoculated pre-symptom
days and for plant 5 (inoculated, no masks) — candidate: ℓ=0 until first csv, then binned.
Controls {8,9,10,11,12,13} are ℓ=0 on all days by construction.

### Splitting & sequence length
- **Split by plant, never by frame** — consecutive days of the same plant are near-duplicates;
  a frame-level split leaks test into train and inflates every arm identically, hiding rather
  than causing a discrepancy. With only the March group, **n = 13 (7 infected / 6 control)**.
- Consider stratifying the split by (sub-trial × infection status); n per cell is tiny, so
  document the exact split and keep it fixed (see Repo conventions).
- Sequences are **variable length and gapped**: T ranges 9–15 real timepoints. Treat T=15
  (days post-inoculation) as the padded upper bound, and feed real elapsed Δt (below), not a
  dense day index.
- Δt source: difference of `acquisition date` fields in the `.hdr` files for the same plant
  (equivalently, the `day_<D>` differences, since `day_<D>` = days post-inoculation). Confirmed.

### Housekeeping done
Deleted from the archive: `__MACOSX/` (AppleDouble cruft) and a stray non-spec RGB export
`day_7/plant 5/371.hdr` + `371.img`. The 17.4 GB `Data _March2021.zip` is still present and
re-downloadable from DOI 10.57745/R6AMN3 — extraction has been verified complete (all 153
cubes, 41 csvs).

## Statistics (small-n specific — do not use default parametric tests)
- Report μ ± σ over multiple seeds, seeds PAIRED across arms (identical seed set for every arm).
- Given n≈13-23, prefer bootstrap confidence intervals over plant-level paired differences
  rather than a parametric significance test (e.g. paired t-test) — assumptions are shaky at
  this n. Effect size d = (μ_E − μ_D) / s_pooled still reported alongside.
- Before running the full C/D/E sweep: run Arms A and B first, measure σ across 10 seeds, and
  do a power check (roughly: minimum detectable paired difference ≈ 2.9 × σ_diff / √n_seeds).
  If that's larger than a difference worth caring about, flag it — more seeds are cheap here
  since the trainable model is tiny, so this is a "run more seeds" fix, not a blocker.

## Confirmed architecture decisions
- **Backbone: DINOv2** (not MobileNetV2 — that was an earlier decision for a since-abandoned
  single-image-only scope). Frozen, features cached to disk once. ViT-S/14 patch tokens
  (384-d, 16×16 grid) unless a ceiling effect (near-100% accuracy on Arms A/B leaving no
  headroom to detect a quantum effect) argues for dropping to a weaker encoder — that's an
  empirical call to make after the Arm A/B checkpoint, not before.
- **Full sequence pipeline is the project** — Stages 1-7 below, not a trimmed single-image
  variant. Lesion-graph construction (Stage 3) and Graph-Mamba (Stage 4) are core, not optional.
- **Sequence-native throughout, single dataset**: no single-image dataset in the primary audit
  (Wang et al. is citation-only now). Fire-blight sequences are variable-length and gapped —
  9–15 real timepoints; T=15 (days post-inoculation) is the padded upper bound (see Dataset).
- **Mamba: default to GRU fallback**, not the CUDA-only mamba_ssm package — declared explicitly
  in the paper as a substitution. Revisit real Mamba only if there's time to spare later.
- **Ring vs open CNOT chain**: implement both behind a config flag (`entangle_topology:
  open|ring`), default open, run both once QMSAN Fig. 5 is checked (see Open Items).
- **Quantum circuit is fixed (non-trainable)** by current spec — zero trainable quantum
  parameters, which makes Arms B/D/E parameter-matched by construction. Do not add trainable
  rotation layers without an explicit go-ahead (would shift Arm B's parameter target).

## The full pipeline (Stages 1-7 + 3 heads)
1. **Adaptive preprocessing** — radiometric correction is **already applied by the camera**
   (files are `REFLECTANCE_*`, Spectralon-referenced); there are no white/dark cubes, so this
   sub-step is a no-op for this dataset — do not re-normalise. Keep only geometric correction
   (align every frame to the plant's first day via phase cross-correlation) plus a fixed
   per-cube scale/clip if needed for the specular >1 pixels. Fixed, not learned — must be
   identical across every arm so it can't itself create a difference.
2. **Foundation encoder (DINOv2, frozen)** — per-frame patch grid, N × 384-d. Cache to disk,
   keyed on a hash of (preprocessing config, encoder name, band-reduction config).
3. **Lesion graph construction** — one node per lesion region (disease index, Otsu threshold,
   skimage connected components), node features = pooled patch features + regionprops
   geometric features, edges = k-NN (k=4) via torch_geometric.knn_graph. Two inputs: mask from
   Stage 1's output, features from Stage 2's output. Lesions are NOT tracked across days —
   each day's graph built independently (merging makes cross-day identity ill-defined by ~day
   10). Edge case: zero-lesion frames (healthy/early control) get a single zero-feature
   "null lesion" node with a self-loop so the graph is always valid — do not skip this case.
   The `pos_p<N>.csv` files are too sparse/partial (6 plants, symptom onset only, none for
   plant 5) to drive masking directly — use them to sanity-check the Otsu-derived lesion
   masks where they exist and as the severity signal, not as the Stage 3 mask input.
4. **Graph-Mamba reasoning** — 2× GATv2 layers (node message passing) → mean+max pool per day
   → sequence model (Mamba or GRU fallback) over the T daily vectors → one 512-d summary.
   Elapsed time Δt (actual days since last photo, not step index) feeds the state update.
   Δt is computed from the `.hdr` `acquisition date` fields per plant (see Dataset); the
   sequences have real gaps (SS1 missing days 9–11, SS3 missing days 3/12/13).
5. **Classical prototype + quantum circuit (parallel)** — classical: dense 512→6 + tanh.
   Quantum: 6-qubit circuit (PennyLane, default.qubit, torch interface, backprop) — see below.
6. **Residual fusion**: p = p_c + α·q, α from config. α=0 must exactly reproduce a pure-classical
   forward pass (this is THE test that makes the audit valid — write it as an automated test,
   not a manual check).
7. **Uncertainty-guided refinement**: loop while uncertainty ε ≥ threshold τ (fixed across arms).
   Log number of refinement passes per run alongside accuracy — it's a variance source in a
   study measuring small differences.
8. **Cross-domain calibration**: per-capture-session rescaling. OFF by default (single-instrument
   study) — switch on only for a dedicated cross-session evaluation, state which is which.

Prediction heads (read off the same calibrated 6-d prototype):
- Disease class (binary infected/healthy — single pathogen in this data, no multi-class head)
- Severity: dial readout, anchor angles θ_ℓ = ℓ·π/4 for ℓ∈{0,1,2,3}, (u,v)=W_d·p+b_d,
  θ̂=atan2(v,u), nearest anchor wins. Loss L_sev = 1 − cos(θ̂ − θ*), implemented via
  1 − (u·cosθ* + v·sinθ*)/√(u²+v²) so atan2 never appears in the backward pass.
- Progression: extrapolate the dial trajectory (days 1..T) to day T+k. Exact extrapolation
  model NOT YET DECIDED (see Open Items) — flag before implementing, don't default silently.
  Target severity per timepoint is the DERIVED label from symptomatic-pixel-count growth
  (see Dataset — bin thresholds still to be decided).

## Quantum circuit (Stage 5) — full spec
6 qubits, PennyLane `default.qubit`, `interface="torch"`, `diff_method="backprop"`.
1. Angle embedding: `qml.AngleEmbedding(x, wires=range(6), rotation="Y")` — data-dependent,
   not trainable.
2. Ordinal gates: fixed `qml.RX(phi[i], wires=i)`, phi indexed by WIRE not severity label,
   stored as a torch buffer (not nn.Parameter) so the optimizer cannot touch it.
3. Entangling layer: CNOT chain 0→1→2→3→4→5 (or ring, see config flag above) — the ONLY gate
   that lets qubits interact; this is what Arm D removes (replace with identity, everything
   else bit-for-bit identical to Arm E).
4. Measurement: Pauli-Z expectation per qubit → 6 values in [-1, 1].

Closed-form unit tests (implement the circuit against these before trusting it):
- No entanglement (Arm D): z_i = cos(φ_i)·cos(x_i), independent per qubit.
- With chain (Arm E): q_k = ∏_{i=1}^{k} cos(φ_i)·cos(x_i) — product of everything upstream.

## The 5 arms
Config-driven, ONE model class, differing in exactly `alpha`, `use_entanglement`, `head` —
write a test that diffs every arm config against `base.yaml` and fails if any other key differs.
Stages 1-4 must be byte-identical across all five arms.

| Arm | What | Tells us |
|---|---|---|
| A | Classical CNN, flat 4-class | The floor |
| B | Classical, ordinal, param-matched to E | The real rival — if B≈E, quantum adds nothing |
| C | Quantum, flat output | Checks whether published claims replicate |
| D | Quantum, entanglement removed (identity instead of CNOT chain) | The key control |
| E | QOrd-Net — quantum + ordinal dial | The full system |

## Metrics
MAE in severity levels, quadratic weighted kappa (κ_w, weights (i−j)²/(K−1)², K=4), confusion
matrix, refinement-pass count (Stage 7). Lead with MAE/kappa on the 4-stage task, not binary
accuracy — more headroom to detect an effect than infected/healthy accuracy.

## Repo conventions
- One entry point: `python -m src.train --config configs/arm_e.yaml --seed 0`.
- `set_seed()` covering python/numpy/torch; `torch.use_deterministic_algorithms(True)`.
- Every run writes to `runs/<arm>/<seed>/`: config snapshot, git commit hash, metrics JSON,
  per-epoch log.
- Plant-level train/val/test split saved once as a JSON of plant IDs — never recomputed
  per-run, never allowed to drift between runs.
- Raw data, feature cache, and venv are gitignored — never `git add` these.

## Implementation order — work in this order, stop and report at each checkpoint
1. Repo scaffold + harness (config diff test, seeding, run logging) — no model code yet.
2. Vertical slice on fake/random data: every stage stubbed with correctly-shaped random
   tensors, all 5 arms run end-to-end on ~20 synthetic samples. Must pass before touching
   real data: (a) alpha=0 bit-identical to pure-classical forward, same seed; (b) Arms D and
   E identical to each other at alpha=0.
3. Real components, in this order:
   a. M6 quantum circuit (fully independent, unit-test against closed forms above)
   b. M1 dataset + plant-level split (structure now known — see Dataset above; ENVI BIL
      float32 reader, derived severity label, per-plant split)
   c. M2 preprocessing + M3 encoder + caching
   d. M9 heads + metrics
   e. **CHECKPOINT: Arms A and B end-to-end, 10 seeds, measure σ, run the power check above.
      Stop and report before continuing — this decides whether the study is viable as scoped.**
   f. M4 lesion graph (test the null-lesion-node case explicitly)
   g. M5 GATv2 + sequence encoder (GRU fallback)
   h. M7 fusion → Arms C/D/E
   i. M8 refinement + calibration, both OFF by default
4. Pre-register the analysis plan (comparisons, metric, test, seed count) as a committed file
   BEFORE running the full Arms C/D/E sweep.

Do not skip ahead to a later module because it seems easy — the dependency order exists so
that a bug is caught at the smallest possible stage.

## Open items — ASK before deciding, don't assume
- Ring vs open CNOT chain: check QMSAN Fig. 5 directly first.
- Progression-head extrapolation model: linear fit on recent angle trajectory vs. a small
  learned forecaster — not decided.
- Pooling to n≈23 would require downloading the July 2022 group (not in hand) AND checking its
  protocol matches (25 cm vs 50 cm camera distance, different inoculum concentration, weekday-
  only cadence). Deferred — audit runs on the March group (n=13) unless/until July is fetched.
- Severity bin thresholds for the derived ordinal label (raw pixel count vs leaf-area-
  normalised; cut points for ℓ∈{0,1,2,3}; handling of pre-symptom inoculated days and plant 5)
  — decide before implementing the severity head.
- Band-reduction method for feeding 204-band cubes into DINOv2 (fixed selection vs PCA;
  earlier reasoning favored fixed/deterministic band selection to avoid train-only-fitting
  complications) — confirm before implementing Stage 2's input adapter.
