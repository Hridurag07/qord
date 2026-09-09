# working notes

## what this is
A GRU(Mamba fallback) + trainable quantum circuit model for apple fire-blight
severity over time. Severity is an ordinal 0-3 "dial" plus a forward-in-time
forecast. The quantum branch is a variational circuit with data re-uploading;
`use_quantum=False` is the ablation.

## data (pooled, both temporal campaigns)
- March 2021 (Chantecler): 13 plants (7 inoculated, 6 control)
- July 2022 (Gala): 10 temporal plants (3 inoculated, 7 control) - water-stress
  arm (plants 11-17) dropped, and the short 2.2-2.5 leaves dropped
- May 2022 (orchard, single visit, non-temporal) - not used
- 23 plants total. same camera / 204 bands / 397-1004 nm, but different variety +
  inoculum + camera distance -> pooled anyway, noted as a limitation.
- severity = lesion pixels / plant pixels (area fraction), cumulative max, then
  bins (0.001, 0.003, 0.02). area fraction so March and July are comparable.
  controls forced to 0. plant march_5 excluded (inoculated, never labelled).
- split by plant, stratified by (campaign, status): train 13 / val 4 / test 6.

## code
- config.py - one CONFIG dict, config(**overrides)
- data.py   - ENVI read, campaigns + combined index, severity, split, preprocess,
              DINOv2, lesion graph, feature cache
- model.py  - trainable quantum circuit (StronglyEntanglingLayers + data
              re-uploading) + QOrdNet + loss + metrics
- train.py  - train / evaluate / quantum on-off ablation
- tests.py

## pipeline
align frames -> frozen DINOv2 patch tokens -> per-frame lesion graph -> GATv2 ->
GRU over days (with delta_t) -> classical branch + quantum branch -> fuse ->
refine -> heads (disease / severity dial / progression line-fit).

## how to run
python -m pytest tests.py
python data.py                       # split + cache (once, ~25 min)
python train.py --seed 0 --march
python train.py --ablation --seeds 12

## status / results
- see docs/results.md (updated after each ablation run).
- on the first 13-plant version a trainable circuit hurt (on-off d=+1.4). pooling
  to 23 plants to see if that holds.

## open / next
- progression eval: currently predicts the held-out last step, which persistence
  already nails because severity plateaus. forecast the rising phase instead.
- severity range is thin: only july_2.1 reaches level 3. real spread needs more
  severe cases.
- add a campaign indicator to the model? (domain shift March vs July)
- May 2022 as an out-of-distribution field test
- LOPO-CV for tighter CIs
