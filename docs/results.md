# results

## setup
- 23 temporal plants: March 2021 (13) + July 2022 (10, water-stress dropped).
- split by plant: train 13 / val 4 / test 6.
- severity = lesion-area fraction, cumulative max, bins (0.001, 0.003, 0.02).
- model: DINOv2 -> lesion graph -> GATv2 -> GRU -> classical branch + trainable
  quantum branch (StronglyEntanglingLayers, 8 qubits, 3 data-reuploading blocks)
  -> fuse -> heads.
- ablation: quantum on vs off, everything else identical, 12 paired seeds,
  val used during training, test scored once.

## quantum on/off ablation (12 seeds, pooled 23 plants)

|              | severity MAE | severity QWK |
|--------------|--------------|--------------|
| quantum ON   | 1.42 +- 0.75 | 0.01 +- 0.48 |
| quantum OFF  | 1.38 +- 0.46 | -0.12 +- 0.22 |

on - off MAE per-plant paired diff: **+0.042**, 95% CI **[-0.222, 0.250]**, d = 0.12.

**Read:** no measurable difference. The quantum branch neither helps nor hurts
severity accuracy at this scale. QWK is a touch higher with quantum on (0.01 vs
-0.12) but with much larger variance - not something to lean on.

## history
- First pass, 13 plants only (7 training): the trainable circuit *hurt* -
  on-off d = +1.4, CI excluding zero. Almost certainly overfitting: extra
  trainable parameters on 7 training sequences.
- Pooling to 23 plants: that negative effect disappears (d = +0.12, CI crosses
  zero). More data removed the harm but did not turn it into a benefit.

## the real limits
- **Severity barely learns at all**: QWK near 0 for both on and off, MAE ~1.4 on
  a 0-3 scale. There isn't a strong model here for the quantum part to improve.
- **Thin severity range**: only 1 of 6 test plants (july_2.1) reaches level 3;
  most infected plants top out at level 1-2. Fire blight in these controlled
  experiments makes small lesions.
- **6 test plants** -> every CI is wide.
- March vs July is a domain shift (variety, inoculum, camera distance) that the
  pooled model has to absorb.

## honest bottom line
The trainable quantum circuit was built and evaluated properly. On the data
available it makes no measurable difference to severity prediction. Getting the
quantum part to demonstrably help would require, at minimum, a base model that
actually predicts severity well (so there is headroom), a wider severity range,
and more test plants for usable CIs.
