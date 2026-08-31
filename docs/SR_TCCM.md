# SR-TCCM

SR-TCCM is a training-only query calibration mechanism applied to the
compressed RT-DETRv2 detector.

It does not add an inference-time Backbone, Hybrid Encoder, Transformer
Decoder, or detection-head module.

The method contains four stages:

1. Persistent Competitive Query Mining
2. Cross-Layer Trajectory Stability and Competitive Persistence
3. Scale Reliability and Winner Localization Advantage
4. Budgeted Adaptive Margin Ranking Loss

The final-layer one-to-one Hungarian matched query is used as the winner
anchor. Unmatched queries that repeatedly compete for the same ground-truth
object across decoder layers are mined and calibrated.

SR-TCCM does not change the original Hungarian assignment and does not turn
competitive queries into additional positive targets.

For exact hyperparameters, use:

```text
configs/cottonweed/CQ_RTDETR_SR_TCCM.yml
```

For the exact implementation actually exported from the experiment project,
inspect the files listed by:

```bash
bash scripts/show_sr_tccm_code.sh
```
