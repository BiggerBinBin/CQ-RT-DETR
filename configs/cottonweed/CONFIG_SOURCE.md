# Configuration provenance

The three configurations in this directory were copied directly from the
formal experiment configurations used in the local RT-DETRv2 project.

| Experiment | Repository configuration | Original local configuration |
|---|---|---|
| Original RT-DETRv2 | `A_RTDETRv2.yml` | `configs/weedlab_formal/A_baseline.yml` |
| B_Lite | `B_Lite.yml` | `configs/weedlab_formal/E_lite3_cb.yml` |
| CQ-RT-DETR + SR-TCCM | `CQ_RTDETR_SR_TCCM.yml` | `configs/weedlab_formal/sr_tccm_experiments/S2_scale_reliable_tccm.yml` |

B_Lite and CQ-RT-DETR use the same compressed inference architecture.
CQ-RT-DETR additionally enables SR-TCCM during training.
