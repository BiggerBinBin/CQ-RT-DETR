# Dataset preparation

The code expects a COCO-format object-detection dataset.

A typical layout is:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── instances_train.json
    ├── instances_train_cb.json
    ├── instances_val.json
    └── instances_test.json
```

For the class-balanced experiments, `instances_train_cb.json` represents the
repeated-sampling training sequence.

Validation and test sets must keep their original distributions and must not
be repeatedly sampled.

Before training, edit the dataset paths in:

- `configs/cottonweed/A_RTDETRv2.yml`
- `configs/cottonweed/B_Lite.yml`
- `configs/cottonweed/CQ_RTDETR_SR_TCCM.yml`

so that they point to the dataset on your machine.
