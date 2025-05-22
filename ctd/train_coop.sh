#!/bin/bash

python train.py \
  --root data \
  --seed 17 \
  --trainer CoOp \
  --dataset-config-file configs/datasets/df40.yaml \
  --config-file configs/trainers/CoOp/vit_l14_ep2.yaml \
  --output-dir train_outputs/coop_100k_2epochs
