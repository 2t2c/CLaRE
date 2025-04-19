#!/bin/bash

# Set root path
ROOT="$TMPDIR/dataset/progan"
mkdir -p "$ROOT/train" "$ROOT/val" "$ROOT/test"
cd "$ROOT/train" || exit 1

# Dataset 1
# Paper: CNN-generated images are surprisingly easy to spot...for now
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007 &
wait $(jobs -p)

7z x progan_train.7z.001
rm progan_train.7z.*
unzip progan_train.zip
rm progan_train.zip

# validation set
cd "$ROOT/val" || exit 1
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip
rm progan_val.zip

# test set
cd "$ROOT/test" || exit 1
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip
unzip CNN_synth_testset.zip
rm CNN_synth_testset.zip

