#!/bin/bash

# load modules
module load p7zip/17.05-GCCcore-13.3.0

# Set root path
#ROOT="$TMPDIR/datasets" # does not work with validate.job due to mkdir dynamics
ROOT="/scratch-shared/scur0555/datasets" # working
mkdir -p "$ROOT/cnn_detection/train" "$ROOT/cnn_detection/val" "$ROOT/cnn_detection/test"

# Dataset 1: CNN-generated images are surprisingly easy to spot...for now

# train set: skip
#cd "$ROOT/train" || exit 1
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001 &
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002 &
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003 &
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004 &
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005 &
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006 &
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007 &
#wait $(jobs -p)
#7z x progan_train.7z.001
#rm progan_train.7z.*
#unzip progan_train.zip
#rm progan_train.zip

# validation set: skip
#cd "$ROOT/val" || exit 1
#wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip
#unzip progan_val.zip
#rm progan_val.zip

# test set
cd "$ROOT/cnn_detection/test" || exit 1
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip
unzip CNN_synth_testset.zip
rm CNN_synth_testset.zip

# Dataset 2: Diffusion LDM/Glide
cd "$ROOT" || exit 1
pip install gdown --quiet
FILEID=1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t
gdown https://drive.google.com/uc?id=$FILEID
unzip diffusion_datasets.zip
rm diffusion_datasets.zip