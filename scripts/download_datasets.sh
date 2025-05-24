#!/bin/bash

# load modules
module load 2024
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


# Datset 3: DF40 (test set, 40 .zip files in total)
cd "$ROOT" || exit 1
mkdir -p "$ROOT/df40/train" "$ROOT/df40/val" "$ROOT/df40/test"
cd "$ROOT/df40/test" || exit 1
FOLDERID=1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_
# follow this: https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work
ACCESS_TOKEN="<access-token>"

# step 1: save the files inside the folder to a .json object
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q='$FOLDERID'+in+parents+and+trashed=false&fields=files(id,name,mimeType)" \
  -o df40_test_files.json

# step 2: download each .zip file iteratively by reading the saved .json mappings

# single file download (example)
# FILEID=1Y15vpgltFD1amMEABCRkOiDEK-aQpRJ8
# FILENAME=DiT
# curl -H "Authorization: Bearer ${ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/${FILEID}?alt=media -o ${FILENAME}.zip

jq -r '.files[] | select(.mimeType != "application/vnd.google-apps.folder" and (.name | endswith(".zip"))) | [.id, .name] | @tsv' df40_test_files.json | \
while IFS=$'\t' read -r FILEID FILENAME; do
  echo "Downloading $FILENAME $FILEID"
  curl -H "Authorization: Bearer ${ACCESS_TOKEN}" \
       -L "https://www.googleapis.com/drive/v3/files/${FILEID}?alt=media" \
       -o "${FILENAME}"
done

# step 3: unzipping
for file in *.zip; do
  echo "Unzipping $file"
  unzip -o "$file"
done
# e4e.zip -> 7z x -y file.zip


# Datset 3: DF40 (train set, 31 .zip files in total)
cd "$ROOT/df40/train" || exit 1
FOLDERID=1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD
# follow this: https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work
ACCESS_TOKEN="<access-token>"

# step 1: save the files inside the folder to a .json object
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q='$FOLDERID'+in+parents+and+trashed=false&fields=files(id,name,mimeType)" \
  -o df40_train_files.json

# step 2: download each .zip file iteratively by reading the saved .json mappings
# single file: curl -H "Authorization: Bearer ${ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/${FILEID}?alt=media -o ${FILENAME}.zip

jq -r '.files[] | select(.mimeType != "application/vnd.google-apps.folder" and (.name | endswith(".zip"))) | [.id, .name] | @tsv' df40_train_files.json | \
while IFS=$'\t' read -r FILEID FILENAME; do
  echo "Downloading $FILENAME $FILEID"
  curl -H "Authorization: Bearer ${ACCESS_TOKEN}" \
       -L "https://www.googleapis.com/drive/v3/files/${FILEID}?alt=media" \
       -o "${FILENAME}"
done

# step 3: unzipping
for file in *.zip; do
  echo "Unzipping $file"
  unzip -o "$file"
done

# step 4: dataset configs
cd "$ROOT/df40" || exit 1
mkdir -p "$ROOT/df40/configs"
FOLDERID=19VhAL4aDJOKvhl9stEq_ymFeHiXo6_j-
# gdown --folder https://drive.google.com/drive/folders/$FOLDERID --remaining-ok

# step 4.1: save the files inside the folder to a .json object
curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://www.googleapis.com/drive/v3/files?q='$FOLDERID'+in+parents+and+trashed=false&fields=files(id,name,mimeType)" \
  -o df40_config_files.json

# step 4.2: download configs (82 files in total)
cd "$ROOT/df40/configs" || exit 1
jq -r '.files[] | select(.mimeType != "application/vnd.google-apps.folder" and (.name | endswith(".json"))) | [.id, .name] | @tsv' ../df40_config_files.json | \
while IFS=$'\t' read -r FILEID FILENAME; do
  echo "Downloading $FILENAME"
  curl -H "Authorization: Bearer ${ACCESS_TOKEN}" \
       -L "https://www.googleapis.com/drive/v3/files/${FILEID}?alt=media" \
       --progress-bar -o "${FILENAME}"
done


# Datset 4: FaceForensics++
cd "$ROOT" || exit 1
mkdir -p "$ROOT/face_forensics"
cd "$ROOT/face_forensics" || exit 1
FILEID=1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb
gdown https://drive.google.com/uc?id=$FILEID
unzip FaceForensics++_real_data_for_DF40.zip


# Datset 5: Celeb-DF
cd "$ROOT" || exit 1
mkdir -p "$ROOT/celeb_df"
cd "$ROOT/celeb_df" || exit 1
FILEID=1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw
gdown https://drive.google.com/uc?id=$FILEID
unzip Celeb-DF-v2_real_data_for_DF40.zip
