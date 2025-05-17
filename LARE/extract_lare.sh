#python feature_extractor.py \
#--input_path /home/petterluo/data/aigc_data/GenImage/anns/train_sd1d5.txt \
#--output_path /home/petterluo/project/FakeImageDetection/outputs/train_sd1d5 \
#--t 261 \
#--prompt '' \
#--ensemble_size 8 \
#--pretrained_model_name_or_path "/home/petterluo/huggingface_models/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819" \
#--img_size 256 256 \
#--use_prompt_template \
#--n-gpus 8

#strings=("train_adm" "") # , ,"","","","","","","","","")
#strings=("train_biggan" "train_glide" "train_imagenet_biggan")
#strings=("train_imagenet_glide" "train_imagenet_mj" "train_imagenet_sd1d4")
#strings=("train_imagenet_vqdm" "train_imagenet_wukong" "train_mj")
#strings=("train_sd1d4" "train_vqdm" "train_wukong")
#strings=()

# fix
#strings=("train_adm" "train_imagenet_adm" "train_sd1d5" "train_imagenet_sd1d5" )
#strings=("train_biggan" "train_imagenet_biggan" "train_glide" "train_imagenet_glide" )
#strings=("train_sd1d4" "train_imagenet_sd1d4" "train_vqdm" "train_imagenet_vqdm" )
#strings=("train_wukong" "train_imagenet_wukong" "train_mj" "train_imagenet_mj" )
strings=("val_mj" "val_imagenet_mj" "val_adm" "val_imagenet_adm" "val_biggan" "val_imagenet_biggan" "val_glide" "val_imagenet_glide" "val_sd1d4" "val_imagenet_sd1d4" "val_vqdm" "val_imagenet_vqdm" "val_wukong" "val_imagenet_wukong")
for string in "${strings[@]}"; do
  python extract_lare.py \
    --input_path /home/petterluo/data/aigc_data/GenImage/anns_newceph/${string}.txt \
    --output_path /home/petterluo/project/FakeImageDetection/outputs/${string} \
    --t 200 \
    --prompt 'a photo' \
    --ensemble_size 4 \
    --pretrained_model_name_or_path "/home/petterluo/pretrained_models/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819" \
    --img_size 256 256 \
    --use_prompt_template \
    --n-gpus 8
done
