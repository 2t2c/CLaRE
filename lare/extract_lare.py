# imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join("..")))
from src.utils import display_args
from rich import print as rprint
import argparse
import json
import multiprocessing as mp
import os
from os.path import join as opj
from pathlib import Path

import PIL
# import accelerate
# import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.state import AcceleratorState
# from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
# from huggingface_hub import create_repo, upload_folder
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer



class GenImageProcessor:
    def __init__(self,
                 file_idx_to_folder='./anns/idx_to_folder.txt',
                 file_idx_to_clsname='./anns/idx_to_clsname.json',
                 filename_to_folder='./anns/filename_to_folder.json'):
        label_map_idx_to_folder = {}
        label_map_folder_to_idx = {}
        with open(file_idx_to_folder) as f:
            for line in f:
                cmd, idx, folder = line.strip().split(' ')
                idx = idx.split('/')[0]
                folder = folder.split('/')[0]
                label_map_idx_to_folder[idx] = folder
                label_map_folder_to_idx[folder] = idx

        self.label_map_idx_to_clsname = json.load(open(file_idx_to_clsname))
        self.filename_to_folder = json.load(open(filename_to_folder))

        self.label_map_folder_to_clsname = {}
        for folder, idx in label_map_folder_to_idx.items():
            self.label_map_folder_to_clsname[folder] = self.label_map_idx_to_clsname[idx]

    def __call__(self, info, use_full_name=False):
        image_path, label = info.strip().split('\t')
        filename = Path(image_path).name
        folder_or_index = filename.split('_')[0]
        if 'df40' in image_path:
            clsname_simple = 'real' if 'real' in image_path else 'fake'
            return image_path, label, filename, clsname_simple
        try:
            if len(folder_or_index) <= 3:  # is a index
                index = str(int(folder_or_index))
                clsname_full = self.label_map_idx_to_clsname[index]
            elif folder_or_index == 'ILSVRC2012':  # is val image:
                folder = self.filename_to_folder[filename]
                clsname_full = self.label_map_folder_to_clsname[folder]
            elif folder_or_index == 'GLIDE':
                index = str(int(filename.split('_')[4]))
                clsname_full = self.label_map_idx_to_clsname[index]
            elif folder_or_index == 'VQDM':
                index = str(int(filename.split('_')[4]))
                clsname_full = self.label_map_idx_to_clsname[index]
            else:  # is a folder
                # print(len(folder_or_index))
                folder = folder_or_index
                clsname_full = self.label_map_folder_to_clsname[folder]
        except Exception as e:
            print(image_path)
            raise

        if use_full_name:
            return image_path, label, filename, clsname_full
        else:
            clsname_simple = clsname_full.split(',')[0].strip()
            return image_path, label, filename, clsname_simple


def main(args, input_infos, device):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=None,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None,
    )
    # text_encoder = CLIPVisionModel.from_pretrained(
    #     args.clip_path,
    # )
    # clip_processor = AutoProcessor.from_pretrained(args.clip_path)
    print('Load VAE')
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=None
    )

    if args.dtype == "fp16":
        unet = unet.half()
        text_encoder = text_encoder.half()
        vae = vae.half()

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # type is epsilon
    print(f'noise pred type: {noise_scheduler.config.prediction_type}')
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    # clip_model = clip_model.to('cuda')
    unet = unet.to(device)

    unet.eval()
    vae.eval()
    text_encoder.eval()
    genimage_processor = GenImageProcessor()
    for info in tqdm(input_infos):
        image_path, label, filename, clsname = genimage_processor(info, use_full_name=args.use_full_clsname)
        # save_path = opj(args.output_path, filename.split('.')[0] + '.pt')
        save_path = opj(args.output_path, clsname + "_" + filename.split('.')[0] + '.pt')
        try:
            img = Image.open(image_path).convert('RGB')
        except PIL.UnidentifiedImageError:
            print(f'Bad Image {filename}')
            torch.save(torch.zeros((4, 32, 32)), save_path)
            continue

        if args.img_size[0] > 0:
            img = img.resize(args.img_size)
        img_tensor = (transforms.PILToTensor()(img) / 255.0 - 0.5) * 2
        image_sd = img_tensor.unsqueeze(0).to(device)
        # image_clip = image_clip.to(device)

        if args.dtype == "fp16":
            image_sd = image_sd.half()
            image_clip = image_clip.half()
        # print(image_sd.shape)
        # print(image_clip.shape)
        latents = vae.encode(image_sd).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        latents = latents.repeat((args.ensemble_size, 1, 1, 1))
        bsz = latents.shape[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(args.t, args.t + 1, (bsz,),
                                  device=latents.device)  # 1000
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if args.use_prompt_template:
            prompt = args.prompt_template.replace('[CLS]', clsname)
        else:
            prompt = args.prompt
        text_input = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                               return_tensors="pt"
                               ).to(device)
        encoder_hidden_states = text_encoder(text_input["input_ids"])[0]
        encoder_hidden_states = encoder_hidden_states.repeat((args.ensemble_size, 1, 1))
        # print(encoder_hidden_states.shape)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")  # bs * 4 * 32 * 32
        loss = torch.mean(loss, dim=0)  # 1 * 4 * 32 * 32
        # print(loss.shape)
        torch.save(loss.squeeze(0).cpu(), save_path)

        # noise_save_path = opj(args.output_path, 'noisze_' + filename.split('.')[0] + '.pt')
        # torch.save(target.squeeze(0).cpu(), noise_save_path)

        # v_target = noise_scheduler.get_velocity(latents, noise, timesteps)
        # noiseV_save_path = opj(args.output_path, 'noiszeV_' + filename.split('.')[0] + '.pt')
        # torch.save(v_target.squeeze(0).cpu(), noiseV_save_path)


def split_list(lst, n):
    return [lst[i::n] for i in range(n)]


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(
        description='''Extract dift from input image, and save it as torch tenosr,
                    in the shape of [c, h, w].''')

    parser.add_argument('--img_size', nargs='+', type=int, default=[512, 512],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--pretrained_model_name_or_path', default='stabilityai/stable-diffusion-2-1', type=str,
                        help='model_id of the diffusion model in huggingface')
    parser.add_argument('--t', default=280, type=int,
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=1, type=int, choices=[0, 1, 2, 3],
                        help='which upsampling block of U-Net to extract the feature map')
    parser.add_argument('--use_prompt_template', action='store_true', default=False,
                        help='use instance-wise prompt')
    parser.add_argument('--use_full_clsname', action='store_true', default=False,
                        help='use full wordnet name')
    parser.add_argument('--prompt_template', default='a photo of a [CLS]', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument('--prompt', default='', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument('--ensemble_size', default=8, type=int,
                        help='number of repeated images in each batch used to get features')
    # parser.add_argument('--input_path', type=str,
    #                     help='paths to the input image file')
    parser.add_argument('--json_path', type=str,
                        help='paths to the input json config file')
    parser.add_argument('--output_path', type=str, default='dift.pt',
                        help='path to save the outputs features as torch tensor')
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--dtype', type=str, default='fp32',
                        help='paths to the input image file')
    args = parser.parse_args()
    display_args(args)

    # load dataset
    TEST_DATA_PATH = '/scratch-shared/scur0555/datasets/df40/test/'
    json_path = args.json_path
    with open(json_path, "r") as file:
        data = json.load(file)
        dataset_name = json_path.split("/")[-1].split(".")[0]

    # with open(f"{dataset_name}.txt", "w") as file:
    input_infos = []
    for label_type in ['Real', 'Fake']:
        test_data = data[dataset_name][dataset_name + "_" + label_type]['test']
        for key in test_data:
            label = 0 if label_type == 'Real' else 1
            img_paths = test_data[key]['frames']
            for img_path in img_paths:
                input_infos.append(
                    img_path.replace("deepfakes_detection_datasets/DF40/", TEST_DATA_PATH) + " " + str(label))
                # file.write(img_path.replace("deepfakes_detection_datasets/", "") + " " + str(label) + "\n")

    # prepare
    setattr(args, 'output_path', os.path.abspath(args.output_path + f'/{dataset_name}'))
    # setattr(args, 'output_path', os.path.abspath(args.output_path))
    rprint('create folder:', args.output_path)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if args.use_prompt_template:
        rprint(f'use prompt: {args.prompt_template}')
    else:
        rprint(f'use prompt: {args.prompt}')

    # with open(args.input_path) as f:
    #     input_infos = f.readlines()

    # preprocess anns
    genimage_processor = GenImageProcessor()
    out_infos = []
    for info in tqdm(input_infos):
        image_path, label, filename, clsname = genimage_processor(info, use_full_name=args.use_full_clsname)
        # save_path = opj(args.output_path, filename.split('.')[0] + '.pt')
        save_path = opj(args.output_path, clsname + "_" + filename.split('.')[0] + '.pt')
        out_info = '\t'.join([save_path, label]) + '\n'
        out_infos.append(out_info)
    info_save_path = opj(args.output_path, 'ann.txt')
    with open(info_save_path, 'w') as f:
        f.writelines(out_infos)

    num_gpus = torch.cuda.device_count()
    assert num_gpus == args.n_gpus
    splited_infos = split_list(input_infos, num_gpus)
    # run

    # debug
    # main(args, splited_infos[0], "cuda:0")
    with mp.Pool(processes=num_gpus) as pool:
        pool.starmap(main, [(args, splited_infos[i], f"cuda:{i}") for i in range(num_gpus)])
    rprint("Successfully extract features from input images!")
