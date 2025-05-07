# imports
import wandb
import yaml
from src.dataset import UFD, DF40, describe_dataloader
from models import get_model
from dataset_paths import DATASET_PATHS
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score, accuracy_score
from scipy.ndimage import gaussian_filter
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data
import torch
import numpy as np
from io import BytesIO
from copy import deepcopy
import shutil
import random
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import os
os.environ["WANDB__SERVICE_WAIT"] = "300"


# logging
# wandb.login()

# global variables
SEED = 0
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    # ranging from 0-95, 75 is default
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def validate(model, loader, find_thres=False, dataset_name=None):
    with torch.no_grad():
        y_true, y_pred = [], []
        # print("Length of dataset: %d" % (len(loader)))
        if dataset_name == "df40":
            for batch in loader:
                img, label = batch["image"], batch["label"]
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())
        else:
            for img, label in loader:
                in_tens = img.cuda()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== #
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #

    # Get AP
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

def main(args):
    # load model weights
    model = get_model(args.arch)
    # load linear classifier weights
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print("Model loaded...")
    model.eval()
    model.cuda()

    if (args.real_path == None) or (args.fake_path == None) or (args.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [dict(real_path=args.real_path,
                              fake_path=args.fake_path, data_mode=args.data_mode)]

    # creating wandb session
    dataset_type = args.real_path.split('/')[-3]
    dataset_name = args.real_path.split('/')[-1]
    experiment_name = dataset_type + '_' + dataset_name
    if args.run_name is not None:
        experiment_name = args.run_name + '_' + experiment_name
    wandb.init(
        project="debugging",
        entity="FoMo",
        name=experiment_name + "-" +
        str(datetime.now(ZoneInfo("Europe/Amsterdam"))),
        config={
            "architecture": args.arch,
            "ckpt": args.ckpt,
            "batch_size": args.batch_size,
            "result_folder": args.result_folder,
            "seed": SEED,
            "dataset_type": dataset_type,
            "dataset_name": dataset_name,
        },
        settings=wandb.Settings(_service_wait=300, init_timeout=120))

    # set seed for deterministic results
    set_seed()
    
    # loading dataset
    if args.dataset == "ufd":
        dataset = UFD(
            args.real_path,
            args.fake_path,
            args.data_mode,
            args.max_sample,
            args.arch,
            jpeg_quality=args.jpeg_quality,
            gaussian_sigma=args.gaussian_sigma,)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == "df40":
        # load the config file
        with open(args.df40_config, 'r') as f:
            config = yaml.safe_load(f)
        if args.df40_name is not None:
            config['test_dataset'] = args.df40_name
        dataset = DF40(config=config, mode='test')
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
    describe_dataloader(loader)
    
    # run validation
    ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader,
                                                                          find_thres=True,
                                                                          dataset_name=args.dataset)

    # log metrics
    metrics = {
        "average_precision": ap,
        "real_accuracy_0.5": r_acc0,
        "fake_accuracy_0.5": f_acc0,
        "accuracy_0.5": acc0,
        "real_accuracy_best": r_acc1,
        "fake_accuracy_best": f_acc1,
        "accuracy_best": acc1,
        "best_threshold": best_thres
    }
    wandb.log(metrics)

    # export metrics
    export_path = os.path.join(
        args.result_folder, f"{dataset_type}_metrics.json")

    # read existing or create new
    if os.path.exists(export_path):
        with open(export_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # update by dataset_name key
    all_metrics[dataset_name] = metrics

    # save updated
    with open(export_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    # exit session
    wandb.finish()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str,
                        choices=['ufd', 'df40'], required=True)
    
    # wandb arguments
    parser.add_argument('--project', type=str,
                        default="debugging", help='wandb run name')
    parser.add_argument('--run_name', type=str,
                        default="debug", help='wandb run name')
    
    # add UFD-specific arguments
    parser.add_argument('--real_path', type=str,
                        default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str,
                        default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str,
                        default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000,
                        help='only check this number of images for both fake/real')
    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str,
                        default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--jpeg_quality', type=int, default=None,
                        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None,
                        help="0,1,2,3,4. Used to test robustness of our model. Not apply if None")
    
    # add DF40-specific config path
    parser.add_argument("--df40_name", type=str, default=None,
                        help="DF40 dataset name")
    parser.add_argument('--df40_config', type=str,
                        default="./configs/df40/test_config.yaml")
    
    # generic params
    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    # create result directories
    # if os.path.exists(args.result_folder):
    #     shutil.rmtree(args.result_folder)
    os.makedirs(args.result_folder, exist_ok=True)

    # call the main function
    main(args)