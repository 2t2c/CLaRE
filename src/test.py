"""
Script to test the trained architectures' pipeline.
"""

import os
import sys

os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import get_model
from datetime import datetime
import json
from datetime import timedelta
from zoneinfo import ZoneInfo
import time
from tqdm import tqdm
import yaml
import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.data.distributed
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from rich import print as rprint
import wandb
from loss import LabelSmoothingLoss
from utils import set_seed, get_device, display_metrics, display_model_summary, load_config, display_args
from dataset import describe_dataloader, CTD, LARE
from yacs.config import CfgNode as CN
import logging


# fetch logger
logger = logging.getLogger("fomo_logger")

# DF40 test only
TEST_DATASETS = ["heygen", "MidJourney", "whichisreal", "stargan", "starganv2", "styleclip", "CollabDiff"]

def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k. Yoinked from DASSL.

    :param:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    :returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def search_best_acc(gt_labels, pred_probs):
    best_acc = -1
    best_threshold = -1
    acc_dict = {}
    for thresh in sorted(pred_probs.tolist()):
        pred_probs_copy = np.array(pred_probs)
        pred_probs_copy[pred_probs_copy > thresh] = 1
        pred_probs_copy[pred_probs_copy <= thresh] = 0
        acc = accuracy_score(gt_labels, pred_probs_copy)
        acc_dict[thresh] = acc
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    return best_acc, best_threshold


def test_contrastive(model, data_loader, module, device):
    """
    Testing function for the model.
    """
    logger.info('Testing Started!')
    model.eval()

    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    pbar = tqdm(data_loader, desc=f"Testing", unit="batch")

    with torch.no_grad():
        for batch in pbar:
            if module in ["fusion", "lare"]:
                images, labels, loss_maps = batch
                images = images.to(device)
                labels = labels.flatten().squeeze().to(device)
                loss_maps = loss_maps.to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            try:
                if module in ["fusion", "lare"]:
                    logits = model(images, loss_maps)
                else:
                    logits = model(images)
                prob = torch.softmax(logits, dim=-1)
            except:
                logger.warning('Bad evaluation batch!', exc_info=True)
                raise
            gt_labels_list.append(labels)
            prob_labels_list.append(prob[:, 1])

    # gather ground truth labels and predicted labels
    gt_labels = torch.cat(gt_labels_list, dim=0)
    prob_labels = torch.cat(prob_labels_list, dim=0)
    gt_labels_list = gt_labels.cpu().numpy()
    prob_labels_list = prob_labels.cpu().numpy()

    fpr, tpr, thres = roc_curve(gt_labels_list, prob_labels_list)
    thresh = thres[len(thres) // 2]
    logger.info(f'Thresh Old: {thresh}')
    precision, recall, thresholds = precision_recall_curve(gt_labels_list, prob_labels_list)
    f_score = precision * recall / (precision + recall)
    thresh = thresholds[np.argmax(f_score)]
    # thresh = 0.5
    logger.info(f'Thresh New: {thresh}')

    pred_labels_list = np.array(prob_labels_list)
    pred_labels_list[pred_labels_list > thresh] = 1
    pred_labels_list[pred_labels_list <= thresh] = 0

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)

    # best thresh accuracy
    best_acc, best_thresh = search_best_acc(gt_labels_list, prob_labels_list)
    logger.info(f'Search ACC: {best_acc}, Search Thresh: {best_thresh}')
    r_acc = accuracy_score(gt_labels_list[gt_labels_list == 0], prob_labels_list[gt_labels_list == 0] > best_thresh)
    f_acc = accuracy_score(gt_labels_list[gt_labels_list == 1], prob_labels_list[gt_labels_list == 1] > best_thresh)

    # raw accuracy
    raw_acc = accuracy_score(gt_labels_list, prob_labels_list > 0.5)
    raw_r_acc = accuracy_score(gt_labels_list[gt_labels_list == 0], prob_labels_list[gt_labels_list == 0] > 0.5)
    raw_f_acc = accuracy_score(gt_labels_list[gt_labels_list == 1], prob_labels_list[gt_labels_list == 1] > 0.5)

    return auc, ap, best_acc, r_acc, f_acc, raw_acc, raw_r_acc, raw_f_acc, best_thresh


def test(args):
    # load the config file
    with open(args.config, 'r') as f:
        config_file = yaml.safe_load(f)
    # convert to yacs
    cfg = load_config(config_file)
    # add args inside cfg as CfgNode
    cfg.args = CN(vars(args))
    # change args
    cfg.dataset.subset = []
    if args.test_ratio:
        cfg.dataset.frame_num.test = args.test_ratio
    # dump the config
    with open(f"{args.log_dir}/config.yaml", "w") as f:
        f.write(cfg.dump())
    # pretty print args
    display_args(args, title="Config Arguments")

    # setup wandb
    if args.logging:
        train_uid = args.checkpoint.split("/")[-1]
        wandb.init(
            project=args.project,
            entity="FoMo",
            name=args.run_name + "/" + train_uid,
            config={
                "uid": args.uid,
                "train_uid": train_uid,
                "architecture": args.model,
                "clip_type": args.clip_type,
                "batch_size": args.batch_size,
                "log_dir": args.log_dir,
                "seed": args.seed,
                "mode": args.mode,
                "device": args.device,
                "test_datasets": args.test_datasets,
                "test_ratio": cfg.dataset.frame_num.test,
                "checkpoint": args.checkpoint,
            },
            settings=wandb.Settings(_service_wait=300, init_timeout=120))

    # load model
    model = get_model(name=args.model, clip_type=args.clip_type, cfg=cfg)
    device = get_device(args.device)
    model.to(device)
    # load model
    checkpoint_path = os.path.join(args.checkpoint, "best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Model loaded from checkpoint - {args.checkpoint}")
    display_model_summary(model, input_shape=(1, 3, cfg.dataset.resolution,
                                              cfg.dataset.resolution), device=device)

    # start testing
    logger.info(f'Testing Datasets - "{args.test_datasets}"')
    for dataset in args.test_datasets:
        try:
            # change dataset name
            cfg.dataset.subset = []
            if dataset not in TEST_DATASETS:
                cfg.dataset.test_dataset = dataset + "_ff"
            else:
                cfg.dataset.test_dataset = dataset
            # load validation data
            logger.info(f"Testing on '{dataset}'")
            if args.module == "clipping":
                test_dataset = CTD(config=cfg.dataset,
                                mode="test",
                                jpeg_quality=cfg.dataset.jpeg_quality,
                                debug=args.debug)
            elif args.module in ["fusion", "lare"]:
                test_dataset = LARE(config=cfg.dataset,
                                mode="test",
                                jpeg_quality=cfg.dataset.jpeg_quality,
                                debug=args.debug)
            else:
                logger.error("Invalid module. Choose 'lare', 'clipping', or 'fusion'.")
                return
            test_data_loader = DataLoader(
                test_dataset, args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=None)
            describe_dataloader(test_data_loader, title="Test DataLoader Summary")
        
            auc, ap, best_acc, r_acc, f_acc, raw_acc, raw_r_acc, raw_f_acc, best_thresh = test_contrastive(model, test_data_loader,
                                                                                                        args.module, device)
        
            # log metrics
            metrics = {
                f"roc_auc/{dataset}": auc,
                f"average_precision/{dataset}": ap,
                f"accuracy/{dataset}": best_acc,
                f"real_accuracy/{dataset}": r_acc,
                f"fake_accuracy/{dataset}": f_acc,
                f"raw_accuracy/{dataset}": raw_acc,
                f"raw_real_accuracy/{dataset}": raw_r_acc,
                f"raw_fake_accuracy/{dataset}": raw_f_acc,
                f"best_thresh/{dataset}": best_thresh
            }
            display_metrics(metrics, title=f"{dataset} Test Metrics")
            wandb.log(metrics)
        
            # export metrics
            export_path = os.path.join(
                args.log_dir, f"metrics.json")
        
            # read existing or create new
            if os.path.exists(export_path):
                with open(export_path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}
        
            # update by dataset_name key
            all_metrics[dataset] = metrics
        
            # save updated
            with open(export_path, "w") as f:
                json.dump(all_metrics, f, indent=4)
        except Exception as e:
            logger.error(e)
    
    if args.logging:
        # exit session
        wandb.finish()