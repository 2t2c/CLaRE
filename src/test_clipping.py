"""
Script to train the CLIPping prompt-tuning architecture pipeline.
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
from dataset import describe_dataloader, CTD
from yacs.config import CfgNode as CN
import logging


# fetch logger
logger = logging.getLogger("fomo_logger")


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


def test_contrastive(model, data_loader, device):
    """
    Testing function for the model.
    """
    logger.info('Testing Started!')
    model.eval()

    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    pbar = tqdm(data_loader, desc=f"Validating", unit="batch")

    with torch.no_grad():
        for batch in pbar:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            try:
                with torch.no_grad():
                    logits = model(images)
                    prob = torch.softmax(logits, dim=-1)  # bs * 2
            except:  # skip last batch
                logger.warning('Bad evaluation batch!', exc_info=True)
                raise
                # continue
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
    # dump the config
    with open(f"{args.log_dir}/config.yaml", "w") as f:
        f.write(cfg.dump())
    # pretty print args
    display_args(args, title="Config Arguments")

    # setup wandb
    if args.logging:
        wandb.init(
            project=args.project,
            entity="FoMo",
            name=args.run_name + "/" + args.uid,
            config={
                "architecture": args.model,
                "clip_type": args.clip_type,
                "batch_size": args.batch_size,
                "log_dir": args.log_dir,
                "seed": args.seed,
                "mode": args.mode,
                "device": args.device,
                "test_dataset": cfg.dataset.test_dataset,
                "checkpoint": args.checkpoint,
            },
            settings=wandb.Settings(_service_wait=300, init_timeout=120))

    # load model
    model = get_model(name=args.model, clip_type=args.clip_type, cfg=cfg)
    # freeze all except prompt_learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
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
    for dataset in args.test_datasets:
        # change dataset name
        cfg.dataset.test_dataset = dataset
        cfg.dataset.subset = []
        # load validation data
        test_dataset = CTD(config=cfg.dataset,
                          mode="test",
                          jpeg_quality=cfg.dataset.jpeg_quality,
                          debug=args.debug)
        test_data_loader = DataLoader(
            test_dataset, args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, sampler=None)
        describe_dataloader(test_data_loader, title="Test DataLoader Summary")
    
        auc, ap, best_acc, r_acc, f_acc, raw_acc, raw_r_acc, raw_f_acc, best_thresh = test_contrastive(model, test_data_loader, device)
    
        # log metrics
        metrics = {
            f"{dataset}/roc_auc": auc,
            f"{dataset}/average_precision": ap,
            f"{dataset}/accuracy": best_acc,
            f"{dataset}/real_accuracy": r_acc,
            f"{dataset}/fake_accuracy": f_acc,
            f"{dataset}/raw_accuracy": raw_acc,
            f"{dataset}/raw_real_accuracy": raw_r_acc,
            f"{dataset}/raw_fake_accuracy": raw_f_acc,
            f"{dataset}/best_thresh": best_thresh
        }
        wandb.log(metrics)
    
        # export metrics
        export_path = os.path.join(
            cfg.log_dir, f"metrics.json")
    
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
    
        if args.logging:
            # exit session
            wandb.finish()