"""
Script to train the LaRE architecture pipeline.
"""

import os
import sys
os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import get_model
from datetime import datetime
from datetime import timedelta
from zoneinfo import ZoneInfo
import time
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.utils.data.distributed
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from rich import print as rprint
import wandb
from loss import LabelSmoothingLoss
from utils import set_seed, get_device, display_metrics, display_model_summary, load_config
from dataset import describe_dataloader, LARE
import logging

# fetch logger
logger = logging.getLogger("fomo_logger")

test_best = -1
test_best_close = -1


class StatsMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, train_data_loader, val_data_loader, 
                    optimizer, epoch,
                    loss_meter, auc_meter, args, device,
                    step):
    # set model to training mode
    model.train()
    # set variables
    loss_meter.reset()
    auc_meter.reset()
    best_val, best_step = 0, 0
    start_time = time.time()
    pbar = tqdm(train_data_loader, desc=f"Epoch {epoch}", unit="batch")

    for batch in pbar:
        images, labels, loss_maps = batch
        images = images.to(device)
        labels = labels.to(device).flatten().squeeze()
        loss_maps = loss_maps.to(device)
        logits = model(images, loss_maps)

        # image-axis loss
        loss_img = args.criterion_ce(logits, labels)
        # total loss
        loss = loss_img
        # autograd
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        # update variables
        loss_meter.update(loss.item(), images.shape[0])
        step += 1

        # training statistics
        if step % args.log_every == 0:
            lr = get_lr(optimizer)
            metrics = {
                "train/epoch": epoch,
                "train/loss": loss_meter.avg,
                "train/lr": lr,
                "train/step": step,
            }
            if args.logging:
                wandb.log(metrics, step=step)
            elasped = time.time() - start_time
            display_metrics(metrics=metrics, elasped=elasped, title="Training Metrics")
            loss_meter.reset()

        # validation statistics
        if step % args.eval_every == 0:
            # save directly after training to avoid errors and wasted training
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'latest.pt'))
            val_auc, val_acc, val_ap, val_raw_acc, val_r_acc, val_f_acc = validation_contrastive(model, val_data_loader,
                                                                                                 step, device)
            if val_acc > best_val:
                best_val = val_acc
                best_step = step
                ckpt = {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_step": best_step
                }
                name = "best.pth"
                torch.save(ckpt, os.path.join(args.out_dir, name))
                logger.info(f'Epoch {epoch}, Step {step}: New best val accuracy, model saved.')
                auc_meter.update(best_val, images.shape[0])

            # log metrics to wandb
            val_metrics = {
                "val/epoch": epoch,
                "val/roc_auc": val_auc,
                "val/accuracy": val_acc,
                "val/average_precision": val_ap,
                "val/raw_accuracy": val_raw_acc,
                "val/real_accuracy": val_r_acc,
                "val/fake_accuracy": val_f_acc,
            }
            if args.logging:
                wandb.log(val_metrics, step=step)
            elasped = time.time() - start_time
            display_metrics(metrics=val_metrics, elasped=elasped)

            model.train()  # switch back to training mode after validation

    return step


def validation_contrastive(model, data_loader, step, device):
    """
    Validation function for the model.
    """
    logger.info('Evaluation Started!')
    model.eval()

    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    pbar = tqdm(data_loader, desc=f"Step {step}", unit="batch")

    with torch.no_grad():
        for batch in pbar:
            images, labels, loss_maps = batch
            images = images.to(device)
            b, C, H, W = images.shape
            images = images.reshape(b, C, H, W)
            labels = labels.flatten().squeeze().to(device)
            loss_maps = loss_maps.to(device)
            try:
                with torch.no_grad():
                    logits = model(images, loss_maps)
                    prob = torch.softmax(logits, dim=-1)  # bs * 2
            except:  # skip last batch
                logger.info('Bad eval')
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
    accuracy = accuracy_score(gt_labels_list, pred_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)
    model.train()

    best_acc, best_thresh = search_best_acc(gt_labels_list, prob_labels_list)
    logger.info(f'Search ACC: {best_acc}, Search Thresh: {best_thresh}')

    r_acc = accuracy_score(gt_labels_list[gt_labels_list == 0], prob_labels_list[gt_labels_list == 0] > 0.5)
    f_acc = accuracy_score(gt_labels_list[gt_labels_list == 1], prob_labels_list[gt_labels_list == 1] > 0.5)
    raw_acc = accuracy_score(gt_labels_list, prob_labels_list > 0.5)

    return auc, best_acc, ap, raw_acc, r_acc, f_acc


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


def train(args):
    # load the config file
    with open(args.config, 'r') as f:
        config_file = yaml.safe_load(f)
    # convert to yacs
    cfg = load_config(config_file)

    # setup wandb
    if args.logging:
        wandb.init(
            project=args.project,
            entity="FoMo",
            name=args.run_name+ "/" + args.uid,
            config={
                "architecture": args.model,
                "clip_type": args.clip_type,
                "batch_size": args.batch_size,
                "out_dir": args.out_dir,
                "seed": args.seed,
                "mode": args.mode,
                "device": args.device,
                "eval_every": args.eval_every,
                "log_every": args.log_every,
                "epochs": args.epochs,
                "train_dataset": cfg["train_dataset"],
                "test_dataset": cfg["test_dataset"],
            },
            settings=wandb.Settings(_service_wait=300, init_timeout=120))

    global test_best
    global test_best_close

    # load model
    model = get_model(name=args.model, type=args.clip_type, roi_pooling=args.roi_pooling)
    device = get_device(args.device)
    model.to(device)
    display_model_summary(model, input_shape=(1, 3, 224, 224), device=device)

    # load training data
    train_dataset = LARE(config=cfg,
                       mode="train",
                       jpeg_quality=args.jpeg_quality,
                       debug=args.debug)
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)
    describe_dataloader(train_data_loader)

    # load validation data
    val_dataset = LARE(config=cfg,
                       mode="test", 
                       jpeg_quality=args.jpeg_quality,
                       debug=args.debug)
    val_data_loader = DataLoader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)
    describe_dataloader(val_data_loader)

    # setting loss criterion
    if not args.label_smooth:
        args.criterion_ce = nn.CrossEntropyLoss().to(device)
    else:
        args.criterion_ce = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.smoothing)
    # args.criterion_ce = torch.nn.CrossEntropyLoss().cuda()
    # args.torchKMeans = torchKMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)


    # if args.resume != '':
    #     if args.gpu is None:
    #         checkpoint = torch.load(args.resume)
    #     elif torch.cuda.is_available():
    #         # map model to be loaded to specified single gpu.
    #         loc = 'cuda:{}'.format(args.gpu)
    #         checkpoint = torch.load(args.resume, map_location=loc)
    #     model.load_state_dict(checkpoint, strict=False)
    # elif args.isTrain == 0:
    #     raise ValueError("Eval mode but no checkpoint path")

    # setting optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)
    # optimizer = optim.AdamW(parameters, lr=args.lr)

    # setting scheduler
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)
    loss_meter = StatsMeter()
    auc_meter = StatsMeter()

    # starting the training
    logger.info(f"Training Started! - Debugging: {args.debug}")
    step = 0
    for epoch in range(args.epochs):
        # train one epoch
        step = train_one_epoch(model, train_data_loader, val_data_loader, 
                               optimizer, epoch,
                               loss_meter, auc_meter, args, device,
                               step)
        # scheduler step
        lr_schedule.step(auc_meter.avg)

    if args.logging:
        # exit session
        wandb.finish()
