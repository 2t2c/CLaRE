"""
Script to train the architecture pipeline.
"""

import os
import sys
os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join("..")))
from core import get_model
from datetime import datetime
from zoneinfo import ZoneInfo
import time
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
from utils import set_seed, get_device, display_metrics, display_model_summary
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


def train_one_epoch(data_loader, model, optimizer, epoch,
                    loss_meter, auc_meter, args, device,
                    step):
    # set variables
    loss_meter.reset()
    auc_meter.reset()
    best_val, best_step = 0, 0
    start_time = time.time()

    for (images, labels, loss_maps) in data_loader:
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
            elapsed = time.time() - start_time
            metrics = {
                "train/epoch": epoch,
                "train/loss": loss_meter.avg,
                "train/lr": lr,
                "train/step": step,
            }
            if args.logging:
                wandb.log(metrics, step=step)
            rprint("Time Elapsed:", elapsed)
            display_metrics(metrics=metrics)
            loss_meter.reset()

        # validation statistics
        if step % args.eval_every == 0:
            # save directly after training to avoid errors and wasted training
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'latest.pt'))
            val_auc, val_acc, val_ap, val_raw_acc, val_r_acc, val_f_acc = validation_contrastive(model, args,
                                                                                                 args.val_file,
                                                                                                 device)
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
            elapsed = time.time() - start_time
            rprint("Time Elapsed:", elapsed)
            display_metrics(metrics=val_metrics)

    return step, None


def validation_contrastive(model, data_loader, test_file, device):
    """
    Validation function for the model.
    """
    logger.info('Starting Evaluation')
    model.eval()
    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    logger.info(f'Val dataset size: {len(data_loader.dataset)}')
    gt_labels_list = []
    pred_scores = []
    # i = 0
    with torch.no_grad():
        for iter, (images, labels, loss_maps) in enumerate(data_loader):
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
            if iter % 50 == 0 and iter > 0:
                logger.info(
                    'Eval: it %03d/%03d' % (
                        iter, len(data_loader)))

    # gather ground truth labels and predicted labels
    gt_labels = torch.cat(gt_labels_list, dim=0)
    prob_labels = torch.cat(prob_labels_list, dim=0)
    gt_labels_list = gt_labels.cpu().numpy()
    prob_labels_list = prob_labels.cpu().numpy()

    fpr, tpr, thres = roc_curve(gt_labels_list, prob_labels_list)
    thresh = thres[len(thres) // 2]
    logger.info(f'thresh old: {thresh}')
    precision, recall, thresholds = precision_recall_curve(gt_labels_list, prob_labels_list)
    f_score = precision * recall / (precision + recall)
    thresh = thresholds[np.argmax(f_score)]
    # thresh = 0.5
    logger.info(f'thresh new: {thresh}')

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
    # setup wandb
    if args.logging:
        wandb.init(
            project=args.project,
            entity="FoMo",
            name=args.run_name+ "_" + args.uid,
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
                # "dataset_type": dataset_type,
                # "dataset_name": dataset_name,
            },
            settings=wandb.Settings(_service_wait=300, init_timeout=120))

    global test_best
    global test_best_close

    # load model
    model = get_model(name=args.model, type=args.clip_type, roi_pooling=args.roi_pooling)
    device = get_device(args.device)
    model.to(device)
    display_model_summary(model, input_shape=(1, 3, 224, 224), device=device)

    # load the config file
    with open("../configs/df40/" + args.config, 'r') as f:
        config = yaml.safe_load(f)
    # load training data
    train_dataset =  LARE(config=config, 
                       mode=args.mode, 
                       jpeg_quality=args.jpeg_quality)
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)
    # train_data_loader.dataset.set_val_False()
    describe_dataloader(train_data_loader)

    # load validation data
    # val_dataset = LARE(args.data_root, args.val_file, data_size=args.data_size, split_anchor=False)
    # val_data_loader = DataLoader(
    #     val_dataset, args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=None)
    # describe_dataloader(val_data_loader)
    # val_data_loader.dataset.set_val_True()
    # val_data_loader.dataset.set_anchor_False()

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
    rprint("Training Started!")
    step = 0
    for epoch in range(args.epochs):
        # set model to training mode
        model.train()
        # train one epoch
        step = train_one_epoch(train_data_loader, model, optimizer, epoch,
                               loss_meter, auc_meter, args, device,
                               step)
        # scheduler step
        lr_schedule.step(auc_meter.avg)

    if args.logging:
        # exit session
        wandb.finish()
