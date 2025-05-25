"""
Script to train the LaRE architecture pipeline.
"""

import os
import sys

os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

import torch
import torch.utils.data.distributed
import wandb
import yaml
from core import get_model
from dataset import LARE, describe_dataloader
from loss import LabelSmoothingLoss
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import dict_to_yacs, display_metrics, display_model_summary, get_device


def train_one_epoch(
    model,
    train_data_loader,
    val_data_loader,
    optimizer,
    epoch,
    loss_meter,
    auc_meter,
    args,
    device,
    step,
):
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
            display_metrics(metrics=metrics, elapsed=elasped, title="Training Metrics")
            loss_meter.reset()

    return step


def train(args):
    # load the config file
    with open(args.config, "r") as f:
        config_file = yaml.safe_load(f)
    # convert to yacs
    cfg = dict_to_yacs(config_file)

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
            settings=wandb.Settings(_service_wait=300, init_timeout=120),
        )

    global test_best
    global test_best_close

    # load model
    model = get_model(
        name=args.model, clip_type=args.clip_type, roi_pooling=args.roi_pooling
    )
    device = get_device(args.device)
    model.to(device)
    display_model_summary(model, input_shape=(1, 3, 224, 224), device=device)

    # load training data
    train_dataset = LARE(
        config=cfg, mode="train", jpeg_quality=args.jpeg_quality, debug=args.debug
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )
    describe_dataloader(train_data_loader)

    # load validation data
    val_dataset = LARE(
        config=cfg, mode="test", jpeg_quality=args.jpeg_quality, debug=args.debug
    )
    val_data_loader = DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )
    describe_dataloader(val_data_loader)

    # setting loss criterion
    if not args.label_smooth:
        args.criterion_ce = nn.CrossEntropyLoss().to(device)
    else:
        args.criterion_ce = LabelSmoothingLoss(
            classes=args.num_classes, smoothing=args.smoothing
        )
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
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7
    )
    loss_meter = StatsMeter()
    auc_meter = StatsMeter()

    # starting the training
    logger.info(f"Training Started! - Debugging: {args.debug}")
    step = 0
    for epoch in range(args.epochs):
        # train one epoch
        step = train_one_epoch(
            model,
            train_data_loader,
            val_data_loader,
            optimizer,
            epoch,
            loss_meter,
            auc_meter,
            args,
            device,
            step,
        )
        # scheduler step
        lr_schedule.step(auc_meter.avg)

    if args.logging:
        # exit session
        wandb.finish()
