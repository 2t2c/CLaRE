import argparse
import datetime
import importlib
import logging as logger
import os
import random
import shutil
import warnings

import albumentations as A
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve
from tabulate import tabulate
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torch_kmeans import KMeans as torchKMeans
# from torch_kmeans.utils.distances import CosineSimilarity
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

test_best = -1
test_best_close = -1


class AverageMeter(object):
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



class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file,
                 data_size=512, val_ratio=None, split_anchor=True,
                 args=None,
                 map_file='/home/petterluo/project/FakeImageDetection/outputs/all_map_anns_final.txt',
                 ):
        self.data_root = data_root
        self.data_size = data_size
        self.train_list = []
        self.anchor_list = []
        self.isAnchor = False
        self.isVal = False
        self.split_anchor = split_anchor
        self.albu_pre_train = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_pre_train_easy = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.args = args

        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()

        while line:
            image_path, image_label = line.rsplit(' ', 1)
            label = int(image_label)
            if self.split_anchor and random.random() < 0.1 and label == 0 and len(self.anchor_list) < 100:
                self.anchor_list.append((image_path, label))
            else:
                self.train_list.append((image_path, label))
                # print('add train')
            line = train_file_buf.readline().strip()
            # print(line)

        if val_ratio is not None:
            np.random.shuffle(self.train_list)
            self.test_list = self.train_list[-int(len(self.train_list) * val_ratio):]
            self.train_list = self.train_list[:-int(len(self.train_list) * val_ratio)]
        else:
            self.test_list = self.train_list
        # print('len train:', len(self.train_list))
        # print('len test:', len(self.test_list))
        filename_to_loss = {}
        with open(map_file) as f:
            for line in f:
                image_path, label = line.strip().split('\t')
                filename = image_path.split('/')[-1].split('.')[0]
                filename_to_loss[filename] = image_path

        ordered_map_paths = []
        for ann in self.train_list:
            image_path = ann[0]
            filename = image_path.split('/')[-1].split('.')[0]
            loss_path = filename_to_loss[filename]
            ordered_map_paths.append(loss_path)
        self.ordered_map_paths = ordered_map_paths

    def transform(self, x):
        if self.isVal:
            x = self.albu_pre_val(image=x)['image']
        else:
            if self.args.no_strong_aug:
                x = self.albu_pre_train_easy(image=x)['image']
            else:
                x = self.albu_pre_train(image=x)['image']
        x = self.imagenet_norm(x)  # .unsqueeze(0)
        return x

    def __len__(self):
        if self.isAnchor:
            return len(self.anchor_list)
        elif self.isVal:
            return len(self.test_list)
        else:
            return len(self.train_list)

    def __getitem__(self, index):
        if self.isAnchor:
            return self.getitem(index, self.anchor_list)
        elif self.isVal:
            return self.getitem(index, self.test_list)
        else:
            return self.getitem(index, self.train_list)

    def getitem(self, index, data_list):
        image_path, onehot_label = data_list[index]
        map_path = self.ordered_map_paths[index]

        loss_map = torch.load(map_path)

        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)

        if image is None:
            logger.info('Error Image: %s' % image_path)
            image = np.zeros([512, 512, 3], dtype=np.uint8)
        image = image[..., ::-1]

        crop = self.transform(image)
        onehot_label = torch.LongTensor([onehot_label])
        return crop, onehot_label, loss_map

    def set_val_True(self):
        self.isVal = True

    def set_val_False(self):
        self.isVal = False

    def set_anchor_True(self):
        self.isAnchor = True

    def set_anchor_False(self):
        self.isAnchor = False


def train_one_epoch(data_loader, model, optimizer, cur_epoch, loss_meter, args, device, writer, ngpus_per_node):
    loss_meter.reset()
    batch_idx = 0
    loss_avg = 0
    for (images, labels, loss_maps) in data_loader:
        images = images.to(device)
        labels = labels.to(device).flatten().squeeze()
        loss_maps = loss_maps.to(device)
        # print('=' * 20)
        # print(images.shape)
        logits = model(images, loss_maps)
        # image-axis loss
        loss_img = args.criterion_ce(logits, labels)
        # total loss
        loss = loss_img

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])
        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                     and args.rank % ngpus_per_node == 0)) and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f' % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, loss_meter.count)
            writer.add_scalar('train/lr', lr, loss_meter.count)
        if args.break_onek and batch_idx > 1000:  # ?
            break
        batch_idx += 1
    logger.info('End Training')
    return loss_avg


def validation_contrastive(model, args, test_file, device, ngpus_per_node):
    logger.info('Start eval')
    model.eval()
    val_dataset = ImageDataset(args.data_root, test_file, data_size=args.data_size, split_anchor=False)
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)  # drop_last=True)
    else:
        val_sampler = None
    data_loader = DataLoader(
        val_dataset, args.test_batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    data_loader.dataset.set_val_True()
    data_loader.dataset.set_anchor_False()
    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
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
            if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                         and args.rank % ngpus_per_node == 0)) and iter % 50 == 0 and iter > 0:
                logger.info(
                    'Eval: it %03d/%03d' % (
                        iter, len(data_loader)))

    gt_labels = torch.cat(gt_labels_list, dim=0)
    prob_labels = torch.cat(prob_labels_list, dim=0)

    if args.multiprocessing_distributed:
        gt_labels_output_tensor = torch.zeros((len(val_dataset),), dtype=gt_labels.dtype).to(device)
        prob_labels_output_tensor = torch.zeros((len(val_dataset),), dtype=prob_labels.dtype).to(device)
        gt_labels_gathered_tensor_list = list(torch.split(gt_labels_output_tensor, len(val_dataset) // 8))
        prob_labels_gathered_tensor_list = list(torch.split(prob_labels_output_tensor, len(val_dataset) // 8))

        torch.distributed.all_gather(gt_labels_gathered_tensor_list, gt_labels)
        torch.distributed.all_gather(prob_labels_gathered_tensor_list, prob_labels)

        gt_labels_list = torch.cat(gt_labels_gathered_tensor_list, dim=0).cpu().numpy()
        prob_labels_list = torch.cat(prob_labels_gathered_tensor_list, dim=0).cpu().numpy()
    else:
        gt_labels_list = gt_labels.cpu().numpy()
        prob_labels_list = prob_labels.cpu().numpy()
    print(gt_labels_list.shape)
    print(prob_labels_list.shape)
    fpr, tpr, thres = roc_curve(gt_labels_list, prob_labels_list)
    thresh = thres[len(thres) // 2]
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        logger.info(f'thresh old: {thresh}')
    precision, recall, thresholds = precision_recall_curve(gt_labels_list, prob_labels_list)
    f_score = precision * recall / (precision + recall)
    thresh = thresholds[np.argmax(f_score)]
    # thresh = 0.5
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        logger.info(f'thresh new: {thresh}')

    pred_labels_list = np.array(prob_labels_list)
    pred_labels_list[pred_labels_list > thresh] = 1
    pred_labels_list[pred_labels_list <= thresh] = 0

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    accuracy = accuracy_score(gt_labels_list, pred_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)
    model.train()

    best_acc, best_thresh = search_best_acc(gt_labels_list, prob_labels_list)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
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


def main(gpu, ngpus_per_node, args):
    global test_best
    global test_best_close

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir))
    else:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            writer = FakeWriter()
    args.gpu = gpu
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = getattr(importlib.import_module('model'), args.model)(num_class=args.num_class, clip_type=args.clip_type)
    # model = torch.nn.DataParallel(model).cuda()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    logger.info(
                        f'Batch size={args.batch_size}, ngpus_per_node={ngpus_per_node}, workers={args.workers}')
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # load data
    train_dataset = ImageDataset(args.data_root, args.train_file, data_size=args.data_size, val_ratio=None,
                                 split_anchor=False, args=args)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        logger.info(f'Train dataset size: {len(train_dataset)}')
    # train_data_loader = DataLoader(train_dataset,
    #                                args.batch_size, shuffle=True, num_workers=min(48, args.batch_size), drop_last=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        # val_sampler = None

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.test_file == '':
        test_file_list = [
            ('annotation/val_midjourney_12k.txt', 'Midjourney'),
            ('annotation/val_sdv4_12k.txt', 'StableDiffusionV1.4'),
            ('annotation/val_sdv5_12k.txt', 'StableDiffusionV1.5'),
            ('annotation/val_adm_12k.txt', 'ADM'),
            ('annotation/val_glide_12k.txt', 'GLIDE'),
            ('annotation/val_wukong_12k.txt', 'WuKong'),
            ('annotation/val_vqdm_12k.txt', 'VQDM'),
            ('annotation/val_biggan_12k.txt', 'Biggan'),
        ]
    elif args.test_file == 'robust':
        test_file_list = [
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/midjourney_jpg1_ann.txt', 'Midjourney'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/sdv4_jpg1_ann.txt', 'StableDiffusionV1.4'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/sdv5_jpg1_ann.txt', 'StableDiffusionV1.5'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/adm_jpg1_ann.txt', 'ADM'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/glide_jpg1_ann.txt', 'GLIDE'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/wukong_jpg1_ann.txt', 'WuKong'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/vqdm_jpg1_ann.txt', 'VQDM'),
            ('/home/petterluo/project/FakeImageDetection/outputs/robust_test/anns/biggan_jpg1_ann.txt', 'Biggan'),
        ]
    else:
        test_file_list = [
            (args.test_file, 'Test Dataset'),
        ]
    if not args.label_smooth:
        args.criterion_ce = nn.CrossEntropyLoss().to(device)
    else:
        args.criterion_ce = LabelSmoothingLoss(classes=args.num_class, smoothing=args.smoothing)
    # args.criterion_ce = torch.nn.CrossEntropyLoss().cuda()
    # args.torchKMeans = torchKMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        logger.info('Params: %.2f' % (params / (1024 ** 2)))

    if args.resume != '':
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint, strict=False)
    elif args.isTrain == 0:
        raise ValueError("Eval mode but no checkpoint path")

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)
    # optimizer = optim.AdamW(parameters, lr=args.lr)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    loss_meter = AverageMeter()

    for epoch in range(args.epoches):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        train_data_loader.dataset.set_val_False()
        if args.isTrain == 1:
            train_one_epoch(train_data_loader, model, optimizer, epoch, loss_meter, args, device, writer,
                            ngpus_per_node)

            # 训练完直接保存，防止报错白训
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'latest.pt'))
            if args.val_method == 'cluster':
                val_auc, val_acc, val_ap = validation_cluster(model, args, args.val_file)
            elif args.val_method == 'sim':
                val_auc, val_acc, val_ap = validation_similarity(model, args, args.val_file)
            else:
                val_auc, val_acc, val_ap, val_raw_acc, val_r_acc, val_f_acc = validation_contrastive(model, args,
                                                                                                     args.val_file,
                                                                                                     device,
                                                                                                     ngpus_per_node)

            val_score = val_auc
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                logger.info(
                    f'Score: Validation AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, AP: {val_ap:.4f}, Raw ACC{val_raw_acc:.4f},'
                    f' Real ACC: {val_r_acc:.4f}, Fake ACC: {val_f_acc:.4f}')
                writer.add_scalar('val/AUC', val_auc, epoch)
                writer.add_scalar('val/ACC', val_acc, epoch)
                writer.add_scalar('val/AP', val_ap, epoch)
                writer.add_scalar('val/RawACC', val_raw_acc, epoch)
                writer.add_scalar('val/RealACC', val_r_acc, epoch)
                writer.add_scalar('val/FakeACC', val_f_acc, epoch)
                if val_acc > test_best_close:
                    test_best_close = val_acc
                    saved_name = 'Val_best.pth'
                    torch.save(model.state_dict(), os.path.join(args.out_dir, saved_name))
                    logger.info(f'Epoch: {epoch}, Best Val')

        # Testing:
        table_header = ['Metric']
        raw_acc_row = ['Raw ACC']
        real_acc_row = ['Real ACC']
        fake_acc_row = ['Fake ACC']
        acc_row = ['ACC']
        ap_row = ['AP']
        auc_row = ['AUC']
        test_score_list = []

        for test_file, nickname in test_file_list:
            if args.val_method == 'cluster':
                test_auc, test_acc, test_ap = validation_cluster(model, args, test_file)
            elif args.val_method == 'sim':
                test_auc, test_acc, test_ap = validation_similarity(model, args, test_file)
            else:
                test_auc, test_acc, test_ap, test_raw_acc, test_r_acc, test_f_acc = validation_contrastive(model, args,
                                                                                                           test_file,
                                                                                                           device,
                                                                                                           ngpus_per_node)
            test_score_list.append(test_auc)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                logger.info(f'Eval res of {test_file}')
                logger.info(
                    f'Score of {nickname}: AUC: {test_auc:.4f}, Acc: {test_acc:.4f}, AP: {test_ap:.4f}'
                    f', Raw ACC: {test_raw_acc:.4f}, Real ACC: {test_r_acc:.4f}, Fake ACC: {test_f_acc:.4f}')
                writer.add_scalar(f'test/AUC@{nickname}', test_auc, epoch)
                writer.add_scalar(f'test/ACC@{nickname}', test_acc, epoch)
                writer.add_scalar(f'test/AP@{nickname}', test_ap, epoch)
                writer.add_scalar(f'test/RawACC@{nickname}', test_raw_acc, epoch)
                writer.add_scalar(f'test/RealACC@{nickname}', test_r_acc, epoch)
                writer.add_scalar(f'test/FakeACC@{nickname}', test_f_acc, epoch)

                table_header.append(nickname)
                acc_row.append(test_acc)
                auc_row.append(test_auc)
                raw_acc_row.append(test_raw_acc)
                real_acc_row.append(test_r_acc)
                fake_acc_row.append(test_f_acc)
                ap_row.append(test_ap)

        test_score = np.mean(test_score_list)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            table_data = [acc_row, raw_acc_row, ap_row, auc_row, real_acc_row, fake_acc_row]
            logger.info('\n' + tabulate(table_data, headers=table_header, tablefmt='psql'))

            writer.add_scalar('test/AVGAUC', test_score, epoch)
            isBest = '(Not Best)'
            if args.isTrain and test_score > test_best:
                test_best = test_score
                saved_name = 'Test_best.pth'
                isBest = '(Best)'
                torch.save(model.state_dict(), os.path.join(args.out_dir, saved_name))
                logger.info(f'Epoch: {epoch}, Best Test')

            logger.info('Score: Mean: %5.4f  %s' % (test_score, isBest))

        if args.isTrain == 0:
            exit()

        lr_schedule.step(test_score)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class FakeWriter:
    def __init__(self):
        pass

    def add_scalar(self, p1, p2, p3):
        pass


if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument("--data_root", type=str, default='data/',
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str,
                      default='annotation/Train_num398700.txt')
    conf.add_argument("--val_file", type=str,
                      default='annotation/Test_MidjourneyV5_num2000.txt')
    conf.add_argument("--test_file", type=str,
                      default='annotation/Test_MidjourneyV5_num2000.txt')
    conf.add_argument('--val_ratio', type=float, default=0.005)
    conf.add_argument('--isTrain', type=int, default=1)
    conf.add_argument("--model", type=str, default='LASTED')
    conf.add_argument("--num_class", type=int, default=2, help='The class number of training dataset')
    conf.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    conf.add_argument("--weights", type=str, default='out_dir', help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=9999, help='The training epoches.')
    conf.add_argument('--batch_size', type=int, default=48, help='The training batch size over all gpus.')
    conf.add_argument('--test_batch_size', type=int, default=48, help='The training batch size over all gpus.')
    conf.add_argument('--data_size', type=int, default=448, help='The image size for training.')
    # conf.add_argument('--gpu', type=str, default='0,1,2,3', help='The gpu')
    conf.add_argument("--resume", type=str, default='')
    conf.add_argument("--out_dir", type=str, default='')
    conf.add_argument("--break_onek", action='store_true', default=False)
    conf.add_argument("--val_method", type=str, default="cluster", choices=["cluster", "sim", "con"])
    conf.add_argument("--no_strong_aug", action='store_true', default=False)
    conf.add_argument("--label_smooth", action='store_true', default=False)
    conf.add_argument('--smoothing', type=float, default=0.1)
    conf.add_argument("--seed", type=int, default=None)
    conf.add_argument("--gpu", type=int, default=None)
    conf.add_argument("--exp_name", type=str, default='')
    conf.add_argument("--clip_type", type=str, default='RN50x64')
    conf.add_argument('--multiprocessing-distributed', action='store_true',
                      help='Use multi-processing distributed training to launch '
                           'N processes per node, which has N GPUs. This is the '
                           'fastest way to use PyTorch for either single node or '
                           'multi node data parallel training')
    conf.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                      help='url used to set up distributed training')
    conf.add_argument('--dist-backend', default='nccl', type=str,
                      help='distributed backend')
    conf.add_argument('--world-size', default=-1, type=int,
                      help='number of nodes for distributed training')
    conf.add_argument('--rank', default=-1, type=int,
                      help='node rank for distributed training')
    conf.add_argument('-j', '--workers', default=48, type=int, metavar='N',
                      help='number of data loading workers (default: 4)')
    args = conf.parse_args()
    # os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))

    logger.info(args)

    global_iter = 0
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.isTrain == 1:
        logger.info('Train Mode')
        date_now = datetime.datetime.now()
        date_now = 'Exp' + args.exp_name + '_Log_v%02d%02d%02d%02d' % (
            date_now.month, date_now.day, date_now.hour, date_now.minute)
        args.time = date_now
        args.out_dir = os.path.join(args.out_dir, args.time)
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
            os.makedirs(args.out_dir, exist_ok=True)
    else:
        logger.info('Eval Mode')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main(args.gpu, 1, args)

    # main(args)
