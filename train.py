from __future__ import print_function,division
import sys
sys.path.append("models")

from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import datasets as DA
from torch.utils.data import DataLoader
from raft import RAFT

import numpy as np
import time
import math

from models import *

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(disp_preds,disp_gt,mask):
    n_predictions=len(disp_preds)
    disp_loss=0.0
    for i in range(n_predictions):
        disp_loss = F.smooth_l1_loss(disp_preds[i][mask], disp_gt[mask],size_average=True)

    epe = torch.abs(disp_preds[-1][mask] - disp_gt[mask])
    #epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model=nn.DataParallel(RAFT(args),device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    all_left_img, all_right_img, all_left_disp = DA.dataloader(args.datapath)

    #if args.restore_ckpt is not None:
     #   model.load_state_dict(torch.load(args.restore_ckpt),strict=False)

    model.cuda()
    model.train()

    train_dataset= DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True)
    train_loader= torch.utils.data.DataLoader(train_dataset,
         batch_size= 2, shuffle= True, num_workers= 4, drop_last=False)
    print('Training with %d image pairs' % len(train_dataset))
   
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000

    should_keep_training = True
    while should_keep_training:

        for batch_idx,data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            imgL_crop,imgR_crop,disp_crop_L=[x.cuda() for x in data_blob]

            disp_predicitons=model(imgL_crop,imgR_crop,iters=args.iters)

            mask = (disp_crop_L < args.maxdisp) &(disp_crop_L>0)
            mask.detach_()
            loss,metrics=sequence_loss(disp_predicitons,disp_crop_L,mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

            total_step=1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='stereo matching', help="name your experiment")
    parser.add_argument('--maxdisp', type=int, default=192,help='maximum disparity')
    parser.add_argument('--datapath', default='/home/wy/datasets/KITTI/',help='datapath')

    #parser.add_argument('--stage', help="determines which dataset to use for training")
    #parser.add_argument('--restore_ckpt', help="restore checkpoint")
    #parser.add_argument('--small', action='store_true', help='use small model')
    #parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    #parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    #parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)