from __future__ import print_function, division
import sys
sys.path.append("core")

from torch.utils.tensorboard import SummaryWriter
import argparse
import os

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import  core.datasets as DA
from torch.utils.data import DataLoader
from core.raft import RAFT
import matplotlib.pyplot as plt

import numpy as np
import math

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

# -----------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='stereo matching', help="name your experiment")
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default='/home/wy/文档/Data/SceneFLow/', help='datapath')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
# parser.add_argument('--stage', help="determines which dataset to use for training")
parser.add_argument('--restore_ckpt', default='/',
                    help="restore checkpoint")
parser.add_argument('--validation', type=str, nargs='+')
parser.add_argument('--add_noise', action='store_true', default=True)


parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--num_steps', type=int, default=2000000)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--gpus', type=int, nargs='+', default=[4,5,6,7])

parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--savemodel', default='./trained/radius4', help='save model')
parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512])

parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
# parser.add_argument('--add_noise', action='store_true')

args = parser.parse_args()

torch.manual_seed(1234)
np.random.seed(1234)

def update_metrics_status(metrics, running_loss):
    """print metrics_status"""
    for key in metrics:
        if key not in running_loss:
            running_loss[key] = 0.0

        running_loss[key] += metrics[key]


def sequence_loss(disp_preds, disp_gt, valid, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    disp_loss = 0.0
    disp_loss1 = 0.0

    for i in range(1, n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        disp_loss1 += i_weight * F.smooth_l1_loss(disp_preds[i][valid], disp_gt[valid])

    epe = (disp_preds[-1] - disp_gt).abs()
    epe = epe.view(-1)[valid.view(-1)]

    disp_loss2 = 0.5 * F.smooth_l1_loss(disp_preds[0][valid], disp_gt[valid])

    disp_loss = disp_loss1 + disp_loss2
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return disp_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


# ---------------------------------------------------------- #
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = \
    DA.dataloader_SceneFlow(args.datapath)

train_dataset = DA.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True)
test_dataset = DA.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False)

TrainImgLoader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model = nn.DataParallel(RAFT(args).cuda(), device_ids=[0])
model.load_state_dict(torch.load('./trained/r8/checkpoint_1.tar')['state_dict'])
# if args.restore_ckpt is not None:
#     print('loading the model')
#     model.load_state_dict(torch.load('./trained/r4n/checkpoint_11.tar')['state_dict'])

model.cuda()
optimizer, scheduler = fetch_optimizer(args, model)
scaler = GradScaler()
# -------------------------------------------------------------------------#
def train(imgL, imgR, disp_L):

    model.train()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    valid = (disp_true < 192) & (disp_true > 0)
    valid = valid.byte().bool()
    # -------
    optimizer.zero_grad()

    disp_predicitons = model(imgL, imgR, iters=args.iters)

    loss, metrics = sequence_loss(disp_predicitons, disp_true, valid, args.gamma)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    scaler.step(optimizer)
    scheduler.step()
    scaler.update()

    return loss, metrics


def test(imgL, imgR, disp_true):

    model.eval()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    valid = (disp_true < 192) & (disp_true > 0)
    valid = valid.byte().bool()

    if imgL.shape[2] % 4 != 0:
        times = imgL.shape[2] // 4
        top_pad = (times + 1) * 4  - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 4  != 0:
        times = imgL.shape[3] // 4
        right_pad = (times + 1) * 4  - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        disp_low, disp_pr = model(imgL, imgR, iters=args.iters, training=False)

    if top_pad != 0:
        disp_pr = disp_pr[:, top_pad:, :]
    else:
        pass
    if right_pad != 0:
        disp_pr = disp_pr[:, :, :right_pad]
    else:
        pass

    loss = F.smooth_l1_loss(disp_pr[valid], disp_true[valid])
    epe = (disp_pr - disp_true).abs()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }
    return loss.data.cpu(), metrics


def main():

    for epoch in range(0, args.epochs): # load 23-th model_dict
        print("Parameter Count: %d" % count_parameters(model))
        print('Training with %d image pairs' % len(train_dataset))
        print('This is %d-th epoch' % (epoch))

        total_train_loss = 0.0
        running_train_loss = {}
        ## training ##
        for batch_idx, data_blob in enumerate(TrainImgLoader):
            imgL_crop, imgR_crop, disp_crop_L = [x.cuda() for x in data_blob]

            # if args.add_noise:
            #     stdv = np.random.uniform(0.0, 5.0)
            #     imgL_crop = (imgL_crop + stdv * torch.randn(*imgL_crop.shape).cuda()).clamp(0.0, 255.0)
            #     imgR_crop = (imgR_crop + stdv * torch.randn(*imgR_crop.shape).cuda()).clamp(0.0, 255.0)

            loss, metrics = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f, lr=%.8f' % (batch_idx, loss, scheduler.get_last_lr()[0]))
            total_train_loss += loss
            update_metrics_status(metrics, running_train_loss)

        metrics_train_data = [running_train_loss[k]/len(TrainImgLoader) for k in sorted(running_train_loss.keys())]
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        print(("{:10.4f}, " * len(metrics_train_data)).format(*metrics_train_data))

            # SAVE

        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename, _use_new_zipfile_serialization=False)

        # ------------- TEST ------------------------------------------------------------
        if epoch > 29:
            total_test_loss = 0
            running_test_loss = {}
            # test ##
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, metrics_test = test(imgL, imgR, disp_L)
                print('Iter %d test loss = %.3f, lr=%.8f' % (batch_idx, test_loss, scheduler.get_last_lr()[0]))

                total_test_loss += test_loss
                update_metrics_status(metrics_test, running_test_loss)

            metrics_test_data = [running_test_loss[k] / len(TestImgLoader) for k in sorted(running_test_loss.keys())]
            print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
            print(("{:10.4f},"*len(metrics_test_data)).format(*metrics_test_data))
            # ----------------------------------------------------------------------------------
            # SAVE test information
            savefilename = './tested' + str(epoch) + 'test_information.tar'
            torch.save({
            'test_loss': total_test_loss / len(TestImgLoader),
            }, savefilename)


if __name__ == '__main__':
    main()
