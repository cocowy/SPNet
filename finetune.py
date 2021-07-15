from __future__ import print_function, division
import sys

sys.path.append("core")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import core.datasets as DA
from core.raft import RAFT
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np

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
parser.add_argument('--datapath0', default='/home/wy/文档/Data/KITTI_2015/training/', help='datapath')
parser.add_argument('--datapath1', default='/home/wy/文档/Data/KITTI/training/', help='datapath')

parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
# parser.add_argument('--stage', help="determines which dataset to use for training")
parser.add_argument('--restore_ckpt', default='/home/wy/文档/RAFT(wy)/trained/checkpoint_29.tar',
                    help="restore checkpoint")
parser.add_argument('--validation', type=str, nargs='+')
parser.add_argument('--small', action='store_true', help='use small model')

parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--num_steps', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpus', type=int, nargs='+', default=[0])

parser.add_argument('--image_size', type=int, nargs='+', default=[288, 960])
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--savemodel', default='./', help='save model')

# parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
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

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        # disp_loss
        disp_loss += i_weight * F.smooth_l1_loss(disp_preds[i][valid], disp_gt[valid])

    # metrics
    epe = (disp_preds[-1] - disp_gt).abs()
    epe = epe.view(-1)[valid.view(-1)]

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
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    #
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
    #                                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wdecay,
                           eps=args.epsilon)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 6000, gamma=0.5)
    return optimizer, scheduler


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


error_colormap = gen_error_colormap()
# ---------------------------------------------------------- #
all_left_img_0, all_right_img_0, all_left_disp_0, test_left_img, test_right_img, test_left_disp = \
    DA.dataloader_KITTI2015(args.datapath0)
all_left_img_1, all_right_img_1, all_left_disp_1, test_left_im_1, test_right_img_1, test_left_disp_1 = \
    DA.dataloader_KITTI(args.datapath1)

all_left_img = all_left_img_0 + all_left_img_1
all_right_img = all_right_img_0 + all_right_img_1
all_left_disp = all_left_disp_0 + all_left_disp_1

train_dataset = DA.myImageFloder_KITTI(all_left_img, all_right_img, all_left_disp, True)
test_dataset = DA.myImageFloder_KITTI(test_left_img, test_right_img, test_left_disp, False)

TrainImgLoader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model = nn.DataParallel(RAFT(args), device_ids=[0])
if args.restore_ckpt is not None:
    model.load_state_dict(torch.load('./fined/checkpoint_1999.tar'))

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
    # img_ = imgL
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # img_ = img_.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    # plt.imshow(img_)
    # plt.show()
    #
    # d_ = disp_true.cpu().squeeze(0).detach().numpy()
    # plt.subplot(1, 3, 2)
    # plt.imshow(d_)
    # plt.show()

    disp_predicitons = model(imgL, imgR, iters=args.iters)
    # d = disp_predicitons[-1].squeeze(0).cpu().detach().numpy()
    # plt.subplot(1, 3, 3)
    # plt.imshow(d)
    # plt.show()
    # disp_pr_ = disp_predicitons[-1].squeeze(0).cpu().detach().numpy()
    # skimage.io.imsave('/home/wy/文档/RAFT(wy)/disp_pr', (disp_pr_*256).astype('uint16'))

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

    abs_thres = 3
    rel_thres = 0.05
    valid = (disp_true < 192) & (disp_true > 0)
    valid = valid.byte().bool()

    if imgL.shape[2] % 8 != 0:
        times = imgL.shape[2] // 8
        top_pad = (times + 1) * 8 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 8 != 0:
        times = imgL.shape[3] // 8
        right_pad = (times + 1) * 8 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        disp_low, disp_pr = model(imgL, imgR, iters=16, training=False)
    if top_pad != 0:
        disp_pr = disp_pr[:, top_pad:, :]
    else:
        pass
    if right_pad != 0:
        disp_pr = disp_pr[:, :, :right_pad]
    else:
        pass
    #
    # img_ = imgL
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # img_ = img_.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    # img_ = img_ / 256
    # plt.imshow(img_)
    #
    # D_gt_np = disp_true.cpu().squeeze(0).detach().numpy()
    # plt.subplot(1, 3, 1)
    # plt.imshow(D_gt_np)
    # valid = D_gt_np > 0
    # d = disp_pr.squeeze(0).cpu().detach().numpy()
    # skimage.io.imsave('/home/wy/文档/RAFT_4resolution1(wy)/disp_pr/', (disp_pr_*256).astype('uint16'))

    loss = F.smooth_l1_loss(disp_pr[valid], disp_true[valid])
    epe = (disp_pr - disp_true).abs()
    epe = epe.view(-1)[valid.view(-1)]
    #
    # D_est_np = disp_pr[-1].squeeze(0).cpu().detach().numpy()
    # # plt.subplot(1, 3, 2)
    # # plt.imshow(D_est_np)
    #
    # valid = D_gt_np > 0
    # # plt.subplot(1, 1, 1)
    # error = np.abs(D_est_np-D_gt_np)
    # error[np.logical_not(valid)] = 0
    # error[valid] = np.minimum(error[valid] / abs_thres, (error[valid] / D_gt_np[valid]) / rel_thres)
    #
    # # get colormap
    # cols = error_colormap
    # # create error image
    # H, W = D_gt_np.shape
    # error_image = np.zeros([H, W, 3], dtype=np.float32)
    # for i in range(cols.shape[0]):
    #         error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # error_image[np.logical_not(valid)] = 0.
    #
    # for i in range(cols.shape[0]):
    #     distance = 20
    #     error_image[:10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    #
    # save_name = '/home/wy/文档/RAFT_4resolution1(wy)/error_map/' + str(batch_i) + '.png'
    # imageio.imsave(save_name, (error_image * 256).astype('uint8'))
    # plt.imshow(error_image)
    # plt.show()

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }
    return loss.data.cpu(), metrics


def main():
    PER_FREQ = 1
    for epoch in range(0, 1):
        print("Parameter Count: %d" % count_parameters(model))
        print('Training with %d image pairs' % len(train_dataset))
        print('This is %d-th epoch' % (epoch))

        # total_train_loss = 0.0
        # running_train_loss = {}
        # ## training ##
        # for batch_idx, data_blob in enumerate(TrainImgLoader):
        #     imgL_crop, imgR_crop, disp_crop_L = [x.cuda() for x in data_blob]
        #     loss, metrics = train(imgL_crop, imgR_crop, disp_crop_L)
        #
        #     # print('Iter %d train loss = %.3f, lr=%.7f' % (batch_idx, loss, scheduler.get_last_lr()[0]))
        #     total_train_loss += loss
        #     update_metrics_status(metrics, running_train_loss)
        #
        # metrics_train_data = [running_train_loss[k] / len(TrainImgLoader) for k in sorted(running_train_loss.keys())]
        #
        # print('total train loss = %.3f' % (total_train_loss / len(TrainImgLoader)))
        # print(("{:10.4f}," * len(metrics_train_data)).format(*metrics_train_data))
        #
        # # SAVE
        # if epoch % PER_FREQ == PER_FREQ - 1:
        #     PATH = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'train_loss': total_train_loss / len(TrainImgLoader),
        #         'epe': metrics_train_data[3]
        #     }, PATH)
        #     torch.save(model.state_dict(), PATH)
        # logger.close()

        # ------------- TEST ------------------------------------------------------------

        ##test ##
        if epoch % PER_FREQ == PER_FREQ - 1:
            print('Testing with %d image pairs' % len(test_dataset))
            total_test_loss = 0.0
            running_test_loss = {}
            for batch_i, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, metrics_test = test(imgL, imgR, disp_L)
                print('Iter %d test loss = %.3f, lr=%.7f' % (batch_i, test_loss, scheduler.get_last_lr()[0]))
                total_test_loss += test_loss
                update_metrics_status(metrics_test, running_test_loss)

            metrics_test_data = [running_test_loss[k] / len(TestImgLoader) for k in sorted(running_test_loss.keys())]

            print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
            print(("{:10.4f}," * len(metrics_test_data)).format(*metrics_test_data))
            ###----------------------------------------------------------------------------------
            ### SAVE test information

            # if epoch % PER_FREQ == PER_FREQ-1:
            #     PATH = args.savemodel + 'testinformation.tar'
            #     torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    main()
