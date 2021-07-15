import os
import os.path
import cv2
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils_lab.data_io import get_transform, pfm_imread

from torchvision.transforms import ColorJitter
import torch.utils.data as data
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


# SceneFlow_disparity
def disparity_loader_SceneFlow(path):
    return pfm_imread(path)


# KITTI_disparity
def disparity_loader_KITTI(path):
    return Image.open(path)


def dataloader_KITTI(filepath):
    left_fold = 'colored_0/'
    right_fold = 'colored_1/'
    disp_noc = 'disp_occ/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    train = image[:]
    val = image[:]

    left_train = [filepath + left_fold + img for img in train]
    right_train = [filepath + right_fold + img for img in train]
    disp_train = [filepath + disp_noc + img for img in train]

    left_val = [filepath + left_fold + img for img in val]
    right_val = [filepath + right_fold + img for img in val]
    disp_val = [filepath + disp_noc + img for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val


def dataloader_KITTI2015(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    train = image[:180]
    val = image[:]

    left_train = [filepath + left_fold + img for img in train]
    right_train = [filepath + right_fold + img for img in train]
    disp_train_L = [filepath + disp_L + img for img in train]
    # disp_train_R = [filepath+disp_R+img for img in train]

    left_val = [filepath + left_fold + img for img in val]
    right_val = [filepath + right_fold + img for img in val]
    disp_val_L = [filepath + disp_L + img for img in val]
    # disp_val_R = [filepath+disp_R+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    # return left_train, right_train, disp_train_L


def dataloader_SceneFlow(filepath):
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

    monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

    monkaa_dir = os.listdir(monkaa_path)

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
            if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
                all_left_img.append(monkaa_path + '/' + dd + '/left/' + im)
                all_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split(".")[0] + '.pfm')

        for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
            if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
                all_right_img.append(monkaa_path + '/' + dd + '/right/' + im)

    flying_path = filepath + [x for x in image if x == 'frames_finalpass'][0]
    flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    flying_dir = flying_path + '/TRAIN/'
    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    flying_dir = flying_path + '/TEST/'

    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir + ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
            for im in imm_l:
                if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                    test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                    test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                        all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                    all_left_disp.append(
                        driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                        all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


class myImageFloder_SceneFlow(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader,
                 dploader=disparity_loader_SceneFlow):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

        else:
            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)


class myImageFloder_KITTI(data.Dataset):

    def __init__(self, left, right, left_disparity, training, loader=default_loader,
                 dploader=disparity_loader_KITTI):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.min_scale = -0.2
        self.max_scale = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        mean_color = np.mean(img2.reshape(-1, 3), axis=0)
        for _ in range(np.random.randint(1, 3)):
            x0 = np.random.randint(0, wd)
            y0 = np.random.randint(0, ht)
            dx = np.random.randint(bounds[0], bounds[1])
            dy = np.random.randint(bounds[0], bounds[1])
            img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            #photometric augmentation
            left_img, right_img = self.color_transform(left_img, right_img)
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            # erase
            if random.random() < 0.5:
                left_img, right_img = self.eraser_transform(left_img, right_img)

            ## randomly sample scale
            # if random.random() < 0.5:
            #     min_scale = np.maximum(
            #         (th + 1) / float(h),
            #         (tw + 1) / float(w))
            
            #     scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            #     scale_x = np.clip(scale, min_scale, None)
            #     scale_y = np.clip(scale, min_scale, None)
            
            #     left_img = cv2.resize(left_img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     right_img = cv2.resize(right_img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     dataL = cv2.resize(dataL, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     dataL = dataL * scale_x

            h_new, w_new = left_img.shape[:2]
            # random crop
            x1 = random.randint(0, w_new - tw)
            y1 = random.randint(0, h_new - th)
            left_img = left_img[y1:y1 + th, x1:x1 + tw]
            right_img = right_img[y1:y1 + th, x1:x1 + tw]

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            #to tensor, normalize
            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            # vertical flip
            if random.random() < 0.5:
                left_img = torch.flip(left_img, [1])
                right_img = torch.flip(right_img, [1])
                dataL = np.ascontiguousarray(np.flip(dataL, 0))

            return left_img, right_img, dataL

        # if self.training:  
        #    w, h = left_img.size
        #    th, tw = 256, 512

        #     ## erase
        #     # if random.random() < 0.50 :
        #     #     left_img, right_img = self.eraser_transform(left_img, right_img)

        #    x1 = random.randint(0, w - tw)
        #    y1 = random.randint(0, h - th)

        #    left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
        #    right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

        #    dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
        #    dataL = dataL[y1:y1 + th, x1:x1 + tw]

        #    processed = get_transform(augment=False)  
        #    left_img   = processed(left_img)
        #    right_img  = processed(right_img)

        #     if random.random() < 0.50:  
        #         left_img = torch.flip(left_img, [1])
        #         right_img = torch.flip(right_img, [1])
        #         dataL = np.ascontiguousarray(np.flip(dataL, 0))

        #    return left_img, right_img, dataL

        else:
            w, h = left_img.size
            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))

            dataL = dataL.crop((w - 1232, h - 368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
