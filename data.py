import os
import os.path as path
from os import listdir
import glob
import typing
import cv2
import functools
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def modcrop(img, scale):
    # img: numpy, HWC or HW
    img = np.copy(img)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, _ = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def shave(img, border=0):
    # img: numpy, HWC or HW
    img = np.copy(img)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img
    
def aug_data(img, rot_flip):
    # img: numpy, HWC
    assert 0 <= rot_flip <= 7
    rot_flag = rot_flip // 2
    flip_flag = rot_flip % 2

    if rot_flag > 0:
        img = np.rot90(img, k=rot_flag, axes=(0, 1))
    if flip_flag >= 0:
        img = np.flip(img, axis=0)
    return img

class DataGenerator(Dataset):
    def __init__(self, data_dir='/root/proj/SVLRM/dataset/train_data',
                upscaling_factor = 4,
                patch_size = None,
                data_aug = True,
                crop = True
                ):
        # 1. Initialize file paths or a list of file names.
        super(DataGenerator, self).__init__() 

        assert path.isdir(data_dir)

        self.data_dir =  data_dir
        self.upscaling_factor = upscaling_factor
        self.patch_size = patch_size
        self.data_augment = data_aug
        self.crop = crop

        imagefilenames = glob.glob(path.join(self.data_dir, 'depth_gt', '*'))
        imagefilenames = [path.split(x)[1] for x in imagefilenames if is_image_file(x)]
        imagefilenames.sort()
        self.imagefilenames = imagefilenames

            
    def __getitem__(self, index):
        file_index = self.imagefilenames[index]
        gt_data = self._read_png(file_index, 'depth_gt')
        gt_data = cv2.cvtColor(gt_data, cv2.COLOR_BGR2GRAY)
        # # normalization
        # gt_data = gt_data.astype(np.float64) / 255.
        
        guided_data = self._read_png(file_index, 'guided')
        guided_data = cv2.cvtColor(guided_data, cv2.COLOR_BGR2GRAY)
        # # normalization
        # guided_data = guided_data.astype(np.float64) / 255.
        
        # 2. Preprocess the data
        H, W = gt_data.shape
        gt_data = modcrop(gt_data, self.upscaling_factor)
        guided_data = modcrop(guided_data, self.upscaling_factor)

        if self.crop:
            sh = np.random.randint(0, H - self.patch_size + 1)
            sw = np.random.randint(0, W - self.patch_size + 1)

            # print(f'sh {sh:4d}, sw {sw:4d} | gt {gt_data.shape} | input {lr_data.shape} | guided {guided_data.shape}', end=' | ')
            gt_data = gt_data[sh : sh + self.patch_size, sw : sw + self.patch_size]
            # lr_data = lr_data[sh : sh + self.patch_size, sw : sw + self.patch_size]
            guided_data = guided_data[sh : sh + self.patch_size, sw : sw + self.patch_size]

        if self.data_augment:
            rot_flip = np.random.randint(0, 8) 
            gt_data = aug_data(gt_data, rot_flip).copy()
            # lr_data = aug_data(lr_data, rot_flip).copy()
            guided_data = aug_data(guided_data, rot_flip).copy()
        
        lr_data = gt_data[self.upscaling_factor-1::self.upscaling_factor, self.upscaling_factor-1::self.upscaling_factor]
        lr_data = cv2.resize(lr_data, dsize=(0, 0), fx=self.upscaling_factor, fy=self.upscaling_factor, interpolation=cv2.INTER_CUBIC)
        
        gt_data = np.expand_dims(gt_data, axis=2)
        lr_data = np.expand_dims(lr_data, axis=2)
        guided_data = np.expand_dims(guided_data, axis=2)

        # to tensor
        gt_data = ToTensor()(gt_data) # Converts a numpy.ndarray (H x W x C) [0, 255] to a torch.FloatTensor of shape (C x H x W) [0.0, 1.0]
        lr_data = ToTensor()(lr_data)
        guided_data = ToTensor()(guided_data)

        # 3. Return a data pair 
        return gt_data, lr_data, guided_data

    def __len__(self):
        return len(self.imagefilenames) 

    @functools.lru_cache(maxsize=128)
    def _read_png(self, file_index, dtype):
        data = cv2.imread(self._png_path(file_index, dtype))
        return data

    def _png_path(self, file_index, dtype):
        if dtype == 'depth_gt':
            return path.join(self.data_dir, 'depth_gt', file_index)
        if dtype == 'guided':
            return path.join(self.data_dir, 'images', file_index)

if __name__=='__main__':
    ds = DataGenerator(patch_size=64)
    print(len(ds))
    for i in range(0, 5):
        data = ds[i]
        print(data[0].shape, data[0])
        print(data[1].shape, data[1])
        print(data[2].shape, data[2])
