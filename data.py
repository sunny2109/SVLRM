import os
import os.path as path
from os import listdir
import glob
import typing
import cv2
import functools
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
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

def shave(img_in, border=0):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img

def aug_data(img_in, rot_flip):
    # img_in: Numpy, CHW
    assert 0 <= rot_flip <= 7
    rot_times = rot_flip // 2
    flip_times = rot_flip % 2
    if rot_times > 0:
        img_in = np.rot90(img_in, k=rot_times, axes=(-2, -1))
    if flip_times > 0:
        img_in = np.flip(img_in, axis=-2)
    return img_in

class DataGenerator(Dataset):
    def __init__(self, data_dir='./dataset/',
                upscaling_factor = 4,
                patch_size = None,
                data_aug = True
                ):
        # 1. Initialize file paths or a list of file names.
        super(DataGenerator, self).__init__() 

        assert path.isdir(data_dir)

        self.data_dir =  data_dir
        self.upscaling_factor = upscaling_factor
        self.patch_size = patch_size
        self.data_augment = data_aug

        imagefilenames = glob.glob(path.join(self.data_dir, 'depth_gt', '*'))
        imagefilenames = [path.split(x)[1] for x in imagefilenames if is_image_file(x)]
        imagefilenames.sort()
        self.imagefilenames = imagefilenames

            
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        file_index = self.imagefilenames[index]
        gt_data = self._read_png(file_index, 'depth_gt')
        gt_data = cv2.cvtColor(gt_data, cv2.COLOR_BGR2GRAY)
        gt_data = gt_data.astype(np.float32) /255.0
        
        guided_data = self._read_png(file_index, 'guided')
        guided_data = cv2.cvtColor(guided_data, cv2.COLOR_BGR2GRAY)
        guided_data = guided_data.astype(np.float32) / 255.0
        
        gt_data = modcrop(gt_data, self.upscaling_factor)
        guided_data = modcrop(guided_data, self.upscaling_factor)

        H, W = gt_data.shape

        lr_data = gt_data[0::self.upscaling_factor, 0::self.upscaling_factor]
        input_data = cv2.resize(lr_data, dsize=(0,0), fx=self.upscaling_factor, fy=self.upscaling_factor, interpolation=cv2.INTER_CUBIC)
        
        sh = np.random.randint(0, H - self.patch_size + 1)
        sw = np.random.randint(0, W - self.patch_size + 1)

        # print(f'sh {sh:4d}, sw {sw:4d} | gt {gt_data.shape} | input {input_data.shape} | guided {guided_data.shape}', end=' | ')
        img_gt = gt_data[sh : sh + self.patch_size, sw : sw + self.patch_size]
        img_input = input_data[sh : sh + self.patch_size, sw : sw + self.patch_size]
        img_guided = guided_data[sh : sh + self.patch_size, sw : sw + self.patch_size]

        img_gt = np.expand_dims(img_gt, axis=2)
        img_input = np.expand_dims(img_input, axis=2)
        img_guided = np.expand_dims(img_guided, axis=2)

        # Transpose HWC to CHW
        img_gt = np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))
        img_input = np.ascontiguousarray(np.transpose(img_input, (2, 0, 1)))
        img_guided = np.ascontiguousarray(np.transpose(img_guided, (2, 0, 1)))

        if self.data_augment:
            rot_flip = np.random.randint(0, 8)  # 0 <= rot_flip <= 7
            # img_gt = aug_data(img_gt, rot_flip).copy()
            img_input = aug_data(img_input, rot_flip).copy()
            img_guided = aug_data(img_guided, rot_flip).copy()

        # 3. Return a data pair (e.g. image and label).
        return img_gt, img_input, img_guided

    def __len__(self):
        # You should change 0 to the total size of your dataset.
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

class testDataGenerator(Dataset):
    def __init__(self, data_dir='./dataset/test_depth_sr/',
                upscaling_factor = 4,
                ):
        # 1. Initialize file paths or a list of file names.
        super(testDataGenerator, self).__init__() 

        assert path.isdir(data_dir)

        self.data_dir =  data_dir
        self.upscaling_factor = upscaling_factor

        imagefilenames = glob.glob(path.join(self.data_dir, 'depth_gt', '*'))
        imagefilenames = [path.split(x)[1] for x in imagefilenames if is_image_file(x)]
        imagefilenames.sort()
        self.imagefilenames = imagefilenames

            
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        file_index = self.imagefilenames[index]
        gt_data = self._read_png(file_index, 'depth_gt')
        gt_data = gt_data.astype(np.float32) / 255.0
        
        guided_data = self._read_png(file_index, 'guided')
        guided_data = guided_data.astype(np.float32) / 255.0
        

        # 2. Preprocess the data (e.g. torchvision.Transform).
        gt_data = modcrop(gt_data, self.upscaling_factor)
        guided_data = modcrop(guided_data, self.upscaling_factor)

        lr_data = gt_data[0::self.upscaling_factor, 0::self.upscaling_factor]
        input_data = cv2.resize(lr_data, dsize=(0,0), fx=self.upscaling_factor, fy=self.upscaling_factor, interpolation=cv2.INTER_CUBIC)

        input_data = np.expand_dims(input_data, axis=2)
        guided_data = np.expand_dims(guided_data, axis=2)

        # Transpose HWC to CHW
        img_input = np.ascontiguousarray(np.transpose(input_data, (2, 0, 1)))
        img_guided = np.ascontiguousarray(np.transpose(guided_data, (2, 0, 1)))

        # 3. Return a data pair (e.g. image and label).
        return img_input, img_guided

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imagefilenames) 

    @functools.lru_cache(maxsize=128)
    def _read_png(self, file_index, dtype):
        data = cv2.imread(self._png_path(file_index, dtype), cv2.IMREAD_GRAYSCALE)
        return data

    def _png_path(self, file_index, dtype):
        if dtype == 'depth_gt':
            return path.join(self.data_dir, 'depth_gt', file_index)
        if dtype == 'guided':
            return path.join(self.data_dir, 'images', file_index)

if __name__=='__main__':
    ds = DataGenerator(patch_size=64)
    print(len(ds))
    for i in range(5):
        data = ds[i]
        print(data[0].shape, data[0])
        print(data[1].shape, data[1])
        print(data[2].shape, data[2])

    # for i in range(2):
    #     print("The gt image of {0} batch image: {1}, and the shape is:{2}".format(i+1, ds[i][0], ds[i][0].shape))
    #     print("The guided image of {0} batch image: {1}, and the shape is:{2}".format(i+1, ds[i][1], ds[i][1].shape))
    #     print("The input image of {0} batch image: {1}, and the shape is:{2}".format(i+1, ds[i][2], ds[i][2].shape))

    # print(ds[2][1])

    # for i in range(len(ds)):
    # for i in range(600, 650):
    #     data = ds[i]
    #     print(f"{i}  |  gt {data[0].shape} |  inpt {data[2].shape}  |  guide {data[1].shape}")

    # training_data = DataLoader(ds, batch_size=20, num_workers=8, shuffle=True)
    # print(len(training_data))
    # print('total size:{}'.format(len(training_data) / 20))
    # test_data = testDataGenerator()
    # print(len(test_data))