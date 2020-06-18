import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import RandomCrop, ToTensor, Resize, ToPILImage, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation

def downsampling(img, upscale_factors):
    img_array = np.asarray(img)
    lr_image = img_array[upscale_factors-1::upscale_factors, upscale_factors-1::upscale_factors]
    lr_image = Image.fromarray(lr_image)

    return lr_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGBA')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor=4, patch_size=64, crop=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.image_filenames.sort()

        self.crop = crop
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index]) #input is preprocessed, the shape is []
        
        # data argumentation
        if self.crop:
            input = RandomCrop(self.patch_size)(input)  #obtain an image patch
            input = RandomHorizontalFlip()(input) 
            input = RandomVerticalFlip()(input) 
            input = RandomRotation(180)(input)
        
        input_tensor = ToTensor()(input)
        rgb_tensor = torch.zeros(3,input_tensor.shape[1],input_tensor.shape[2])
        depth_tensor = torch.zeros(1, input_tensor.shape[1], input_tensor.shape[2])
        
        rgb_tensor[0, :, :] = input_tensor[0, :, :]
        rgb_tensor[1, :, :] = input_tensor[1, :, :]
        rgb_tensor[2, :, :] = input_tensor[2, :, :]
        depth_tensor[0, :, :] = input_tensor[3, :, :]
        
        depth = ToPILImage()(depth_tensor)
        size = min(depth.size[0], depth.size[1])
        guide = ToPILImage()(rgb_tensor)
        target = depth.copy()

        guide = guide.convert('L')
        
        # Generating LR images
        depth = downsampling(depth,self.upscale_factor)
        depth = Resize(size=size,interpolation=Image.BICUBIC)(depth)

        depth = ToTensor()(depth)
        guide = ToTensor()(guide)
        depth = torch.cat((depth, guide), dim=0) #concatenating for input
        target = ToTensor()(target)

        return depth, target

    def __len__(self):
        return len(self.image_filenames)
