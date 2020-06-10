import os
import glob
import os.path as path
import cv2
import numpy as np

def load_img(file_path, type):
    dir_path = os.path.join(os.getcwd(), file_path)
    img_path = glob.glob(os.path.join(dir_path, type))
    return img_path

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    return image

def data_aug(rgb_path = '', depth_path = '', savepath = 'RGBD_data'):
    rgb_path = load_img(rgb_path,'*.png')
    depth_path = load_img(depth_path, '*.png')
    rgb_path.sort()
    depth_path.sort()
    
    print(rgb_path)
    print(depth_path)
    save_path = savepath

    if not path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(rgb_path)):
        rgb_image = read_img(rgb_path[i])
        depth_image = read_img(depth_path[i])
        rgb = np.array(rgb_image)
        depth = np.array(depth_image)
        rgbd = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgbd[:, :, 0] = rgb[:, :, 0]
        rgbd[:, :, 1] = rgb[:, :, 1]
        rgbd[:, :, 2] = rgb[:, :, 2]
        rgbd[:, :, 3] = depth
        print('data no{:03d}.'.format(i))
        file_name = savepath + '/' + '{:03d}.png'.format(i+1)
        
        cv2.imwrite(file_name , rgbd)
        print('---------saving---------')

if __name__ == '__main__':
    print('starting data augmentation...')
    rgb_path = '/root/proj/SVLRM/dataset/images'
    depth_path = '/root/proj/SVLRM/dataset/depth_gt'
    savepath = 'RGBD_data_train'
    data_aug(rgb_path, depth_path, savepath)
