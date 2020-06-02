import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from data import testDataGenerator
from torch.utils.data import DataLoader
from model import SVLRM as Net
from utils import save_img

if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for Depth SR")
    parser.add_argument('--upscaling_factor', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--data_augment', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--test_data_dir', type=str, default='./dataset/test_depth_sr/')
    parser.add_argument('--result_save_dir', type=str, default='./results/')
    parser.add_argument('--result_alpha_dir', type=str, default='./results/alpha')
    parser.add_argument('--result_beta_dir', type=str, default='./results/beta')
    parser.add_argument('--model', default='./weights/X8/model_000800_epoch.pth', help='sr pretrained base model')

    opt = parser.parse_args()
    print(opt)

    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # add code for datasets (we always use train and validation/ test set)
    print("===> Loading datasets")
    data_set = testDataGenerator(data_dir= opt.test_data_dir, upscaling_factor = opt.upscaling_factor)
    test_data = DataLoader(dataset=data_set, batch_size=opt.test_batch_size, num_workers=opt.workers, shuffle=False)

    # instantiate network (which has been imported from *networks.py*)
    print("===> Building model")
    devices_ids = list(range(opt.n_gpus))
    net = Net()
 
    # if running on GPU and we want to use cuda move model there
    print("===> Setting GPU")
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    net = nn.DataParallel(net, device_ids=devices_ids).cuda()
    
    net.load_state_dict(torch.load(opt.model), strict=True)
    print('Pre-trained SR model is loaded.')

    def eval():
        net.eval()
        n_count, n_total = 1, len(test_data)
        for batch in test_data:
            img_input, img_guided = batch[0], batch[1]
            img_input = img_input.type(torch.FloatTensor).cuda()
            img_guided = img_guided.type(torch.FloatTensor).cuda() 

            t0 = time.time()
            with torch.no_grad():
                results, param_alpha, param_beta = net(img_input, img_guided)
            t1 = time.time()
            image_name = data_set.imagefilenames[n_count-1]
            print("===> Processing: {}/{} || Timer: {} sec.".format(n_count, n_total, (t1 - t0)))
            save_img(results.cpu(), image_name, opt.result_save_dir, opt.upscaling_factor)
            save_img(param_alpha.cpu(), image_name, opt.result_alpha_dir, opt.upscaling_factor)
            save_img(param_beta.cpu(), image_name, opt.result_beta_dir, opt.upscaling_factor)
            n_count += 1
        print('Done!')
eval()


