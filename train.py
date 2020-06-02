import os
import argparse
import time
import math
import numpy as np
from data import DataGenerator
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import SVLRM as Net
from loss import Loss 
from utils import save_model, print_network
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# set flags / seeds
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for Depth SR")
    parser.add_argument('--data_dir', type=str, default='./dataset/')
    parser.add_argument('--upscaling_factor', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--data_augment', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--snapshot_dir', type=str, default='./weights/')
    parser.add_argument('--snapshot', type=int, default=5000)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=20000000)
    parser.add_argument('--prinf_freq', type=int, default=10)
    parser.add_argument('--test_data_dir', type=str, default='./dataset/test_depth_sr/')
    parser.add_argument('--result_save_dir', type=str, default='./dataset/results/')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str, default='./weights/X8/model_001.pth')

    opt = parser.parse_args()
    print(opt)
    # logger = Logger(os.path.join(opt.log_dir, 'training.txt'))
 
    scale = 'X' + str(opt.upscaling_factor)
    tblog_dir = os.path.join(opt.log_dir, scale)
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir, comment='batch_20_lr_1e-4_150w_epoch')

    def train(training_data_loader, net, optim, criterion):
        net.train()
        curr_step = 0
        while curr_step < opt.n_iters:
            for _, batch in enumerate(training_data_loader, 1):
                curr_step += 1
                img_gt, img_input, img_guided = batch[0], batch[1], batch[2]

                img_gt = img_gt.type(torch.FloatTensor).cuda()
                img_input = img_input.type(torch.FloatTensor).cuda()
                img_guided = img_guided.type(torch.FloatTensor).cuda()
                
                t0 = time.time()
                img_pred, _, _ = net(img_input, img_guided)
                loss = criterion(img_pred, img_gt) / opt.batch_size
                t1 = time.time()
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                # learning rate is decayed with poly policy
                lr_ = opt.lr * (1 - float(curr_step)/opt.n_iters)**2
                for param_group in optim.param_groups:
                    param_group['lr'] = lr_
                
                if curr_step % opt.prinf_freq == 0:
                    print("===> Iters({}/{}): lr:{} || Loss: {:.4f} || Timer: {:.4f} sec.".format(curr_step, opt.n_iters, optim.param_groups[0]['lr'], loss.item(), (t1 - t0)))
                    writer.add_scalar('Training_loss', loss.item(), curr_step)


                if curr_step % opt.snapshot == 0:
                    save_model(net, curr_step, opt.snapshot_dir, opt.upscaling_factor)

    # add code for datasets
    print("===> Loading datasets")
    data_set = DataGenerator(data_dir= opt.data_dir,
                upscaling_factor = opt.upscaling_factor,
                patch_size = opt.patch_size,
                data_aug = opt.data_augment)
    training_data = DataLoader(dataset=data_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, pin_memory=True)
    
    # instantiate network
    print("===> Building model")
    devices_ids = list(range(opt.n_gpus))
    net = Net()

    # create losses
    criterion_L1_cb = Loss(eps=1e-3)
    criterion_L1_cb = criterion_L1_cb.cuda()
 
    # if running on GPU and we want to use cuda move model there
    print("===> Setting GPU")
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    net = nn.DataParallel(net, device_ids=devices_ids)
    net = net.cuda()
    

    print('---------- Networks architecture -------------')
    print_network(net)
    print('----------------------------------------------')

    # optionally ckp from a checkpoint
    if opt.resume:
        if opt.resume_dir != None:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_state_dict(torch.load(opt.resume_dir))
            else:
                net.load_state_dict(torch.load(opt.resume_dir))
            print('Net work loaded from {}'.format(opt.resume_dir))

    # create optimizers
    print("===> Setting Optimizer")
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    
print("===> Training")
train(training_data, net, optim, criterion_L1_cb)
writer.close()   
        