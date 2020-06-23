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
from utils import save_model, print_network, calc_rmse_tensor, save_img
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
    parser.add_argument('--data_dir', type=str, default='./dataset/train_data')
    parser.add_argument('--upscaling_factor', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--data_augment', type=bool, default=True)
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--snapshot_dir', type=str, default='./weights/')
    parser.add_argument('--snapshot', type=int, default=100)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=500000)
    parser.add_argument('--prinf_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=200)
    parser.add_argument('--val_data_dir', type=str, default='./dataset/val_data/')
    parser.add_argument('--result_save_dir', type=str, default='./results/val_results/')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str, default='./weights/X8/model_001.pth')

    opt = parser.parse_args()
    print(opt)

    scale = 'X{}'.format(opt.upscaling_factor)
    tblog_dir = os.path.join(opt.log_dir, scale)
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir, comment='batch_20_lr_1e-4_50w_epoch')

    # add code for datasets
    print("===> Loading datasets")
    data_set = DataGenerator(data_dir= opt.data_dir,
                upscaling_factor = opt.upscaling_factor,
                patch_size = opt.patch_size,
                data_aug = opt.data_augment,
                crop = opt.crop
                )
    val_data_set = DataGenerator(data_dir= opt.val_data_dir, 
                                upscaling_factor = opt.upscaling_factor, 
                                data_aug =False, 
                                crop = False)

    training_data = DataLoader(dataset=data_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True)
    val_data = DataLoader(dataset=val_data_set, batch_size=1, num_workers=opt.workers, shuffle=False)
    
    # instantiate network
    print("===> Building model")
    devices_ids = list(range(opt.n_gpus))
    net = Net()

    # if running on GPU and we want to use cuda move model there
    print("===> Setting GPU")
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    net = nn.DataParallel(net, device_ids=devices_ids)
    net = net.cuda()

    # create loss
    criterion_L1_cb = Loss(eps=1e-3)
    criterion_L1_cb = criterion_L1_cb.cuda()
    
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

    # create optimizer
    print("===> Setting Optimizer")
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    
    print("===> Training")
    curr_step = 0
    while curr_step < opt.n_iters:
        net.train()
        
        # learning rate is decayed with poly policy
        lr_ = opt.lr * (1 - float(curr_step)/opt.n_iters)**2
        for param_group in optim.param_groups:
            param_group['lr'] = lr_

        for _, batch in enumerate(training_data, 1):
            curr_step += 1
            target, lr_data, guided_data = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)

            target = target.cuda()
            lr_data = lr_data.cuda()
            # guided_data = guided_data.type(torch.FloatTensor).cuda()
            guided_data = guided_data.cuda()
            
            optim.zero_grad()
            t0 = time.time()
            img_pred, _, _ = net(lr_data, guided_data)
            loss = criterion_L1_cb(img_pred, target) / opt.batch_size
            t1 = time.time()
            
            loss.backward()
            optim.step()

            if curr_step % opt.prinf_freq == 0:
                print("===> Iters({}/{}): lr:{:.6f} || Loss: {:.4f} || Timer: {:.4f} sec.".format(curr_step, opt.n_iters, optim.param_groups[0]['lr'], loss.item(), (t1 - t0)))
                writer.add_scalar('Training_loss', loss.item(), curr_step)


            if curr_step % opt.val_freq == 0:
                print('Evaluation....')
                net.eval()
                mean_rmse = 0
                n_count, n_total = 1, len(val_data)
                for batch in val_data:
                    target, lr_data, guided_data = batch[0], batch[1], batch[2]
                    lr_data = lr_data.type(torch.FloatTensor).cuda()
                    guided_data = guided_data.type(torch.FloatTensor).cuda()
                    target = target.type(torch.FloatTensor).cuda() 

                    with torch.no_grad():
                        results, _, _ = net(lr_data, guided_data)

                    rmse = calc_rmse_tensor(results, target)
                    mean_rmse += rmse

                    image_name = val_data_set.imagefilenames[n_count-1]
                    print("===> Processing: {}/{}".format(n_count, n_total))
                    save_img(results.cpu(), image_name, opt.result_save_dir, opt.upscaling_factor)
                    n_count += 1
                mean_rmse /= len(val_data)
                writer.add_scalar('Mean_RMSE', mean_rmse, curr_step)
                print("Valid  iter [{}] || rmse: {}".format(curr_step, mean_rmse))
                    
            if curr_step % opt.snapshot == 0:
                save_model(net, curr_step, opt.snapshot_dir, opt.upscaling_factor)
    writer.close()   
        
