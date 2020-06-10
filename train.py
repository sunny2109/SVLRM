import os
import argparse
import time
import math
import numpy as np
from data_loader.data import get_training_set, get_validation_set
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import SVLRM as Net
from loss import Loss 
from utils import save_model, print_network, calc_rmse_tensor, save_img
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
    parser.add_argument('--data_dir', type=str, default='./dataset/RGBD')
    parser.add_argument('--upscaling_factor', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--crop', action='store_true', help='whether to crop?')
    parser.add_argument('--data_augment', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--snapshot_dir', type=str, default='./weights/')
    parser.add_argument('--snapshot', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--decay_step', type=int, default=2000)
    parser.add_argument('--prinf_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=50)
    parser.add_argument('--result_save_dir', type=str, default='./results/val_results/')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str, default='./weights/X8/model_001.pth')

    opt = parser.parse_args()
    print(opt)

    scale = 'X' + str(opt.upscaling_factor)
    tblog_dir = os.path.join(opt.log_dir, scale)
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir, comment='batch_20_lr_1e-4_200w_epoch')

    # add code for datasets
    print("===> Loading datasets")
    train_set = get_training_set(dataset=opt.data_dir, upscale_factor=opt.upscaling_factor, patch_size=opt.patch_size, crop=opt.crop)
    validation_set = get_validation_set(dataset=opt.data_dir, upscale_factor=opt.upscaling_factor)

    training_data = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset=validation_set, batch_size=1, num_workers=opt.workers, shuffle=False)
    
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
   
    for epoch in range(opt.start_epoch, opt.total_epochs + 1):
        # learning rate is decayed by 2 every 40 epochs
        lr_ = opt.lr * (0.5 ** (epoch // opt.decay_step))
        for param_group in optim.param_groups:
            param_group['lr'] = lr_
        
        print("epoch =", epoch, "lr =", optim.param_groups[0]["lr"])
        net.train()
        
        epoch_loss = 0
        for iteration, batch in enumerate(training_data, 1):
            # img_gt, img_input, img_guided = batch[0], batch[1], batch[2]
            input_, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

            input_ = input_.cuda()
            target = target.cuda()
            # img_guided = img_guided.type(torch.FloatTensor).cuda()
            
            optim.zero_grad()
            t0 = time.time()
            # param = net(input)
            # output = param[:, :1, :, :] * input[:, 1, :, :] + param[:, 1:, :, :]
            output, _, _ = net(input_)
            loss = criterion_L1_cb(output, target) / opt.batch_size
            epoch_loss += loss.item()
            
            loss.backward()
            optim.step()

            if iteration % opt.prinf_freq == 0:
                print("===> Epoch[{}/{}]({}/{}): lr:{} || Loss: {:.4f}.".format(epoch, opt.total_epochs, iteration, len(training_data), 
                                                                                optim.param_groups[0]['lr'], loss.item()))

        t1 = time.time()
        print("===> Epoch{}: AVG Loss:{:.4f} || Timer: {:.4f} sec".format(epoch, epoch_loss / len(training_data), (t1 - t0)))
        writer.add_scalar("Epoch Loss", epoch_loss / len(training_data), epoch)
        if epoch % opt.snapshot == 0:
            save_model(net, epoch, opt.snapshot_dir, opt.upscaling_factor)

        if epoch % opt.val_freq == 0:
                print('Evaluation....')
                net.eval()
                mean_rmse = 0
                n_count, n_total = 1, len(val_data)
                for batch in val_data:
        
                    input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
                    input = input.cuda()
                    target = target.cuda()

                    with torch.no_grad():
                        results, _, _ = net(input)
                    
                    rmse = calc_rmse_tensor(results, target)
                    mean_rmse += rmse

                    print("===> Processing: {}/{}".format(n_count, n_total))
                    save_img(results.cpu(), n_count, opt.result_save_dir, opt.upscaling_factor)
                    n_count += 1
                mean_rmse /= len(val_data)
                writer.add_scalar('Mean_RMSE', mean_rmse, epoch)
                print("Valid_epoch [{}] || rmse: {:.4f}".format(epoch, mean_rmse))
    writer.close()   
