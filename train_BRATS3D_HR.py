from __future__ import print_function
import argparse
import os, sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from MEGAN.networks3D_HR import define_D, GANLoss, define_G
from MEGAN.multiscale_patch_dataset import MedicalImagePatches3D, PEDataLoader
from MEGAN.image_utils import save_image
from MEGAN.network_utils import print_network, set_requires_grad

"""Train patchwise resolution upgrading for each scale>0. We are aiming to learn high resolution patches (HRP) 
from low resolution patches (LRP) and high resolution sketch"""
# Training settings
parser = argparse.ArgumentParser(description='Training patchwise high resolution')
parser.add_argument('--data_root', type=str, default='../Data',
                    help='root directory of the dataset')
parser.add_argument('--results_root', type=str, default='../Results',
                    help='root directory for saving results')
parser.add_argument('--experiment_type', type=str, default='SKETCH2BRATST23D',
                    help='name of experiment (also name of folders)')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--patch_size', type=int, default=32, help='training patch size')
parser.add_argument('--LR_size', type=int, default=64,
                    help='LR image size (image size of full resolution low-resolution images)')
parser.add_argument('--HR_size', type=int, default=128,
                    help='HR image size (image size of full resolution high-resolution images)')
parser.add_argument('--level_name', type=str, default='HR1',
                    help='creates an output folder with this name for each scale')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=float, default=10, help='weight on L1 term in objective')
parser.add_argument('--device', type=int, default=0, help='device nr (def: 0)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Define whether to start from scratch (epoch 0) or used saved epochs')

opt = parser.parse_args()

print(opt)

batch_size = opt.batch_size

cudnn.benchmark = True
device = torch.device('cuda', opt.device)
with torch.cuda.device(opt.device):
    print('===> Loading datasets')
    input_dir = os.path.join(opt.data_root, opt.experiment_type)
    output_dir = os.path.join(opt.results_root, opt.experiment_type + '_' + opt.level_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    val_out_dir = os.path.join(output_dir, 'Val')
    if not os.path.exists(val_out_dir):
        os.makedirs(val_out_dir)

    traindataset = MedicalImagePatches3D(input_dir + '/A/train/data_list.txt', input_dir + '/B/train/data_list.txt',
                                         sample_size=20, patch_shape=(opt.patch_size, opt.patch_size, opt.patch_size),
                                         LR_size=(opt.LR_size, opt.LR_size, opt.LR_size),
                                         HR_size=(opt.HR_size, opt.HR_size, opt.HR_size))

    training_data_loader = PEDataLoader(traindataset,
                                        batch_size=batch_size, shuffle=True,
                                        num_workers=8, pin_memory=False)

    validdataset = MedicalImagePatches3D(input_dir + '/A/train/data_list_valid.txt',
                                         input_dir + '/B/train/data_list_valid.txt',
                                         sample_size=4, patch_shape=(opt.patch_size, opt.patch_size, opt.patch_size),
                                         LR_size=(opt.LR_size, opt.LR_size, opt.LR_size),
                                         HR_size=(opt.HR_size, opt.HR_size, opt.HR_size))
    valid_data_loader = PEDataLoader(validdataset,
                                     batch_size=1, shuffle=False,
                                     num_workers=8, pin_memory=False)
    print('===> Building model')
    net_G = define_G(2, 1, 64).to(device)
    net_D = define_D(3, 64, use_sigmoid=False).to(device)

    # load weights from epoch start_epoch to start with
    if opt.start_epoch > 0:
        snapshot_str = '_epoch_' + str(opt.start_epoch) + '_size' + str(opt.LR_size) + '_to_' + str(
            opt.HR_size) + '_' + opt.level_name
        snapshot_str_G = output_dir + '/NETG' + snapshot_str
        snapshot_str_D = output_dir + '/NETD' + snapshot_str
        net_G.load_state_dict(
            torch.load(snapshot_str_G + '.pth',
                       map_location=lambda storage, loc: storage))
        net_D.load_state_dict(
            torch.load(
                snapshot_str_D + '.pth',
                map_location=lambda storage, loc: storage))

    # setup losses
    criterion_GAN = GANLoss(device=device, use_lsgan=True)
    criterion_L1 = nn.L1Loss()
    # setup optimizer
    optimizer_G = optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print('---------- Networks initialized -------------')
    print_network(net_G)
    print_network(net_D)
    print('-----------------------------------------------')

valid_losses = []
# initialize validation loss with a large value
valid_losses.append(1000)


def validate(epoch, best_epoch):
    avg_fake_D = 0
    avg_real_D = 0
    loss_g_l1 = 0
    net_D.eval()
    net_G.eval()
    for i, batch in enumerate(valid_data_loader):
        HRP = batch[0].to(device)
        LRP = batch[1].to(device)
        HREP = batch[2].to(device)

        prediction = net_G((LRP, HREP))

        sketch_img_real = torch.cat((HREP, HRP), 1)
        sketch_img_fake = torch.cat((HREP, prediction), 1)
        avg_fake_D += torch.mean(net_D((sketch_img_fake, LRP)).detach())
        avg_real_D += torch.mean(net_D((sketch_img_real, LRP)).detach())
        loss_g_l1 += criterion_L1(prediction, HRP).item()

        # save validation images
        save_image(127.5 * (prediction.data[0, 0, :, :, :].cpu().numpy() + 1),
                   val_out_dir + '/producedB_%04d_epoch_%04d.nii' % ((i + 1), epoch))
        save_image(127.5 * (HREP.data[0, 0, :, :, :].cpu().numpy() + 1),
                   val_out_dir + '/originalA_%04d_epoch_%04d.nii' % ((i + 1), epoch))
        save_image(127.5 * (HRP.data[0, 0, :, :, :].cpu().numpy() + 1),
                   val_out_dir + '/originalB_%04d_epoch_%04d.nii' % ((i + 1), epoch))
    print("Valid: ===> Avg. D fake: {:.4f} ; Avg. D real: {:.4f}; Avg. G L1: {:.4f}".
          format(avg_fake_D / len(valid_data_loader), avg_real_D / len(valid_data_loader),
                 loss_g_l1 / len(valid_data_loader)))

    if loss_g_l1 <= min(valid_losses):
        best_epoch = epoch
    valid_losses.append(loss_g_l1)
    return best_epoch


def train(epoch):
    net_D.train()
    net_G.train()
    for i, batch in enumerate(training_data_loader):
        ############################################
        ## HRP = High resolution image patch      ##
        ## LRP = Low resolution image patch       ##
        ## HREP = High resolution edge image patch##
        ############################################
        HRP = batch[0].to(device)
        LRP = batch[1].to(device)
        HREP = batch[2].to(device)
        LR_aug = batch[4].to(device)

        noise = torch.empty_like(HREP).detach()
        nn.init.normal_(noise, 0, 0.01)
        HREP = HREP + noise

        # forward G
        HRP_fake = net_G((LR_aug, HREP))

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        set_requires_grad(net_D, True)
        fake_D = net_D.forward((HREP.detach(), HRP_fake.detach(), LRP.detach()))
        loss_d_fake = criterion_GAN(fake_D.view(batch_size, -1), False)

        real_D = net_D.forward((HREP, HRP, LRP))
        loss_d_real = criterion_GAN(real_D.view(batch_size, -1), True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        # only update D when a threshold is passed
        d_real_acu = torch.ge((real_D).squeeze(), 0.5).float()
        d_fake_acu = torch.le((fake_D).squeeze(), 0.5).float()
        d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

        if d_total_acu <= 0.8:
            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        set_requires_grad(net_D, False)
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        # First, G(A) should fake the discriminator
        fake_D = net_D.forward((HREP, HRP_fake, LRP))
        loss_g_gan = criterion_GAN(fake_D.view(batch_size, -1), True)

        # Second, G(A) = B
        loss_g_l1 = criterion_L1(HRP_fake, HRP) * opt.lamb
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizer_G.step()

    print("{} ===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
        time.ctime(time.time()), epoch, i, len(training_data_loader), loss_d.data[0], loss_g.data[0]))


with torch.cuda.device(opt.device):
    best_epoch = 0
    for epoch in range(opt.start_epoch+1, opt.start_epoch + opt.n_epochs + 2):
        snapshot_str = '_epoch_' + str(epoch) + '_size' + str(opt.LR_size) + '_to_' + str(
            opt.HR_size) + '_' + opt.level_name
        snapshot_str_G = output_dir + '/NETG' + snapshot_str
        snapshot_str_D = output_dir + '/NETD' + snapshot_str
        train(epoch)
        # validate every 20 epochs
        if epoch % 20 == 0:
            if len(valid_data_loader)>0:
                best_epoch = validate(epoch, best_epoch)
            if epoch == best_epoch:
                torch.save(net_G.state_dict(),
                           output_dir + '/NETG_best.pth')
                torch.save(net_D.state_dict(),
                           output_dir + '/NETD_best.pth')

            torch.save(net_G.state_dict(),
                       snapshot_str_G + '.pth')
            torch.save(net_D.state_dict(),
                       snapshot_str_D + '.pth')
            # early termination if no best epch appears in the last 100 epochs
            if epoch >= 200 and best_epoch <= epoch - 100:
                print('Early Termination at epoch: ' + str(epoch))
                break
