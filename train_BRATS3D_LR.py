from __future__ import print_function
import argparse
import os,sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from MEGAN.networks3D_LR import define_D, GANLoss, define_G_LR
from MEGAN.dataset import MedicalImageDataset3D
from MEGAN.network_utils import print_network,set_requires_grad
from MEGAN.image_utils import save_image

"""Train generating scale 0. We are aiming to learn a low-resolution image (LRI) from a low-resolution image edges (LRE)"""
# Training settings
parser = argparse.ArgumentParser(description='Training lowest resolution')
parser.add_argument('--data_root', type=str, default='../Data', help='root directory of the dataset')
parser.add_argument('--results_root',type=str, default='../Results',help='root directory for saving results')
parser.add_argument('--experiment_type',type=str,default='SKETCH2BRATST23D',
                    help='name of experiment (also name of folders)')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--img_size', type=int, default=64, help='size of downsampled image')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--device', type=int, default=0, help="Device nr (def: 0)")
parser.add_argument('--pixel_loss', type=str,default='L1',help='pixel level loss. Possible: L1, MSE')
parser.add_argument('--start_epoch',type=int, default=0,
                    help='Define whether to start from scratch (epoch 0) or used saved epochs')
parser.add_argument('--scale_edges',type=int, default=2,
                    help='factor for scaling edge image. If < 2, no scaling is done.')
parser.add_argument('--level_name', type=str, default='LR',
                    help='creates an output folder with this name for each scale')

opt = parser.parse_args()

print(opt)

batch_size=opt.batch_size
cudnn.benchmark = True
torch.manual_seed(opt.seed)
device=torch.device('cuda', opt.device)

with torch.cuda.device(opt.device):
    print('===> Loading datasets')
    input_dir = os.path.join(opt.data_root, opt.experiment_type)
    output_dir=os.path.join(opt.results_root, opt.experiment_type + '_' + opt.level_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    val_out_dir=os.path.join(output_dir, 'Val')
    if not os.path.exists(val_out_dir):
        os.makedirs(val_out_dir)

    #we chose the edge image to be 2x the LR image size sind  otherwise edges disappear
    if opt.scale_edges>1:
        new_size_edges=opt.scale_edges*opt.img_size
    traindataset=MedicalImageDataset3D(input_dir, mode="train", listname='data_list.txt', new_size_imgs=opt.img_size,
                                       new_size_edges=new_size_edges, augmentation=0)
    training_data_loader = DataLoader(traindataset,
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=False)
    validdataset=MedicalImageDataset3D(input_dir, mode="train", listname='data_list_valid.txt', new_size_imgs=opt.img_size,
                                       new_size_edges=new_size_edges)
    valid_data_loader = DataLoader(validdataset,
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=False)


    print('===> Building model')
    # define nets
    net_G=define_G_LR(1, 1, norm='instance', scale=(opt.scale_edges>1)).to(device)
    net_D = define_D(2, 64, norm='batch', gpu_ids=[opt.device], scale=(opt.scale_edges>1)).to(device)
    # load weights from epoch start_epoch to start with
    if opt.start_epoch>0:
        snapshot_str = '_epoch_' + str(opt.start_epoch) + '_size' + str(opt.img_size)
        snapshot_str_G = output_dir + '/NETG' + snapshot_str
        snapshot_str_D = output_dir + '/NETD' + snapshot_str
        net_G.load_state_dict(
            torch.load(snapshot_str_G + '.pth',
                       map_location=lambda storage, loc: storage))
        net_D.load_state_dict(
            torch.load(snapshot_str_D + '.pth',
                       map_location=lambda storage, loc: storage))

    # define losses
    criterion_GAN = GANLoss(device=device, use_lsgan=True).to(device)
    criterion_L1 = nn.L1Loss().to(device)
    criterion_MSE = nn.MSELoss().to(device)

    # setup optimizer
    optimizer_G = optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    print('---------- Networks initialized -------------')
    print_network(net_G)
    print_network(net_D)
    print('-----------------------------------------------')

# initialize validation loss with a large value
valid_losses=[10000]
def validate(epoch, best_epoch):
    avg_fake_D=0
    avg_real_D=0
    loss_g_l1=0

    for i,batch in enumerate(valid_data_loader):
        sketch_small_real = batch['A'].cuda().to(device)
        img_small_real = batch['B'].cuda().to(device)

        net_G.eval()
        net_D.eval()
        prediction = net_G(sketch_small_real)
        avg_fake_D+= torch.sigmoid(torch.mean(net_D((sketch_small_real, prediction)))).item()
        avg_real_D+= torch.sigmoid(torch.mean(net_D((sketch_small_real, img_small_real)))).item()
        loss_g_l1 += criterion_L1(prediction, img_small_real).item()
        img_small_real_np=img_small_real.data[0,0,:,:,:].cpu().numpy()
        prediction_np=np.around(prediction.data[0,0,:,:,:].cpu().numpy(),4).astype(img_small_real_np.dtype)

        save_image(127.5 * prediction_np, val_out_dir + '/producedB_%04d_epoch_%04d.nii' % ((i + 1), epoch))
        save_image(127.5 * (np.around(sketch_small_real.data[0, 0, :, :, :].cpu().numpy(), 4)), val_out_dir + '/originalA_%04d_epoch_%04d.nii' % ((i + 1), epoch))
        save_image(127.5 * (img_small_real.data[0, 0, :, :, :].cpu().numpy()), val_out_dir + '/originalB_%04d_epoch_%04d.nii' % ((i + 1), epoch))
    print("Valid: ===> Avg. D fake: {:.4f} ; Avg. D real: {:.4f}; Avg. G L1: {:.4f}".
          format(avg_fake_D/ len(valid_data_loader),avg_real_D/ len(valid_data_loader),loss_g_l1/len(valid_data_loader)))

    if loss_g_l1<=min(valid_losses):
        best_epoch=epoch
    valid_losses.append(loss_g_l1)
    return best_epoch


def train(epoch):
    net_G.train()
    for i, batch in enumerate(training_data_loader):
        sketch_small_real = batch['A'].cuda().to(device)
        img_small_real = batch['B'].cuda().to(device)
        currbatch_size=sketch_small_real.shape[0]
        noise=torch.empty_like(sketch_small_real).detach()
        nn.init.normal_(noise,0,0.01)
        sketch_small_real=sketch_small_real+noise
        # forward generator
        img_small_fake = net_G(sketch_small_real)
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        set_requires_grad(net_D, True)
        fake_D = net_D.forward((sketch_small_real.detach(), img_small_fake.detach()))
        loss_d_fake = criterion_GAN(fake_D.view(currbatch_size, -1), False)

        real_D = net_D.forward((sketch_small_real, img_small_real))
        loss_d_real = criterion_GAN(real_D.view(currbatch_size, -1), True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        # only update D when a threshold is passed
        d_real_acu = torch.ge(F.sigmoid(real_D).squeeze(), 0.5).float()
        d_fake_acu = torch.le(F.sigmoid(fake_D).squeeze(), 0.5).float()
        d_total_acu = torch.mean(torch.cat((d_real_acu.detach(), d_fake_acu.detach()), 0))
        if d_total_acu <= 0.8:
            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()
        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        net_D.zero_grad()
        set_requires_grad(net_D, False)
        optimizer_G.zero_grad()
        # First, G(A) should fake the discriminator
        fake_D = net_D.forward((sketch_small_real, img_small_fake))
        loss_g_gan = criterion_GAN(fake_D.view(currbatch_size, -1), True)

        # Second, G(A) = B
        if opt.pixel_loss=='MSE':
            pixel_loss= criterion_MSE(img_small_fake, img_small_real) * opt.lamb
        else:
            pixel_loss= criterion_L1(img_small_fake, img_small_real) * opt.lamb
        loss_g = pixel_loss+loss_g_gan
        loss_g.backward()
        optimizer_G.step()

    print("{} ===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
        time.ctime(time.time()), epoch, i, len(training_data_loader), loss_d.item(), loss_g.item()))

with torch.cuda.device(opt.device):
    best_epoch=0
    for epoch in range(opt.start_epoch+1, opt.start_epoch+opt.n_epochs + 2):
        snapshot_str = '_epoch_' + str(epoch) + '_size' + str(opt.img_size)
        snapshot_str_G = output_dir + '/NETG' + snapshot_str
        snapshot_str_D = output_dir + '/NETD' + snapshot_str
        train(epoch)
        if epoch %20== 0:
            if len(valid_data_loader) > 0:
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
            if epoch >=300  and best_epoch <= epoch - 100:
                    print('Early Termination at epoch: ' + str(epoch))
                    break
