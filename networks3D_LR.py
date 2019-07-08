import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from MEGAN.network_utils import get_norm_layer


class UNet_G_LR(nn.Module):
    """Unet Generator for low-resolution images"""

    # initializers
    def __init__(self, d=64, c=1, upsample=False, scale=False):
        super(UNet_G_LR, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv3d(c, d, 4, 2, 1)
        self.conv2 = nn.Conv3d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.InstanceNorm3d(d * 2)
        self.conv3 = nn.Conv3d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.InstanceNorm3d(d * 4)
        self.conv4 = nn.Conv3d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.InstanceNorm3d(d * 8)
        self.conv5 = nn.Conv3d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.InstanceNorm3d(d * 8)
        if not upsample:
            self.deconv3 = nn.ConvTranspose3d(d * 8, d * 8, 4, 2, 1)
            self.deconv3_bn = nn.InstanceNorm3d(d * 8)
            self.deconv4 = nn.ConvTranspose3d(d * 8, d * 4, 4, 2, 1)
            self.deconv4_bn = nn.InstanceNorm3d(d * 8)
            self.deconv5 = nn.ConvTranspose3d(d * 4 * 2, d * 2, 4, 2, 1)
            self.deconv5_bn = nn.InstanceNorm3d(d * 4)
            self.deconv6 = nn.ConvTranspose3d(d * 2 * 2, d, 4, 2, 1)
            self.deconv6_bn = nn.InstanceNorm3d(d * 2)
            self.deconv7 = nn.ConvTranspose3d(d * 2, 1, 4, 2, 1)
        else:
            self.deconv3 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                         nn.Conv3d(d * 8, d * 8, kernel_size=1))
            self.deconv3_bn = nn.InstanceNorm3d(d * 8)
            self.deconv4 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                         nn.Conv3d(d * 8 * 2, d * 8, kernel_size=1))
            self.deconv4_bn = nn.InstanceNorm3d(d * 8)
            self.deconv5 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                         nn.Conv3d(d * 8 * 2, d * 4, kernel_size=1))
            self.deconv5_bn = nn.InstanceNorm3d(d * 4)
            self.deconv6 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                         nn.Conv3d(d * 4 * 2, d * 2, kernel_size=1))
            self.deconv6_bn = nn.InstanceNorm3d(d * 2)
            self.deconv7 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                         nn.Conv3d(d * 2 * 2, 1, kernel_size=1))

        self.dropout1 = nn.Dropout3d(p=0.2)
        self.dropout2 = nn.Dropout3d(p=0.2)

        self.scale = scale

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        if self.scale:
            e4 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        d4 = self.deconv4_bn(self.deconv4(F.relu(e4)))
        d4 = torch.cat([d4, e3], 1)
        d5 = self.dropout2(self.deconv5_bn(self.deconv5(F.relu(d4))))
        d5 = torch.cat([d5, e2], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e1], 1)
        d7 = self.deconv7(F.relu(d6))
        o = F.tanh(d7)
        return o


# Defines the GAN loss which uses either LSGAN or the regular GAN.
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, device=torch.device('cuda', 0)):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.device = device
        self.Tensor = tensor
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda().to(self.device))


# Defines the PatchGAN discriminator.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d, gpu_ids=[], scale=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.scale = scale
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                          padding=padw), norm_layer(ndf * nf_mult,
                                                    affine=True), nn.LeakyReLU(0.2, True)
            ]
        self.n_B = 128
        self.n_C = 16
        nf_mult_prev = nf_mult
        T_ten_init = torch.randn(ndf * nf_mult_prev * 4 * 4 * 4, self.n_B * self.n_C) * 0.1
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)
        self.fc = nn.Linear(ndf * nf_mult_prev * 4 * 4 * 4 + self.n_B, 1)

        self.model = nn.Sequential(*sequence)

        self.conv1 = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        if self.scale:
            input = torch.cat((self.conv1(input[0]), input[1]), 1)
        else:
            input = torch.cat(input, 1)
        feature = self.model(input)
        return feature


def define_D(input_nc, ndf, norm='batch', gpu_ids=[], scale=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, gpu_ids=gpu_ids, scale=scale)

    if use_gpu:
        netD.cuda(gpu_ids[0])
    return netD


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=True, n_blocks=6,
                 has_conv=True, scale=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        if scale:
            model += [nn.Conv3d(ngf, ngf, kernel_size=4, stride=2, padding=1),
                      norm_layer(ngf, affine=True),
                      nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=4,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]
        if has_conv:
            model += [nn.Conv3d(ngf, output_nc, kernel_size=7, stride=1, padding=3)]
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        assert (padding_type == 'zero')
        p = 1

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def define_G_LR(input_nc, output_nc=1, ngf=64, norm='batch', n_blocks=6, use_dropout=True, scale=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks,
                           has_conv=True, scale=scale)
    return netG
