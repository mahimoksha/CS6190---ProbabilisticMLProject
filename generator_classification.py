import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = None
        self.stride = stride

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Conv3x3_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num, stride, dropout_p=0.5):
        super(Conv3x3_BN_RELU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=3,
                                          stride=stride, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))
        if dropout_p != 0:
            self.add_module('dropout', nn.Dropout2d(p=dropout_p))
class Conv3x3_IN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num, stride, dropout_p=0.5):
        super(Conv3x3_IN_RELU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=3,
                                          stride=stride, padding=1, bias=False))
        self.add_module('in', nn.InstanceNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))
        if dropout_p != 0:
            self.add_module('dropout', nn.Dropout2d(p=dropout_p))


class Conv1x1_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num, stride, dropout_p=0.5):
        super(Conv1x1_BN_RELU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=1,
                                          stride=stride, padding=0, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))
        if dropout_p != 0:
            self.add_module('dropout', nn.Dropout2d(p=dropout_p))


class Deconv3x3_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num, stride):
        super(Deconv3x3_BN_RELU, self).__init__()
        if stride == 1:
            output_padding = 0
        elif stride == 2:
            output_padding = 1

        self.add_module('conv', nn.ConvTranspose2d(in_num, out_num, kernel_size=3,
                                                   stride=stride, padding=1,
                                                   output_padding=output_padding,
                                                   bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))


class FC_BN_RELU(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(FC_BN_RELU, self).__init__()
        self.add_module('fc', nn.Linear(in_num, out_num))
        self.add_module('bn', nn.BatchNorm1d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', nn.LeakyReLU(0.2, inplace=True))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ndf = 32
        self.main = nn.Sequential(
            # input is (nc) x 450 x 450
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 225 x 225
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 112 x 112
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 56 x 56
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 28 x 28
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



class cGAN_generator(nn.Module):

    def __init__(self, gen, noise_dim=16, d=64):
        super(cGAN_generator, self).__init__()

        print('using conditional GAN')
        # [1, 16, 32, 64, 128, 256]
        self.gen  = gen
        channels = d
        dropout_p = 0.5

        self.x_conv = nn.Sequential(
                Conv3x3_IN_RELU(3, d, 1),
                BasicBlock(d, 2*d, 2, norm_layer=nn.InstanceNorm2d),
                BasicBlock(2*d, 4*d, 2, norm_layer=nn.InstanceNorm2d),
                BasicBlock(4*d, 8*d, 2, norm_layer=nn.InstanceNorm2d),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                BasicBlock(8*d, 4*d, 1, norm_layer=nn.InstanceNorm2d),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                BasicBlock(4*d, d, 1, norm_layer=nn.InstanceNorm2d),
                )


        self.z_linear = nn.Linear(noise_dim, 128)
        self.z_deconv1 = nn.ConvTranspose2d(128,  d * 16, 4, 1, 0)
        self.z_bn1 = nn.InstanceNorm2d(d * 16)
        self.deconv_convs = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv3x3_IN_RELU(d*16, d*8, 1, dropout_p=dropout_p), # 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv3x3_IN_RELU(d*8, d*4, 1, dropout_p=dropout_p), # 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv3x3_IN_RELU(d*4, d*2, 1, dropout_p=dropout_p), # 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv3x3_IN_RELU(d*2, d, 1, dropout_p), # 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv3x3_IN_RELU(d, d, 1, dropout_p), # 128x128
        )

        self.conv_c = nn.Sequential(
            BasicBlock(2*d, d, 1, norm_layer=nn.InstanceNorm2d),
            BasicBlock(d, d, 1, norm_layer=nn.InstanceNorm2d)
            )

        if self.gen == 'color':
            self.output = nn.Conv2d(d, 1, 1, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.output.weight.data.zero_()
        self.output.bias.data.zero_()


    def forward(self, noise, x):
        x_feat = self.x_conv(x)
        B, C, H, W = x.shape

        z_feat = self.z_linear(noise).view(B, -1, 1, 1)
        z_feat = F.relu(self.z_bn1(self.z_deconv1(z_feat)))
        z_feat = self.deconv_convs(z_feat)

        x_feat = F.interpolate(x_feat, size=x.shape[2:], align_corners=True, mode='bilinear')
        z_feat = F.interpolate(z_feat, size=x.shape[2:], align_corners=True, mode='bilinear')

        c_feat = torch.cat([x_feat, z_feat], dim=1)

        c_feat = self.conv_c(c_feat)
        x = c_feat

        output = self.output(x)

        return output


    def gradientLoss(self, flow, penalize='l2'):

        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

        if penalize == 'l2':
            d = torch.mean(dx**2) + torch.mean(dy**2)
        else:
            d = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

        return d / 2.0



class augnet(nn.Module):

    def __init__(self, gen, d_range, n_range):
        super(augnet, self).__init__()

        self.gen = gen
        self.d_range = d_range
        self.n_range = n_range

        self.aug_net = cGAN_generator(self.gen,16, d=4)

        self.id_theta = torch.tensor([[1, 0, 0],
                                      [0, 1, 0]], dtype=torch.float32).cuda()


    def forward(self, noise, x, label, require_loss=False, require_delta=False):

        B = x.shape[0]
        if self.gen == 'color':
            delta = self.n_range * F.tanh(self.aug_net(noise, x)).cuda()

            x_tf = x + delta

        if self.gen == 'color':
            reg_loss = torch.mean(torch.abs(torch.pow(delta, 2)))

        if require_delta:
            if require_loss:
                return x_tf, delta, reg_loss
            else:
                return x_tf, delta

        else:
            if require_loss:
                return x_tf, reg_loss
            else:
                return x_tf

