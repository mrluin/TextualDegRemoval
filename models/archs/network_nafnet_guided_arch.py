# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.nafnet_arch_utils import LayerNorm2d
from models.archs.nafnet_local_arch import Local_Base


#########################################################
# code in MASA-SR
def pixelUnshuffle(x, r=1):
    b, c, h, w = x.size()
    out_chl = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    x = x.view(b, c, out_h, r, out_w, r)
    out = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_chl, out_h, out_w)

    return out


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


class SAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out


class Encoder(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Encoder, self).__init__()

        # block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(functools.partial(ResidualBlock, nf=nf), n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf*2**1, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(functools.partial(ResidualBlock, nf=nf*2**1), n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf*2**1, nf*2**2, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(functools.partial(ResidualBlock, nf=nf*2**2), n_layers=n_blks[2])

        self.conv_L4 = nn.Conv2d(nf*2**2, nf*2**3, 3, 2, 1, bias=True)
        self.blk_L4 = make_layer(functools.partial(ResidualBlock, nf=nf * 2 ** 3), n_layers=n_blks[2])

        self.conv_L5 = nn.Conv2d(nf*2**3, nf*2**4, 3, 2, 1, bias=True)
        self.blk_L5 = make_layer(functools.partial(ResidualBlock, nf=nf * 2 ** 4), n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))
        fea_L4 = self.blk_L4(self.act(self.conv_L4(fea_L3)))
        fea_L5 = self.blk_L5(self.act(self.conv_L5(fea_L4)))

        return [fea_L1, fea_L2, fea_L3, fea_L4, fea_L5]


class DRAM(nn.Module):
    def __init__(self, nf):
        super(DRAM, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out


##########################################

class SimpleGate(nn.Module):
    def forward(self, x):

        # x or x, ref
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFResFuseBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        # inp concat[x, ref]
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetRefFusion(nn.Module):
    def __init__(self,
                 img_channel=3,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[],

                 nf=64,
                 ext_n_blocks=[4,4,4,4],
                 reffusion_n_blocks=[1,1,1,1],
                 reffusion_n_blocks_middle=1,
                 scale=1,
                 num_nbr=1,
                 psize=3,
                 lr_block_size=8,
                 ref_down_block_size=1.5,
                 dilations=[1,2,3]):
        super().__init__()

        ############
        nf = nf
        ext_n_blocks = ext_n_blocks
        reffusion_n_blocks = reffusion_n_blocks
        reffusion_n_blocks_middle = reffusion_n_blocks_middle

        self.scale = scale
        self.num_nbr = num_nbr
        self.psize = psize
        self.lr_block_size = lr_block_size
        self.ref_down_block_size = ref_down_block_size
        self.dilations = dilations

        self.masa_enc = Encoder(in_chl=img_channel, nf=nf, n_blks=ext_n_blocks)
        self.masa_blk_enc = nn.ModuleList()
        self.masa_blk_middle = nn.ModuleList()
        self.masa_blk_dec = nn.ModuleList()

        ##############

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for index, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            self.masa_blk_enc.append(
                nn.Sequential(*[NAFResFuseBlock(chan * 2) for _ in range(reffusion_n_blocks[index])])
            )

            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        self.masa_blk_middle.append(
            nn.Sequential(*[NAFResFuseBlock(chan * 2) for _ in range(reffusion_n_blocks[index + 1])])
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W]
        # dim: scalar > 0
        # index: [N, Hi, Wi]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]  # views = [N, 1, -1]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1  # expanse = [-1, C*k*k, -1]
        index = index.clone().view(views).expand(expanse)  # [N, Hi, Wi] -> [N, 1, Hi*Wi] - > [N, C*k*k, Hi*Wi]
        return torch.gather(input, dim, index)  # [N, C*k*k, Hi*Wi]

    def search_org(self, lr, reflr, ks=3, pd=1, stride=1):
        # lr: [N, C, H, W]
        # reflr: [N, C, Hr, Wr]

        batch, c, H, W = lr.size()
        _, _, Hr, Wr = reflr.size()

        reflr_unfold = F.unfold(reflr, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, Hr*Wr]
        lr_unfold = F.unfold(lr, kernel_size=(ks, ks), padding=0, stride=stride)
        lr_unfold = lr_unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        lr_unfold = F.normalize(lr_unfold, dim=2)
        reflr_unfold = F.normalize(reflr_unfold, dim=1)

        corr = torch.bmm(lr_unfold, reflr_unfold)  # [N, H*W, Hr*Wr]
        corr = corr.view(batch, H - 2, W - 2, (Hr - 2) * (Wr - 2))
        sorted_corr, ind_l = torch.topk(corr, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, H, W, num_nbr]

        return sorted_corr, ind_l

    def search(self, lr, reflr, ks=3, pd=1, stride=1, dilations=[1, 2, 4]):
        # lr: [N, p*p, C, k_y, k_x]
        # reflr: [N, C, Hr, Wr]

        N, C, Hr, Wr = reflr.size()
        _, _, _, k_y, k_x = lr.size()
        x, y = k_x // 2, k_y // 2
        corr_sum = 0
        for i, dilation in enumerate(dilations):
            reflr_patches = F.unfold(reflr, kernel_size=(ks, ks), padding=dilation, stride=stride, dilation=dilation)  # [N, C*ks*ks, Hr*Wr]
            lr_patches = lr[:, :, :, y - dilation: y + dilation + 1: dilation,
                                     x - dilation: x + dilation + 1: dilation]  # [N, p*p, C, ks, ks]
            lr_patches = lr_patches.contiguous().view(N, -1, C * ks * ks)  # [N, p*p, C*ks*ks]

            lr_patches = F.normalize(lr_patches, dim=2)
            reflr_patches = F.normalize(reflr_patches, dim=1)
            corr = torch.bmm(lr_patches, reflr_patches)  # [N, p*p, Hr*Wr]
            corr_sum = corr_sum + corr

        sorted_corr, ind_l = torch.topk(corr_sum, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, p*p, num_nbr]

        return sorted_corr, ind_l

    def transfer(self, fea, index, soft_att, ks=3, pd=1, stride=1):
        # fea: [N, C, H, W]
        # index: [N, Hi, Wi]
        # soft_att: [N, 1, Hi, Wi]
        scale = stride

        fea_unfold = F.unfold(fea, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, H*W]
        out_unfold = self.bis(fea_unfold, 2, index)  # [N, C*k*k, Hi*Wi]
        divisor = torch.ones_like(out_unfold)

        _, Hi, Wi = index.size()
        out_fold = F.fold(out_unfold, output_size=(Hi * scale, Wi * scale), kernel_size=(ks, ks), padding=pd,
                          stride=stride)
        divisor = F.fold(divisor, output_size=(Hi * scale, Wi * scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        soft_att_resize = F.interpolate(soft_att, size=(Hi * scale, Wi * scale), mode='bilinear')
        out_fold = out_fold / divisor * soft_att_resize
        # out_fold = out_fold / (ks*ks) * soft_att_resize
        return out_fold

    def make_grid(self, idx_x1, idx_y1, diameter_x, diameter_y, s):
        idx_x1 = idx_x1 * s
        idx_y1 = idx_y1 * s
        idx_x1 = idx_x1.view(-1, 1).repeat(1, diameter_x * s)
        idx_y1 = idx_y1.view(-1, 1).repeat(1, diameter_y * s)
        idx_x1 = idx_x1 + torch.arange(0, diameter_x * s, dtype=torch.long, device=idx_x1.device).view(1, -1)
        idx_y1 = idx_y1 + torch.arange(0, diameter_y * s, dtype=torch.long, device=idx_y1.device).view(1, -1)

        ind_y_l = []
        ind_x_l = []
        for i in range(idx_x1.size(0)):
            grid_y, grid_x = torch.meshgrid(idx_y1[i], idx_x1[i])
            ind_y_l.append(grid_y.contiguous().view(-1))
            ind_x_l.append(grid_x.contiguous().view(-1))
        ind_y = torch.cat(ind_y_l)
        ind_x = torch.cat(ind_x_l)

        return ind_y, ind_x

    def check_image_size(self, x):

        _, _, h, w = x.size()
        padder_size = self.padder_size * self.lr_block_size

        mod_pad_h = (padder_size - h % padder_size) % padder_size
        mod_pad_w = (padder_size - w % padder_size) % padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

        return x

    def forward(self, inp, ref):

        _, _, ori_H, ori_W = inp.shape
        inp = self.check_image_size(inp)
        ref = self.check_image_size(ref)

        # print(inp.shape, ref.shape)

        ############################# MASA Search ################################

        _, _, h, w = inp.size()

        # start from the deepest feature, patches info on LR
        px = w // self.padder_size // self.lr_block_size
        py = h // self.padder_size // self.lr_block_size

        k_x = w // self.padder_size // px
        k_y = h // self.padder_size // py

        # print(px, py, k_x, k_y)  # px py 3, k_x k_y 8

        _, _, h, w = ref.size()

        # sxd, block info on Ref
        diameter_x = 2 * int(w // self.padder_size // (2 * px) * self.ref_down_block_size) + 1
        diameter_y = 2 * int(h // self.padder_size // (2 * py) * self.ref_down_block_size) + 1

        # print(diameter_x, diameter_y)  # diameter_x, diameter_y 13

        # extract multi-scale feature from both LR and Ref
        feat_lq = self.masa_enc(inp)
        feat_ref = self.masa_enc(ref)

        # start from the deepest feature
        N, C, H, W = feat_lq[4].size()
        _, _, Hr, Wr = feat_ref[4].size()

        # print(feat_lq[4].shape, feat_ref[4].shape)

        # unfold LR into patches and find correlated patches on Ref
        lr_patches = F.pad(feat_lq[4], pad=(1, 1, 1, 1), mode="replicate")
        lr_patches = F.unfold(lr_patches, kernel_size=(k_y + 2, k_x + 2), padding=(0, 0), stride=(k_y, k_x))
        lr_patches = lr_patches.view(N, C, k_y + 2, k_x + 2, py * px).permute(0, 4, 1, 2, 3)

        # calculate center patch similarity between LR and Ref, with dilations
        sorted_corr, ind_l = self.search(lr_patches, feat_ref[4], ks=3, pd=1, stride=1, dilations=self.dilations)

        # left, right, top, down, with block size
        index = ind_l[:, :, 0]
        idx_x = index % Wr
        idx_y = index // Wr
        idx_x1 = idx_x - diameter_x // 2 - 1
        idx_x2 = idx_x + diameter_x // 2 + 1
        idx_y1 = idx_y - diameter_y // 2 - 1
        idx_y2 = idx_y + diameter_y // 2 + 1

        mask = (idx_x1 < 0).long()
        idx_x1 = idx_x1 * (1 - mask)
        idx_x2 = idx_x2 * (1 - mask) + (diameter_x + 1) * mask

        mask = (idx_x2 > Wr - 1).long()
        idx_x2 = idx_x2 * (1 - mask) + (Wr - 1) * mask
        idx_x1 = idx_x1 * (1 - mask) + (idx_x2 - (diameter_x + 1)) * mask

        mask = (idx_y1 < 0).long()
        idx_y1 = idx_y1 * (1 - mask)
        idx_y2 = idx_y2 * (1 - mask) + (diameter_y + 1) * mask

        mask = (idx_y2 > Hr - 1).long()
        idx_y2 = idx_y2 * (1 - mask) + (Hr - 1) * mask
        idx_y1 = idx_y1 * (1 - mask) + (idx_y2 - (diameter_y + 1)) * mask

        ind_y_x1, ind_x_x1 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 1)
        ind_y_x2, ind_x_x2 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 2)
        ind_y_x4, ind_x_x4 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 4)
        ind_y_x8, ind_x_x8 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 8)
        ind_y_x16, ind_x_x16 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 16)

        ind_b = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * (diameter_y + 2) * (diameter_x + 2))
        ind_b_x2 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y + 2) * 2) * ((diameter_x + 2) * 2))
        ind_b_x4 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y + 2) * 4) * ((diameter_x + 2) * 4))
        ind_b_x8 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y + 2) * 8) * (diameter_x + 2) * 8)
        ind_b_x16 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y + 2) * 16) * (diameter_y + 2) * 16)

        # block on ref
        ref_patches = feat_ref[4][ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2, C).permute(0, 3, 1, 2).contiguous()  # [N*py*px, C, (radius_y+1)*2, (radius_x+1)*2]

        ref_patches_x1 = feat_ref[4][ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2, C).permute(0, 3, 1, 2).contiguous()
        ref_patches_x2 = feat_ref[3][ind_b_x2, :, ind_y_x2, ind_x_x2].view(N * py * px, (diameter_y + 2) * 2, (diameter_x + 2) * 2, C // 2).permute(0, 3, 1, 2).contiguous()
        ref_patches_x4 = feat_ref[2][ind_b_x4, :, ind_y_x4, ind_x_x4].view(N * py * px, (diameter_y + 2) * 4, (diameter_x + 2) * 4, C // 4).permute(0, 3, 1, 2).contiguous()
        ref_patches_x8 = feat_ref[1][ind_b_x8, :, ind_y_x8, ind_x_x8].view(N * py * px, (diameter_y + 2) * 8, (diameter_x + 2) * 8, C // 8).permute(0, 3, 1, 2).contiguous()
        ref_patches_x16 = feat_ref[0][ind_b_x16, :, ind_y_x16, ind_x_x16].view(N * py * px, (diameter_y + 2) * 16, (diameter_x + 2) * 16, C // 16).permute(0, 3, 1, 2).contiguous()

        # patches on LR
        lr_patches = lr_patches.contiguous().view(N * py * px, C, k_y + 2, k_x + 2)
        # calculate similarity between LR patches within Ref blocks
        corr_all_l, index_all_l = self.search_org(lr_patches, ref_patches, ks=self.psize, pd=self.psize // 2, stride=1)

        index_all = index_all_l[:, :, :, 0]
        soft_att_all = corr_all_l[:, :, :, 0:1].permute(0, 3, 1, 2)

        # block -> patches -> transfer
        warp_ref_patches_x1 = self.transfer(ref_patches_x1, index_all, soft_att_all, ks=self.psize, pd=self.psize // 2, stride=1)
        warp_ref_patches_x2 = self.transfer(ref_patches_x2, index_all, soft_att_all, ks=self.psize * 2, pd=self.psize // 2 * 2, stride=2)
        warp_ref_patches_x4 = self.transfer(ref_patches_x4, index_all, soft_att_all, ks=self.psize * 4, pd=self.psize // 2 * 4, stride=4)
        warp_ref_patches_x8 = self.transfer(ref_patches_x8, index_all, soft_att_all, ks=self.psize * 8, pd=self.psize // 2 * 8, stride=8)
        warp_ref_patches_x16 = self.transfer(ref_patches_x16, index_all, soft_att_all, ks=self.psize * 16, pd=self.psize // 2 * 16, stride=16)

        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, py, px, C, H // py, W // px).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, C, H, W)
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, py, px, C // 2, H // py * 2, W // px * 2).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, C // 2, H * 2, W * 2)
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, py, px, C // 4, H // py * 4, W // px * 4).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, C // 4, H * 4, W * 4)
        warp_ref_patches_x8 = warp_ref_patches_x8.view(N, py, px, C // 8, H // py * 8, W // px * 8).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_ref_patches_x8 = warp_ref_patches_x8.view(N, C // 8, H * 8, W * 8)
        warp_ref_patches_x16 = warp_ref_patches_x16.view(N, py, px, C // 16, H // py * 16, W // px * 16).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_ref_patches_x16 = warp_ref_patches_x16.view(N, C // 16, H * 16, W * 16)

        # warped feature
        warp_ref_l = [warp_ref_patches_x16, warp_ref_patches_x8, warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]

        ##############################################

        x = self.intro(inp)

        encs = []

        _, chan, _, _ = x.shape

        for index, (encoder, down, masa_blk_enc) in enumerate(zip(self.encoders, self.downs, self.masa_blk_enc)):

            x = masa_blk_enc(torch.cat([x, warp_ref_l[index]], dim=1))[:, :chan, :, :]  # fused_feature * alpha + x

            x = encoder(x)
            encs.append(x)
            x = down(x)

            chan = chan * 2

        x = self.masa_blk_middle[0](torch.cat([x, warp_ref_l[-1]], dim=1))[:, :chan, :, :]

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):

            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :ori_H, :ori_W]


class NAFNetLocal_RefFusion(Local_Base, NAFNetRefFusion):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetRefFusion.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
