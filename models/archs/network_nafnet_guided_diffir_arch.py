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
from models.archs.nafnet_local_arch import Local_Base, Local_Base_DynamicFusion


from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


class Mapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
    ):
        super(Mapper, self).__init__()

        self.num_words = num_words

        for i in range(self.num_words):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                              nn.LayerNorm(1280),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1280, 1280),
                                                              nn.LayerNorm(1280),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1280, 1280),
                                                              nn.LayerNorm(1280),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1280, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        embs = embs[0]

        for i in range(self.num_words):
            hidden_state = getattr(self, f'mapping_{i}')(embs[:, :1]) + getattr(self, f'mapping_patch_{i}')(embs[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states


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


class RefFusionResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(RefFusionResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf + nf, nf + nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf + nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, ref):

        out = self.conv2(self.act(self.conv1(torch.cat([x, ref], dim=1))))

        return out * self.alpha


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
        fea_L4 = self.blk_L3(self.act(self.conv_L3(fea_L3)))
        fea_L5 = self.blk_L3(self.act(self.conv_L3(fea_L4)))

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


#######################

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleGate_DynamicFusion(nn.Module):
    def __init__(self, dim):
        super(SimpleGate_DynamicFusion, self).__init__()

        # like DiffIR
        self.kernel = nn.Sequential(
            nn.Linear(10 * 1024, dim * 2, bias=False)
        )

    def forward(self, x, k_v):

        b, c, h, w = x.shape

        # print(x.shape)  # 8 64 128 128

        # print(k_v.shape)  # 8 20 1024

        k_v = self.kernel(torch.flatten(k_v, start_dim=1)).view(-1, c*2, 1, 1)

        # print(k_v.shape)  # 4 128 1 1

        k_v1, k_v2 = k_v.chunk(2, dim=1)

        # print(k_v1.shape)
        # print(k_v2.shape)

        x = k_v1 * x + k_v2
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


class NAFBlock_DynamicFusion(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super(NAFBlock_DynamicFusion, self).__init__()

        # dynamic fusion
        self.kernel = nn.Sequential(
            nn.Linear(10 * 1024, c*2, bias=False)
        )


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
        self.sg1 = SimpleGate_DynamicFusion(dim=c * 2)
        self.sg2 = SimpleGate_DynamicFusion(dim=c * 2)

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

    def forward(self, inp, k_v):

        x = inp

        b, c, h, w = inp.shape

        # print(k_v.shape)  # 8, 20, 1024

        # torch.flatten(k_v, dim=1)

        k_v_attention = self.kernel(torch.flatten(k_v, start_dim=1)).view(-1, c*2, 1, 1)

        # print(k_v.shape)  # 160, 64, 1, 1       ->   8, 64, 1, 1

        k_v1, k_v2 = k_v_attention.chunk(2, dim=1)

        # print(k_v1.shape)  # 160, 32, 1, 1
        # print(k_v2.shape)  # 160, 32, 1, 1
        # print(x.shape)  # 8, 32, 128, 128

        x = x * k_v1 + k_v2

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x, k_v)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg2(x, k_v)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class DynamicBasicLayer(nn.Module):
    def __init__(self, chan, num):
        super(DynamicBasicLayer, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(num):
            self.layers.append(NAFBlock_DynamicFusion(chan))

    def forward(self, x, k_v):

        for layer in self.layers:
            x = layer(x, k_v)

        return x


class NAFNetDynamicFusion(nn.Module):

    def __init__(self,
                 img_channel=3,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[]):
        super(NAFNetDynamicFusion, self).__init__()

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

                DynamicBasicLayer(chan, num)

                # nn.Sequential(
                #     *[NAFBlock_DynamicFusion(chan) for _ in range(num)]
                # )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock_DynamicFusion(chan) for _ in range(middle_blk_num)]
        #     )

        self.middle_blks = DynamicBasicLayer(chan, middle_blk_num)

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                # nn.Sequential(
                #     *[NAFBlock_DynamicFusion(chan) for _ in range(num)]
                # )

                DynamicBasicLayer(chan, num)
            )

        self.padder_size = 2 ** len(self.encoders)

        #
        # self.mapper = Mapper(input_dim=1280, output_dim=1024, num_words=20)
        # self.clip_image_encoder = CLIPVisionModel.from_pretrained("/hdd/linjingbo/CLIP-ViT-H-14-laionb_s32b_b79k")

    # def init_mapper_weight(self, path):
    #
    #     self.mapper.load_state_dict(torch.load(path))

    def forward(self, inp, k_v):

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, k_v)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x, k_v)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, k_v)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


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


class NAFNetLocalDynamic(Local_Base_DynamicFusion, NAFNetDynamicFusion):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetDynamicFusion.__init__(self, *args, **kwargs)

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


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)