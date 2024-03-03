import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


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

        # self.conv_L5 = nn.Conv2d(nf*2**3, nf*2**4, 3, 2, 1, bias=True)
        # self.blk_L5 = make_layer(functools.partial(ResidualBlock, nf=nf * 2 ** 4), n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))
        fea_L4 = self.blk_L4(self.act(self.conv_L4(fea_L3)))
        # fea_L5 = self.blk_L5(self.act(self.conv_L5(fea_L4)))

        # return [fea_L1, fea_L2, fea_L3, fea_L4, fea_L5]
        return [fea_L1, fea_L2, fea_L3, fea_L4]


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

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x


##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##  Sparse Transformer Block (STB)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerResFusionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerResFusionBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):

        shortcut = x

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x * self.alpha + shortcut


class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C * len(Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x) * w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))


class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0


class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y


Operations = [
    'sep_conv_1x1',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
    'avg_pool_3x3'
]

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'sep_conv_1x1': lambda C, stride, affine: SepConv(C, C, 1, stride, 0, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_conv_7x7': lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                               bias=False)
        self.conv2 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                               bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


## Mixture of Experts Feature Compensator (MEFC)
class subnet(nn.Module):
    def __init__(self, dim, layer_num=1, steps=4):
        super(subnet, self).__init__()

        self._C = dim
        self.num_ops = len(Operations)
        self._layer_num = layer_num
        self._steps = steps

        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C)
            self.layers += [layer]

    def forward(self, x):

        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(x)
                weights = F.softmax(weights, dim=-1)
            else:
                x = layer(x, weights)

        return x


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DRSformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):
        super(DRSformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level0 = subnet(dim)  ## We do not use MEFC for training Rain200L and SPA-Data

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = subnet(dim=int(dim * 2 ** 1))  ## We do not use MEFC for training Rain200L and SPA-Data

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_enc_level1)  ## We do not use MEFC for training Rain200L and SPA-Data
        out_enc_level1 = self.encoder_level1(inp_enc_level0)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)  ## We do not use MEFC for training Rain200L and SPA-Data

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class DRSformerRefFusion(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'

                 nf=64,
                 ext_n_blocks=[4, 4, 4, 4],
                 reffusion_n_blocks=[1, 1, 1, 1],
                 reffusion_n_blocks_middle=1,
                 scale=1,
                 num_nbr=1,
                 psize=3,
                 lr_block_size=8,
                 ref_down_block_size=1.5,
                 dilations=[1, 2, 3]
                 ):
        super(DRSformerRefFusion, self).__init__()

        ########################
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

        self.padder_size = 2 ** 3

        self.masa_enc = Encoder(in_chl=inp_channels, nf=nf, n_blks=ext_n_blocks)
        self.masa_blk_enc = nn.ModuleList()
        self.masa_blk_middle = nn.ModuleList()
        self.masa_blk_dec = nn.ModuleList()
        ##########################

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level0 = subnet(dim)  ## We do not use MEFC for training Rain200L and SPA-Data

        # 00 fuse
        self.masa_blk_enc_level1 = nn.Sequential(*[
            TransformerResFusionBlock(dim=2 * dim,
                                      num_heads=heads[0],
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type) for i in range(reffusion_n_blocks[0])
        ])

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        # 11 fuse
        self.masa_blk_enc_level2 = nn.Sequential(*[
            TransformerResFusionBlock(dim=2 * dim * 2 ** 1,
                                      num_heads=heads[1],
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type) for i in range(reffusion_n_blocks[1])
        ])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        # 22 fuse
        self.masa_blk_enc_level3 = nn.Sequential(*[
            TransformerResFusionBlock(dim=2 * dim * 2 ** 2,
                                      num_heads=heads[2],
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type) for i in range(reffusion_n_blocks[2])
        ])

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        # 33 fuse
        self.masa_blk_enc_level4 = nn.Sequential(*[
            TransformerResFusionBlock(dim=2 * dim * 2 ** 3,
                                      num_heads=heads[3],
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type) for i in range(reffusion_n_blocks[3])
        ])

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = subnet(dim=int(dim * 2 ** 1))  ## We do not use MEFC for training Rain200L and SPA-Data

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

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
            reflr_patches = F.unfold(reflr, kernel_size=(ks, ks), padding=dilation, stride=stride,
                                     dilation=dilation)  # [N, C*ks*ks, Hr*Wr]
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

    def forward(self, inp_img, ref_img):

        _, _, ori_H, ori_W = inp_img.shape
        inp_img = self.check_image_size(inp_img)
        ref_img = self.check_image_size(ref_img)

        ############################# MASA Search ################################

        _, _, h, w = inp_img.size()

        # start from the deepest feature, patches info on LR
        px = w // self.padder_size // self.lr_block_size
        py = h // self.padder_size // self.lr_block_size

        k_x = w // self.padder_size // px
        k_y = h // self.padder_size // py

        # print(px, py, k_x, k_y)  # px py 3, k_x k_y 8

        _, _, h, w = ref_img.size()

        # sxd, block info on Ref
        diameter_x = 2 * int(w // self.padder_size // (2 * px) * self.ref_down_block_size) + 1
        diameter_y = 2 * int(h // self.padder_size // (2 * py) * self.ref_down_block_size) + 1

        # print(diameter_x, diameter_y)  # diameter_x, diameter_y 13

        # extract multi-scale feature from both LR and Ref
        feat_lq = self.masa_enc(inp_img)
        feat_ref = self.masa_enc(ref_img)

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
        # ind_y_x16, ind_x_x16 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 16)

        ind_b = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device),
                                        py * px * (diameter_y + 2) * (diameter_x + 2))
        ind_b_x2 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device),
                                           py * px * ((diameter_y + 2) * 2) * ((diameter_x + 2) * 2))
        ind_b_x4 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device),
                                           py * px * ((diameter_y + 2) * 4) * ((diameter_x + 2) * 4))
        ind_b_x8 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device),
                                           py * px * ((diameter_y + 2) * 8) * (diameter_x + 2) * 8)
        # ind_b_x16 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device),
        #                                     py * px * ((diameter_y + 2) * 16) * (diameter_y + 2) * 16)

        # block on ref
        ref_patches = feat_ref[4][ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2,
                                                                     C).permute(0, 3, 1,
                                                                                2).contiguous()  # [N*py*px, C, (radius_y+1)*2, (radius_x+1)*2]

        ref_patches_x1 = feat_ref[4][ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2,
                                                                        C).permute(0, 3, 1, 2).contiguous()
        ref_patches_x2 = feat_ref[3][ind_b_x2, :, ind_y_x2, ind_x_x2].view(N * py * px, (diameter_y + 2) * 2,
                                                                           (diameter_x + 2) * 2, C // 2).permute(0, 3,
                                                                                                                 1,
                                                                                                                 2).contiguous()
        ref_patches_x4 = feat_ref[2][ind_b_x4, :, ind_y_x4, ind_x_x4].view(N * py * px, (diameter_y + 2) * 4,
                                                                           (diameter_x + 2) * 4, C // 4).permute(0, 3,
                                                                                                                 1,
                                                                                                                 2).contiguous()
        ref_patches_x8 = feat_ref[1][ind_b_x8, :, ind_y_x8, ind_x_x8].view(N * py * px, (diameter_y + 2) * 8,
                                                                           (diameter_x + 2) * 8, C // 8).permute(0, 3,
                                                                                                                 1,
                                                                                                                 2).contiguous()
        # ref_patches_x16 = feat_ref[0][ind_b_x16, :, ind_y_x16, ind_x_x16].view(N * py * px, (diameter_y + 2) * 16,
        #                                                                        (diameter_x + 2) * 16, C // 16).permute(
        #     0, 3, 1, 2).contiguous()

        # patches on LR
        lr_patches = lr_patches.contiguous().view(N * py * px, C, k_y + 2, k_x + 2)
        # calculate similarity between LR patches within Ref blocks
        corr_all_l, index_all_l = self.search_org(lr_patches, ref_patches, ks=self.psize, pd=self.psize // 2, stride=1)

        index_all = index_all_l[:, :, :, 0]
        soft_att_all = corr_all_l[:, :, :, 0:1].permute(0, 3, 1, 2)

        # block -> patches -> transfer
        warp_ref_patches_x1 = self.transfer(ref_patches_x1, index_all, soft_att_all, ks=self.psize, pd=self.psize // 2,
                                            stride=1)
        warp_ref_patches_x2 = self.transfer(ref_patches_x2, index_all, soft_att_all, ks=self.psize * 2,
                                            pd=self.psize // 2 * 2, stride=2)
        warp_ref_patches_x4 = self.transfer(ref_patches_x4, index_all, soft_att_all, ks=self.psize * 4,
                                            pd=self.psize // 2 * 4, stride=4)
        warp_ref_patches_x8 = self.transfer(ref_patches_x8, index_all, soft_att_all, ks=self.psize * 8,
                                            pd=self.psize // 2 * 8, stride=8)
        # warp_ref_patches_x16 = self.transfer(ref_patches_x16, index_all, soft_att_all, ks=self.psize * 16,
        #                                      pd=self.psize // 2 * 16, stride=16)

        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, py, px, C, H // py, W // px).permute(0, 3, 1, 4, 2,
                                                                                               5).contiguous()
        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, C, H, W)
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, py, px, C // 2, H // py * 2, W // px * 2).permute(0, 3, 1, 4,
                                                                                                            2,
                                                                                                            5).contiguous()
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, C // 2, H * 2, W * 2)
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, py, px, C // 4, H // py * 4, W // px * 4).permute(0, 3, 1, 4,
                                                                                                            2,
                                                                                                            5).contiguous()
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, C // 4, H * 4, W * 4)
        warp_ref_patches_x8 = warp_ref_patches_x8.view(N, py, px, C // 8, H // py * 8, W // px * 8).permute(0, 3, 1, 4,
                                                                                                            2,
                                                                                                            5).contiguous()
        warp_ref_patches_x8 = warp_ref_patches_x8.view(N, C // 8, H * 8, W * 8)
        # warp_ref_patches_x16 = warp_ref_patches_x16.view(N, py, px, C // 16, H // py * 16, W // px * 16).permute(0, 3,
        #                                                                                                          1, 4,
        #                                                                                                          2,
        #                                                                                                          5).contiguous()
        # warp_ref_patches_x16 = warp_ref_patches_x16.view(N, C // 16, H * 16, W * 16)

        # warped feature
        warp_ref_l = [warp_ref_patches_x8, warp_ref_patches_x4, warp_ref_patches_x2,
                      warp_ref_patches_x1]

        ##############################################

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_enc_level1)  ## We do not use MEFC for training Rain200L and SPA-Data

        # 00 fuse
        feat_ref_fuse_in = torch.cat([inp_enc_level0, warp_ref_l[0]], dim=1)
        _, embed_dim, _, _ = feat_ref_fuse_in.shape
        inp_enc_level0 = self.masa_blk_enc_level1(feat_ref_fuse_in)[:, :embed_dim // 2, :, :]

        out_enc_level1 = self.encoder_level1(inp_enc_level0)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        # 11 fuse
        feat_ref_fuse_in = torch.cat([inp_enc_level2, warp_ref_l[1]], dim=1)
        _, embed_dim, _, _ = feat_ref_fuse_in.shape
        inp_enc_level2 = self.masa_blk_enc_level2(feat_ref_fuse_in)[:, :embed_dim // 2, :, :]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        # 22 fuse
        feat_ref_fuse_in = torch.cat([inp_enc_level3, warp_ref_l[2]], dim=1)
        _, embed_dim, _, _ = feat_ref_fuse_in.shape
        inp_enc_level3 = self.masa_blk_enc_level3(feat_ref_fuse_in)[:, :embed_dim // 2, :, :]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        # 33 fuse
        feat_ref_fuse_in = torch.cat([inp_enc_level4, warp_ref_l[3]], dim=1)
        _, embed_dim, _, _ = feat_ref_fuse_in.shape
        inp_enc_level4 = self.masa_blk_enc_level4(feat_ref_fuse_in)[:, :embed_dim // 2, :, :]
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)  ## We do not use MEFC for training Rain200L and SPA-Data

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1[:, :, :ori_H, :ori_W]

