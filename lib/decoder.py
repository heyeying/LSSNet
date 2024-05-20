import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange
import torch.nn.functional as F


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


class ConvNormAct(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, skip=False,
                 inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SGR_module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SGR_module, self).__init__()

        self.align = ConvNormAct(dim_in=in_channel, dim_out=out_channel, kernel_size=1)
        self.conv3x3 = ConvNormAct(dim_in=out_channel//2, dim_out=out_channel//2, kernel_size=3)
        # self.DWConv3x3 = ConvNormAct(dim_in=in_channel//2, dim_out=in_channel//2, groups=in_channel//2, kernel_size=3)
        self.se = SELayer(channel=out_channel//2, reduction=4)
        self.conv1x1 = ConvNormAct(dim_in=out_channel, dim_out=out_channel, kernel_size=1)

    def forward(self, x):
        x = self.align(x)
        c = x.shape[1]
        x_s, x_c = torch.split(x, c // 2, dim=1)    # x_s:space, x_c:channel
        x_s = self.conv3x3(x_s)
        x_c = self.se(x_c)
        xs_xc = x_s * x_c
        x_s = x_s + xs_xc
        x_c = x_c + xs_xc
        out = torch.cat((x_s, x_c), dim=1)
        out = self.conv1x1(out)

        return out


class MFE_module(nn.Module):
    def __init__(self, in_channel, out_channel, exp_ratio=1.0):
        super(MFE_module, self).__init__()

        mid_channel = in_channel * exp_ratio

        self.DWConv = ConvNormAct(mid_channel, mid_channel, kernel_size=3, groups=out_channel // 2)
        self.DWConv3x3 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=3, groups=in_channel // 4)
        self.DWConv5x5 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=5, groups=in_channel // 4)
        self.DWConv7x7 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=7, groups=in_channel // 4)
        self.PWConv1 = ConvNormAct(in_channel, mid_channel, kernel_size=1)
        self.PWConv2 = ConvNormAct(mid_channel, out_channel, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channel)
        # MaxPool2d Capture high-frequency information
        self.Maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        channels = x.size(1)
        channels_per_part = channels // 4
        x1 = x[:, :channels_per_part, :, :]
        x2 = x[:, channels_per_part:2*channels_per_part, :, :]
        x3 = x[:, 2*channels_per_part:3*channels_per_part, :, :]
        x4 = x[:, 3*channels_per_part:, :, :]
        x1 = self.Maxpool(x1)
        x2 = self.DWConv3x3(x2)
        x3 = self.DWConv5x5(x3)
        x4 = self.DWConv7x7(x4)

        x2 = x1 * x2
        x3 = x2 * x3
        x4 = x3 * x4
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.PWConv1(x)
        x = x + self.DWConv(x)
        x = self.PWConv2(x)
        x = x + shortcut

        return x


class IAF_module(nn.Module):

    def __init__(self, dim_in, dim_out, exp_ratio=1.0,
                 dw_ks=3, dilation=1, num_head=8, window_size=7,
                 qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,):
        super().__init__()
        self.norm = LayerNorm2d(dim_in)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = dim_in == dim_out

        assert dim_in % num_head == 0, 'dim should be divisible by num_heads'
        self.num_head = num_head
        self.window_size = window_size
        self.dim_head = dim_in // self.num_head
        self.scale = self.dim_head ** -0.5
        self.k = ConvNormAct(dim_in, dim_in, kernel_size=1, bias=qkv_bias)
        self.q = ConvNormAct(dim_in, dim_in, kernel_size=1, bias=qkv_bias)
        self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # MaxPool2d 捕获高频信息
        self.Maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.proj2 = nn.Conv2d(dim_in, dim_mid, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

        num_head = self.num_head
        self.dwc_v = ConvNormAct(num_head, num_head, kernel_size=3, stride=1, dilation=1, groups=num_head)  # 用于增加秩
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=1, dilation=dilation,
                                      groups=dim_mid)

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x1, x2):   # x1=low, x2=high

        x1 = self.norm(x1)  # (1,128,44,44)
        x2 = self.norm(x2)  # (1,128,44,44)
        B, C, H, W = x1.shape   # B,C=128,H=44,W=44
        window_size_W, window_size_H = self.window_size, self.window_size   # 11, 11
        pad_l, pad_t = 0, 0
        pad_r = (window_size_W - W % window_size_W) % window_size_W
        pad_b = (window_size_H - H % window_size_H) % window_size_H
        x1 = F.pad(x1, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
        x2 = F.pad(x2, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
        n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W  # n1,n2 window_num
        x1 = rearrange(x1, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1,
                       n2=n2).contiguous()  # 划分窗口(16,128,11,11)
        x2 = rearrange(x2, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1,
                       n2=n2).contiguous()  # 划分窗口(16,128,11,11)

        b, c, h, w = x1.shape
        q = self.q(x1)
        k = self.k(x2)
        q = rearrange(q, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head,
                      dim_head=self.dim_head).contiguous()
        k = rearrange(k, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head,
                      dim_head=self.dim_head).contiguous()

        attn_spa = (q @ k.transpose(-2, -1)) * self.scale
        attn_spa = attn_spa.softmax(dim=-1)
        attn_spa = self.attn_drop(attn_spa)

        x2 = rearrange(x2, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
        x_spa = attn_spa @ x2

        x_v = x2
        x_dwc = self.dwc_v(x_v)
        x_spa = x_spa + x_dwc
        x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                          w=w).contiguous()
        x_spa = self.v(x_spa)

        # unpadding
        x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        x = x + self.conv_local(x) if self.has_skip else self.conv_local(x)
        x = self.proj_drop(x)
        x = self.proj(x)

        return x


class SFS_module(nn.Module):     # The shallow feature supplementation (SFS) module
    def __init__(self, dim_in, exp_ratio):
        super(SFS_module, self).__init__()

        self.res = IRB_block(dim_in, exp_ratio)

    def forward(self, x, origin_x, pred):   # x:features of this layer, origin_x: shallow features, pred: prediction map
        pred = F.interpolate(input=pred, scale_factor=2.0, mode='bilinear', align_corners=False)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        att = origin_x * att + x
        out = self.res(att)

        return out


class IRB_block(nn.Module):
    def __init__(self, dim_in, exp_ratio):
        super(IRB_block, self).__init__()
        self.PWConv1 = ConvNormAct(dim_in=dim_in, dim_out=dim_in * exp_ratio, kernel_size=1)
        self.DWConv = ConvNormAct(dim_in=dim_in * exp_ratio, dim_out=dim_in * exp_ratio, kernel_size=3)
        self.PWConv2 = ConvNormAct(dim_in=dim_in * exp_ratio, dim_out=dim_in, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.PWConv1(x)
        x = self.DWConv(x)
        x = self.PWConv2(x)
        x = x + shortcut

        return x


class LSSNet(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], stages=4):
        super(LSSNet, self).__init__()

        self.stages = stages
        self.Conv_1x1 = nn.ModuleList([
            nn.Conv2d(2 * channels[i], channels[i], kernel_size=1, stride=1, padding=0) for i in range(1, self.stages)
        ])
        self.LocalBlock_x = nn.Sequential(
            ConvNormAct(dim_in=3, dim_out=channels[3], kernel_size=3, stride=2),
            ConvNormAct(dim_in=channels[3], dim_out=channels[3], kernel_size=3, stride=2),
            MFE_module(in_channel=channels[3], out_channel=channels[3], exp_ratio=2),
            MFE_module(in_channel=channels[3], out_channel=channels[3], exp_ratio=2),
        )
        self.LocalBlocks = nn.ModuleList([
            MFE_module(in_channel=channels[i], out_channel=channels[i], exp_ratio=2) for i in range(self.stages)
        ])
        self.MixBlock0 = IAF_module(dim_in=channels[0], dim_out=channels[0], window_size=11,
                                    exp_ratio=2)  # MixBlock0：512
        self.MixBlock1 = IAF_module(dim_in=channels[1], dim_out=channels[1], window_size=11,
                                    exp_ratio=2)    # MixBlock1：192
        self.MixBlock2 = IAF_module(dim_in=channels[2], dim_out=channels[2], window_size=11,
                                    exp_ratio=2)      # MixBlock2：88
        self.MixBlock3 = IAF_module(dim_in=channels[3], dim_out=channels[3], window_size=11,
                                    exp_ratio=2)  # MixBlock2：88
        self.HardParts = nn.ModuleList(
            SFS_module(dim_in=channels[i], exp_ratio=2) for i in range(1, self.stages)
        )
        self.Ups = nn.ModuleList([
            up_conv(ch_in=channels[i], ch_out=channels[i+1]) for i in range(self.stages-1)
        ])
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # SELayer
        self.Semantic_fusion1 = SGR_module(in_channel=channels[3], out_channel=channels[3])
        self.Semantic_fusion2 = SGR_module(in_channel=channels[3], out_channel=channels[2])
        self.Semantic_fusion3 = SGR_module(in_channel=channels[3] + channels[2], out_channel=channels[1])
        self.Semantic_fusion4 = SGR_module(in_channel=channels[3] + channels[2] + channels[1], out_channel=channels[0])

        self.origin_down1 = nn.Sequential(
            ConvNormAct(dim_in=64, dim_out=64, kernel_size=3, stride=2),
            ConvNormAct(dim_in=64, dim_out=128, kernel_size=1)
        )
        self.origin_down2 = nn.Sequential(
            ConvNormAct(dim_in=128, dim_out=128, kernel_size=3, stride=2),
            ConvNormAct(dim_in=128, dim_out=320, kernel_size=1)
        )

        # Prediction heads initialization
        n_class = 1
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x, x4, x3, x2, x1):
        origin_x1 = self.LocalBlock_x(x)   # x(1,3,352,352) -> origin_x1(1,64,88,88)
        origin_x2 = self.origin_down1(origin_x1)    # origin_x2(1,128,44,44)
        origin_x3 = self.origin_down2(origin_x2)    # origin_x3(1,320,22,22)

        origin_x1_ = self.Semantic_fusion1(origin_x1)
        x1 = self.MixBlock3(origin_x1_, x1)
        x1 = self.LocalBlocks[3](x1)

        x1_down = self.down(x1)
        x1_ = self.Semantic_fusion2(x1_down)
        x2 = self.MixBlock2(x1_, x2)
        x2 = self.LocalBlocks[2](x2)

        x1_down = self.down(x1_down)
        x2_down = self.down(x2)
        x1_x2 = torch.cat((x1_down, x2_down), dim=1)
        x1_x2 = self.Semantic_fusion3(x1_x2)
        x3 = self.MixBlock1(x1_x2, x3)
        x3 = self.LocalBlocks[1](x3)

        x1_down = self.down(x1_down)
        x2_down = self.down(x2_down)
        x3_down = self.down(x3)
        x1_x2_x3 = torch.cat((x1_down, x2_down, x3_down), dim=1)
        x1_x2_x3 = self.Semantic_fusion4(x1_x2_x3)
        x4 = self.MixBlock0(x1_x2_x3, x4)
        x4 = self.LocalBlocks[0](x4)
        pred4 = self.out_head1(x4)

        u4 = self.Ups[0](x4)
        d3 = torch.mul(x3, u4) + x3
        d3 = self.HardParts[0](d3, origin_x3, pred4)
        pred3 = self.out_head2(d3)

        u3 = self.Ups[1](d3)
        d2 = torch.mul(x2, u3) + x2
        d2 = self.HardParts[1](d2, origin_x2, pred3)
        pred2 = self.out_head3(d2)

        u2 = self.Ups[2](d2)
        d1 = torch.mul(x1, u2) + x1
        d1 = self.HardParts[2](d1, origin_x1, pred2)
        pred1 = self.out_head4(d1)

        return pred4, pred3, pred2, pred1


