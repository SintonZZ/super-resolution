"""SPAN architecture adapted from the authors' Apache-2.0 implementation.

Official implementation:
https://github.com/hongyuanyu/SPAN/blob/main/basicsr/archs/span_arch.py

Module names intentionally match the official implementation so that released
x4 checkpoints can initialize the x2 model. Only ``upsampler.0`` changes shape
when the scale changes from 4 to 2.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class _SiLU(nn.Module):
    """Compatibility fallback for old local PyTorch; PyTorch 2.5 uses nn.SiLU."""

    def forward(self, x):
        return x * torch.sigmoid(x)


def _make_pair(value):
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        bias=bias,
    )


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace=inplace)
    if act_type == "lrelu":
        return nn.LeakyReLU(negative_slope=neg_slope, inplace=inplace)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f"Activation layer is not implemented: {act_type}")


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("OrderedDict is not supported by sequential().")
        return args[0]

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    return sequential(
        conv_layer(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size,
        ),
        nn.PixelShuffle(upscale_factor),
    )


class Conv3XC(nn.Module):
    """Training-time multi-branch convolution re-parameterized to one 3x3 conv."""

    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super().__init__()
        del gain2  # Kept in the signature for checkpoint/code compatibility.

        self.weight_concat = None
        self.bias_concat = None
        self.stride = s
        self.has_relu = relu
        self.deploy = False
        self.eval_params_ready = False

        gain = gain1
        self.sk = nn.Conv2d(c_in, c_out, kernel_size=1, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, kernel_size=1, bias=bias),
            nn.Conv2d(
                c_in * gain,
                c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(c_out * gain, c_out, kernel_size=1, bias=bias),
        )
        self.eval_conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        self.eval_conv.weight.requires_grad = False
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        if any(layer.bias is None for layer in self.conv) or self.sk.bias is None:
            raise ValueError("Conv3XC re-parameterization currently requires bias=True.")

        with torch.no_grad():
            w1 = self.conv[0].weight.detach().clone()
            b1 = self.conv[0].bias.detach().clone()
            w2 = self.conv[1].weight.detach().clone()
            b2 = self.conv[1].bias.detach().clone()
            w3 = self.conv[2].weight.detach().clone()
            b3 = self.conv[2].bias.detach().clone()

            w = F.conv2d(
                w1.flip(2, 3).permute(1, 0, 2, 3),
                w2,
                padding=2,
            ).flip(2, 3).permute(1, 0, 2, 3)
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

            weight_concat = F.conv2d(
                w.flip(2, 3).permute(1, 0, 2, 3),
                w3,
            ).flip(2, 3).permute(1, 0, 2, 3)
            bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

            skip_weight = F.pad(self.sk.weight.detach(), (1, 1, 1, 1))
            weight_concat = weight_concat + skip_weight
            bias_concat = bias_concat + self.sk.bias.detach()

            self.weight_concat = weight_concat
            self.bias_concat = bias_concat
            self.eval_conv.weight.copy_(weight_concat)
            self.eval_conv.bias.copy_(bias_concat)
            self.eval_params_ready = True

    def switch_to_deploy(self):
        self.update_params()
        self.deploy = True
        return self

    def forward(self, x):
        if self.training and not self.deploy:
            self.eval_params_ready = False
            x_pad = F.pad(x, (1, 1, 1, 1), mode="constant", value=0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            if not self.deploy and not self.eval_params_ready:
                self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super().__init__()
        del bias  # The official SPAB uses bias-enabled Conv3XC layers.

        mid_channels = in_channels if mid_channels is None else mid_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = nn.SiLU(inplace=True) if hasattr(nn, "SiLU") else _SiLU()
        self.act2 = activation("lrelu", neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        return out, out1, sim_att


class SPAN(nn.Module):
    """Swift Parameter-free Attention Network for efficient super-resolution."""

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=48,
        upscale=2,
        bias=True,
        img_range=255.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super().__init__()
        if upscale < 1:
            raise ValueError(f"upscale must be positive, got {upscale}")
        if len(rgb_mean) != num_in_ch:
            raise ValueError("rgb_mean length must equal num_in_ch.")

        self.upscale = int(upscale)
        self.img_range = float(img_range)
        # Keep this as a plain tensor to preserve official state_dict keys.
        self.mean = torch.tensor(rgb_mean).view(1, num_in_ch, 1, 1)

        self.conv_1 = Conv3XC(num_in_ch, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)
        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, 1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)
        self.upsampler = pixelshuffle_block(
            feature_channels,
            num_out_ch,
            upscale_factor=self.upscale,
        )

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        # Branch weights may have changed, so the derived eval kernels are stale.
        for module in self.modules():
            if isinstance(module, Conv3XC) and not module.deploy:
                module.eval_params_ready = False
        return result

    def switch_to_deploy(self):
        for module in self.modules():
            if isinstance(module, Conv3XC):
                module.switch_to_deploy()
        self.eval()
        return self

    def forward(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        x = (x - mean) * self.img_range

        out_feature = self.conv_1(x)
        out_b1, _, _ = self.block_1(out_feature)
        out_b2, _, _ = self.block_2(out_b1)
        out_b3, _, _ = self.block_3(out_b2)
        out_b4, _, _ = self.block_4(out_b3)
        out_b5, _, _ = self.block_5(out_b4)
        out_b6, out_b5_2, _ = self.block_6(out_b5)
        out_b6 = self.conv_2(out_b6)

        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], dim=1))
        return self.upsampler(out)
