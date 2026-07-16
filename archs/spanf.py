"""Trainable SPAN-F architecture for efficient super-resolution.

The network topology follows the XiaomiMM SPANF submission to the NTIRE 2025
Efficient Super-Resolution challenge. Its submitted inference model contains
already re-parameterized 3x3 convolutions; this implementation retains SPAN's
multi-branch Conv3XC during training and fuses every branch for evaluation and
export.

Reference implementation:
https://github.com/Amazingren/NTIRE2025_ESR/blob/main/models/team24_SPANF.py
"""

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


class Conv3XC(nn.Module):
    """Training-time multi-branch convolution re-parameterized to one 3x3 conv."""

    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super().__init__()
        del gain2  # Kept in the signature for SPAN code compatibility.

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
    """SPAN-F block with attention only when input and output widths match."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super().__init__()
        del bias  # SPAN/SPAN-F use bias-enabled Conv3XC layers.

        mid_channels = in_channels if mid_channels is None else mid_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = nn.SiLU(inplace=True) if hasattr(nn, "SiLU") else _SiLU()

    def forward(self, x):
        out1 = self.c1_r(x)
        out2 = self.c2_r(self.act1(out1))
        out3 = self.c3_r(self.act1(out2))
        if self.in_channels != self.out_channels:
            return out3

        sim_att = torch.sigmoid(out3) - 0.5
        return (out3 + x) * sim_att


class SPANF(nn.Module):
    """Accelerated SPAN-F network adapted to configurable integer scale."""

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=32,
        upscale=2,
        bias=True,
        nearest_init=True,
    ):
        super().__init__()
        if int(upscale) < 1:
            raise ValueError(f"upscale must be positive, got {upscale}")
        if int(num_in_ch) < 1 or int(num_out_ch) < 1:
            raise ValueError("Input and output channels must be positive.")
        if int(feature_channels) < 1:
            raise ValueError("feature_channels must be positive.")

        self.upscale = int(upscale)
        self.num_in_ch = int(num_in_ch)
        self.num_out_ch = int(num_out_ch)
        self.feature_channels = int(feature_channels)

        scale_channels = self.num_in_ch * (self.upscale ** 2)
        self.conv_near = nn.Conv2d(
            self.num_in_ch,
            scale_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.num_in_ch,
            bias=False,
        )
        if nearest_init:
            self._initialize_nearest_shortcut()

        self.block_1 = SPAB(
            self.num_in_ch,
            self.feature_channels,
            self.feature_channels,
            bias=bias,
        )
        self.block_2 = SPAB(self.feature_channels, bias=bias)
        self.block_3 = SPAB(self.feature_channels, bias=bias)
        self.block_4 = SPAB(self.feature_channels, bias=bias)
        self.block_5 = SPAB(self.feature_channels, bias=bias)

        concat_channels = self.feature_channels * 2 + scale_channels
        self.conv_cat = conv_layer(concat_channels, self.feature_channels, 1, bias=True)
        self.conv_2 = Conv3XC(
            self.feature_channels,
            self.num_out_ch * (self.upscale ** 2),
            gain1=2,
            s=1,
        )
        self.depth_to_space = nn.PixelShuffle(self.upscale)

    def _initialize_nearest_shortcut(self):
        """Initialize the grouped convolution as depth-to-space nearest neighbor."""
        with torch.no_grad():
            self.conv_near.weight.zero_()
            self.conv_near.weight[:, 0, 1, 1] = 1.0

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        for module in self.modules():
            if isinstance(module, Conv3XC) and not module.deploy:
                module.eval_params_ready = False
        return result

    def _convert_conv_near_to_dense(self):
        """Replace the grouped shortcut with an exactly equivalent dense conv."""
        source = self.conv_near
        if source.groups == 1:
            return
        if source.in_channels % source.groups != 0:
            raise ValueError("conv_near input channels must be divisible by groups.")
        if source.out_channels % source.groups != 0:
            raise ValueError("conv_near output channels must be divisible by groups.")

        dense = nn.Conv2d(
            source.in_channels,
            source.out_channels,
            kernel_size=source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=1,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
        ).to(device=source.weight.device, dtype=source.weight.dtype)

        inputs_per_group = source.in_channels // source.groups
        outputs_per_group = source.out_channels // source.groups
        with torch.no_grad():
            dense.weight.zero_()
            for group_index in range(source.groups):
                input_start = group_index * inputs_per_group
                input_end = input_start + inputs_per_group
                output_start = group_index * outputs_per_group
                output_end = output_start + outputs_per_group
                dense.weight[output_start:output_end, input_start:input_end].copy_(
                    source.weight[output_start:output_end]
                )
            if source.bias is not None:
                dense.bias.copy_(source.bias)

        dense.weight.requires_grad = source.weight.requires_grad
        if dense.bias is not None:
            dense.bias.requires_grad = source.bias.requires_grad
        dense.train(source.training)
        self.conv_near = dense

    def switch_to_deploy(self):
        for module in self.modules():
            if isinstance(module, Conv3XC):
                module.switch_to_deploy()
        self._convert_conv_near_to_dense()
        self.eval()
        return self

    def forward(self, x):
        out_feature = self.conv_near(x)
        out_b1 = self.block_1(x)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)

        out = self.conv_cat(torch.cat([out_feature, out_b5, out_b1], dim=1))
        out = self.conv_2(out)
        return self.depth_to_space(out)


__all__ = ["Conv3XC", "SPAB", "SPANF"]
