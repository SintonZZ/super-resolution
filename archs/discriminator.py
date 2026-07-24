"""Training-only discriminator architectures."""

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral normalization from Real-ESRGAN."""

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = bool(skip_connection)
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1))
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    @staticmethod
    def _activate(value):
        return F.leaky_relu(value, negative_slope=0.2, inplace=True)

    def forward(self, image):
        x0 = self._activate(self.conv0(image))
        x1 = self._activate(self.conv1(x0))
        x2 = self._activate(self.conv2(x1))
        x3 = self._activate(self.conv3(x2))

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = self._activate(self.conv4(x3))
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self._activate(self.conv5(x4))
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = self._activate(self.conv6(x5))
        if self.skip_connection:
            x6 = x6 + x0

        output = self._activate(self.conv7(x6))
        output = self._activate(self.conv8(output))
        return self.conv9(output)
