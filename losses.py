"""Training losses used by the two-stage perceptual/GAN pipeline."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


VGG19_LAYER_NAMES = (
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3",
    "relu3_3", "conv3_4", "relu3_4", "pool3",
    "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3",
    "relu4_3", "conv4_4", "relu4_4", "pool4",
    "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3",
    "relu5_3", "conv5_4", "relu5_4", "pool5",
)


def build_pixel_loss(loss_type):
    """Build an unweighted pixel or feature-space loss."""
    loss_type = str(loss_type).lower()
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type in ("mse", "l2"):
        return nn.MSELoss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss type: {loss_type}")


class VGG19FeatureExtractor(nn.Module):
    """Frozen ImageNet VGG19 feature extractor with BasicSR layer names."""

    def __init__(self, layer_names, use_input_norm=True, range_norm=False):
        super().__init__()
        layer_names = tuple(layer_names)
        unknown = sorted(set(layer_names) - set(VGG19_LAYER_NAMES))
        if unknown:
            raise ValueError(f"Unsupported VGG19 feature layer(s): {unknown}")
        if not layer_names:
            raise ValueError("At least one VGG19 feature layer is required.")

        try:
            from torchvision.models import vgg19
            try:
                from torchvision.models import VGG19_Weights
                network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                network = vgg19(pretrained=True)
        except ImportError as error:
            raise ImportError(
                "torchvision is required when perceptual loss is enabled."
            ) from error

        max_index = max(VGG19_LAYER_NAMES.index(name) for name in layer_names)
        self.features = nn.Sequential(*list(network.features.children())[:max_index + 1])
        self.layer_names = layer_names
        self.use_input_norm = bool(use_input_norm)
        self.range_norm = bool(range_norm)
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        """Keep the pretrained loss network in evaluation mode."""
        del mode
        return super().train(False)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "VGG19 perceptual loss expects an NCHW RGB tensor."
            )
        if self.range_norm:
            image = (image + 1.0) / 2.0
        if self.use_input_norm:
            image = (image - self.mean) / self.std

        outputs = OrderedDict()
        wanted = set(self.layer_names)
        for index, layer in enumerate(self.features):
            image = layer(image)
            name = VGG19_LAYER_NAMES[index]
            if name in wanted:
                outputs[name] = image
        return outputs


class PerceptualLoss(nn.Module):
    """Multi-layer VGG19 content loss and optional Gram-matrix style loss."""

    def __init__(
        self,
        layer_weights,
        criterion="l1",
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0.0,
        feature_extractor=None,
    ):
        super().__init__()
        if not isinstance(layer_weights, dict) or not layer_weights:
            raise ValueError("perceptual.layer_weights must be a non-empty object.")
        self.layer_weights = {
            str(name): float(weight) for name, weight in layer_weights.items()
        }
        self.criterion = build_pixel_loss(criterion)
        self.perceptual_weight = float(perceptual_weight)
        self.style_weight = float(style_weight)
        self.vgg = feature_extractor or VGG19FeatureExtractor(
            self.layer_weights,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

    @staticmethod
    def gram_matrix(feature):
        batch, channels, height, width = feature.shape
        flattened = feature.reshape(batch, channels, height * width)
        return flattened.bmm(flattened.transpose(1, 2)) / (
            channels * height * width
        )

    def forward(self, prediction, target):
        prediction_features = self.vgg(prediction)
        with torch.no_grad():
            target_features = self.vgg(target.detach())

        content = prediction.new_zeros(())
        style = prediction.new_zeros(())
        for name, weight in self.layer_weights.items():
            content = content + weight * self.criterion(
                prediction_features[name], target_features[name]
            )
            if self.style_weight > 0:
                style = style + weight * self.criterion(
                    self.gram_matrix(prediction_features[name]),
                    self.gram_matrix(target_features[name]),
                )
        return content * self.perceptual_weight, style * self.style_weight


class GANLoss(nn.Module):
    """Vanilla GAN loss operating on discriminator logits."""

    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.real_label = float(real_label)
        self.fake_label = float(fake_label)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, target_is_real):
        value = self.real_label if target_is_real else self.fake_label
        target = logits.new_full(logits.shape, value)
        return self.criterion(logits, target)


class USMSharp(nn.Module):
    """Unsharp-mask sharpening compatible with Real-ESRGAN supervision."""

    def __init__(self, radius=51, sigma=0.0, weight=0.5, threshold=10.0):
        super().__init__()
        radius = int(radius)
        if radius < 3:
            raise ValueError("USM radius must be at least 3.")
        if radius % 2 == 0:
            radius += 1
        sigma = float(sigma)
        if sigma <= 0:
            sigma = 0.3 * ((radius - 1) * 0.5 - 1) + 0.8
        coordinates = torch.arange(radius, dtype=torch.float32) - radius // 2
        kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel[:, None] * kernel[None, :]
        self.register_buffer("kernel", kernel.view(1, 1, radius, radius))
        self.weight = float(weight)
        self.threshold = float(threshold)

    def _filter(self, image):
        channels = image.shape[1]
        padding = self.kernel.shape[-1] // 2
        mode = "reflect" if min(image.shape[-2:]) > padding else "replicate"
        padded = F.pad(image, (padding, padding, padding, padding), mode=mode)
        kernel = self.kernel.to(dtype=image.dtype).expand(channels, 1, -1, -1)
        return F.conv2d(padded, kernel, groups=channels)

    def forward(self, image):
        blurred = self._filter(image)
        residual = image - blurred
        mask = (residual.abs() * 255.0 > self.threshold).to(image.dtype)
        soft_mask = self._filter(mask)
        sharpened = (image + self.weight * residual).clamp(0.0, 1.0)
        return (
            soft_mask * sharpened + (1.0 - soft_mask) * image
        ).clamp(0.0, 1.0)
