import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def setup_logger(save_dir, name="SPANFx2"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(save_dir / "train.log", mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class AverageMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def resolve_auto_device(device):
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def get_lr(epoch, cycle_length=300, max_lr=1e-4, min_lr=1e-6):
    if cycle_length <= 0:
        return max_lr
    t_cur = epoch % cycle_length
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * t_cur / cycle_length)
    )


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_torch(path, map_location="cpu"):
    path = os.fspath(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict or contain a state dict.")

    for key in ("params_ema", "params", "model_state_dict", "state_dict", "net_g"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value, key

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "root"
    raise ValueError(
        "No state dict found. Supported keys: params_ema, params, "
        "model_state_dict, state_dict, net_g, or a raw state dict."
    )


def clean_state_dict(state_dict):
    prefixes = ("module.", "_orig_mod.", "net_g.", "generator.", "model.")
    cleaned = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def load_model_weights(model, weight_path, device="cpu", strict=True):
    checkpoint = load_torch(weight_path, map_location=device)
    state_dict, _ = extract_state_dict(checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=strict)
    return checkpoint


def rgb_to_y(image):
    if image.shape[1] != 3:
        raise ValueError("Y-channel PSNR requires a 3-channel RGB tensor.")
    coefficients = image.new_tensor((65.481, 128.553, 24.966)).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True) / 255.0 + 16.0 / 255.0


def calc_psnr(pred, target, crop_border=0, test_y_channel=False, eps=1e-12):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    if pred.shape != target.shape:
        raise ValueError(f"PSNR shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if crop_border > 0:
        if pred.shape[-2] <= 2 * crop_border or pred.shape[-1] <= 2 * crop_border:
            raise ValueError("crop_border is too large for the evaluated image.")
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= eps:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def forward_tiled(model, x, scale, tile_size=0, tile_pad=24):
    """Run tiled inference in LR coordinates and crop overlaps when stitching."""
    if tile_size is None or int(tile_size) <= 0:
        return model(x)

    tile_size = int(tile_size)
    tile_pad = max(int(tile_pad), 0)
    batch, _, height, width = x.shape
    output = None

    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        in_top = max(top - tile_pad, 0)
        in_bottom = min(bottom + tile_pad, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            in_left = max(left - tile_pad, 0)
            in_right = min(right + tile_pad, width)

            tile = x[..., in_top:in_bottom, in_left:in_right]
            tile_output = model(tile)
            if output is None:
                output = tile_output.new_zeros(
                    batch,
                    tile_output.shape[1],
                    height * scale,
                    width * scale,
                )

            crop_top = (top - in_top) * scale
            crop_left = (left - in_left) * scale
            crop_bottom = crop_top + (bottom - top) * scale
            crop_right = crop_left + (right - left) * scale
            output[..., top * scale:bottom * scale, left * scale:right * scale] = (
                tile_output[..., crop_top:crop_bottom, crop_left:crop_right]
            )
    return output


def tensor_to_uint8_image(tensor):
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only batch size 1 can be converted to a single image.")
        tensor = tensor[0]
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return np.round(array * 255.0).astype(np.uint8)


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)


def save_sr_comparison(input_tensor, output_tensor, path):
    """Save LR bilinear upsampling and model SR output side by side."""
    output_height, output_width = output_tensor.shape[-2:]
    bilinear = F.interpolate(
        input_tensor,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    bilinear_image = tensor_to_uint8_image(bilinear)
    output_image = tensor_to_uint8_image(output_tensor)
    if bilinear_image.shape != output_image.shape:
        raise ValueError(
            "Comparison shape mismatch after bilinear interpolation: "
            f"{bilinear_image.shape} vs {output_image.shape}."
        )

    header_height = 32
    gap = 8
    canvas = Image.new(
        "RGB",
        (output_width * 2 + gap, output_height + header_height),
        color=(24, 24, 24),
    )
    canvas.paste(Image.fromarray(bilinear_image), (0, header_height))
    canvas.paste(Image.fromarray(output_image), (output_width + gap, header_height))

    input_height, input_width = input_tensor.shape[-2:]
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 9),
        f"LR {input_width}x{input_height} (bilinear)",
        fill=(255, 255, 255),
    )
    draw.text(
        (output_width + gap + 8, 9),
        f"Model SR {output_width}x{output_height}",
        fill=(255, 255, 255),
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def list_image_files(root, recursive=True):
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Image path not found: {root}")
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {root}")
        return [root]
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
