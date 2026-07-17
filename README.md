# SPAN-F ×2 超分辨率训练

这个项目用于从头训练 SPAN-F ×2 单图超分辨率模型。网络拓扑基于 XiaomiMM 在
NTIRE 2025 Efficient Super-Resolution Challenge 提交的
[SPANF](https://github.com/Amazingren/NTIRE2025_ESR/blob/main/models/team24_SPANF.py)，
并保留 SPAN 的训练态 Conv3XC 多分支重参数化能力。

## 模型结构

相较原始 CH48 SPAN，SPAN-F 默认采用：

- 32 个特征通道和 5 个 SPAB；
- 一个 nearest-neighbor-like 分组卷积输入捷径；
- `shortcut / block_5 / block_1` 三路特征聚合；
- 直接生成 `3 × scale²` 通道并通过 PixelShuffle 放大；
- 推理和 ONNX 导出前，将每个 Conv3XC 融合成一个 3×3 卷积；
- 部署时将输入捷径无损转换成 `groups=1` 的普通卷积，导出图不包含分组卷积。

当前实现固定用于 ×2 训练。它不加载原始 SPAN ×4 或 SPAN-F ×4 权重；`--resume` 只接受
本项目 `train.py` 生成的 SPAN-F ×2 checkpoint。

## 环境

目标训练环境为 PyTorch 2.5.1。推荐在项目目录创建 uv 虚拟环境：

```bash
uv venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

如果训练服务器已经在系统 Python 中安装了 PyTorch 2.5.1，可以复用系统包：

```bash
uv venv --python 3.10 --system-site-packages
uv pip install --python .venv/bin/python numpy pillow tqdm onnx
```

## 数据格式

支持成对 LR/HR 数据和只有 HR 的数据。

### 成对 LR/HR

```text
dataset/train/HR/0001.png
dataset/train/LR/0001x2.png
dataset/val/HR/0801.png
dataset/val/LR/0801x2.png
```

对应配置：

```json
{
  "train_hr_dir": "dataset/train/HR",
  "train_lr_dir": "dataset/train/LR",
  "val_hr_dir": "dataset/val/HR",
  "val_lr_dir": "dataset/val/LR",
  "filename_template": "{}x2"
}
```

如果 LR/HR 文件名完全相同，使用默认的 `"filename_template": "{}"`。

### 只有 HR

把 `train_lr_dir` 和 `val_lr_dir` 设为 `null`，dataset 会从 HR 在线生成 ×2 LR。默认使用
Pillow bicubic；将 `dataset.degradation.type` 设为 `"realistic"` 后，训练集会依次应用随机
Gaussian/anisotropic blur、bilinear/bicubic/Lanczos 下采样、Gaussian/Poisson 噪声和 JPEG 压缩。
各退化步骤的概率和强度范围均可在 `dataset.degradation` 中配置。验证集使用完全相同的退化
参数分布，但会根据 `validation_seed` 和图片相对路径固定每张图的随机参数，使不同 epoch 的
PSNR 可直接比较；训练集仍会在每次读取时重新随机退化。

如果没有单独验证集，也可以把 `val_hr_dir` 设为 `null`，代码会按 `val_ratio` 从训练 HR 中
固定划分验证集。若使用真实退化，推荐的初始配置如下：

```json
{
  "degradation": {
    "type": "realistic",
    "validation_seed": 1234,
    "blur_probability": 0.8,
    "kernel_size": 15,
    "isotropic_probability": 0.5,
    "sigma_range": [0.2, 2.0],
    "rotation_range": [-3.1416, 3.1416],
    "resize_modes": ["bicubic", "bilinear", "lanczos"],
    "resize_probabilities": [0.5, 0.25, 0.25],
    "noise_probability": 0.8,
    "gaussian_noise_probability": 0.6,
    "gray_noise_probability": 0.2,
    "gaussian_sigma_range": [1.0, 10.0],
    "poisson_peak_range": [100.0, 1000.0],
    "jpeg_probability": 0.8,
    "jpeg_quality_range": [60, 95],
    "jpeg_subsampling": 2
  }
}
```

`gaussian_sigma_range` 的单位是 8-bit 像素值，`poisson_peak_range` 越小表示噪声越强，
`jpeg_quality_range` 越小表示压缩越强；`jpeg_subsampling=2` 表示常见的 4:2:0 色度抽样。
`validation_seed` 只控制固定验证退化，不会让训练退化失去随机性。将 `type` 改回
`"bicubic"` 即可让训练集和验证集都恢复原始 bicubic 行为。

可以直接运行 `dataset.py` 预览指定样本的 LR/HR 对比。脚本会把 LR 放大到 HR 尺寸后并排
保存，`--seed` 可用于观察同一 HR 的不同随机退化结果：

```bash
.venv/bin/python dataset.py \
  --config config/train.local.json \
  --split train \
  --index 0 \
  --seed 1234 \
  --output results/dataset_preview/sample_0000.png
```

增加 `--show` 可在带桌面环境的机器上同时打开图片；`--resize-mode nearest` 更容易观察 LR
的原始像素和压缩伪影，默认的 `bicubic` 更接近常规放大预览。

所有图片按 RGB、`float32 [0, 1]` 读取。成对数据必须严格满足
`HR 高宽 = LR 高宽 × 2`；训练时在 LR 空间随机裁剪，并对 HR 做对应裁剪。

## 训练

先复制一份本机配置并填写数据路径：

```bash
cp config/train.json config/train.local.json
```

从头训练：

```bash
.venv/bin/python train.py --config config/train.local.json
```

也可以从命令行覆盖常用参数：

```bash
.venv/bin/python train.py \
  --config config/train.json \
  --train-hr-dir /path/to/train/HR \
  --train-lr-dir /path/to/train/LR \
  --val-hr-dir /path/to/val/HR \
  --val-lr-dir /path/to/val/LR
```

输出目录包含 `config.json`、`train.log`、`latest_model.pth`、`best_model.pth` 和周期 checkpoint。
恢复中断训练使用 `--resume /path/to/latest_model.pth`，它会严格恢复模型、optimizer、epoch 和
AMP scaler。

## 测试与推理

成对测试并计算 RGB/Y 通道 PSNR：

```bash
.venv/bin/python test.py \
  --weights checkpoints/run_xxx/best_model.pth \
  --test-hr-dir /path/to/test/HR \
  --test-lr-dir /path/to/test/LR
```

测试结果中的 `comparisons` 目录会保存两栏对比图：左侧为 LR 双线性放大结果，右侧为模型预测的
SR 图像；真实 HR 仅用于计算 PSNR。

对任意图片或目录做 ×2 推理：

```bash
.venv/bin/python inference.py \
  --weights checkpoints/run_xxx/best_model.pth \
  --input /path/to/lr_images \
  --output-dir results/inference
```

显存不足时可增加 `--tile-size 256 --tile-pad 24`。`tile-size` 单位是 LR 像素。

## ONNX 导出

导出前会自动完成 Conv3XC 重参数化：

```bash
.venv/bin/python export_onnx.py \
  --weights checkpoints/run_xxx/best_model.pth \
  --output checkpoints/run_xxx/spanf_x2.onnx
```

默认导出动态 NCHW；增加 `--static-shape --input-height 256 --input-width 256` 可导出固定尺寸。

如果尚未训练、只想提前验证 NPU 编译兼容性和推理效率，可以不传 `--weights`。此时脚本会从
`config/train.json` 的 `model` 配置构建随机初始化模型，模型结构与训练模型相同，但输出图像
没有质量意义：

```bash
.venv/bin/python export_onnx.py \
  --output checkpoints/spanf_x2_random_256.onnx \
  --static-shape --input-height 256 --input-width 256
```

可以通过 `--config /path/to/train.json` 指定其他模型配置，通过 `--seed` 固定随机参数。

### ONNX Runtime PTQ

NPU 性能测试建议先导出固定输入尺寸的 FP32 模型：

```bash
.venv/bin/python export_onnx.py \
  --output checkpoints/spanf_x2_random_256.onnx \
  --static-shape --input-height 256 --input-width 256
```

准备 32–128 张具有代表性的 LR RGB 图片作为校准集，然后执行静态 PTQ：

如果校准数据是一张大图，可以先按模型输入尺寸切成互不重叠的 PNG 小图：

```bash
.venv/bin/python quantization/split_calibration_image.py \
  --input-image /path/to/large_lr.png \
  --output-dir calib_data/spanf_x2_256 \
  --tile-height 256 \
  --tile-width 256 \
  --max-tiles 128
```

切片按从左到右、从上到下的顺序生成，步长等于切片尺寸；不足一个完整切片的右侧和底部区域
会被丢弃，因此所有输出都严格为指定尺寸且互不重叠。输出目录已有同名文件时默认拒绝覆盖，
确认覆盖可增加 `--force`。

然后执行静态 PTQ：

```bash
.venv/bin/python quantization/onnx_ptq.py \
  --model-input checkpoints/spanf_x2_random_256.onnx \
  --model-output checkpoints/spanf_x2_random_256_ptq.onnx \
  --calib-dir /path/to/calibration/LR \
  --max-samples 128 \
  --force
```

脚本默认使用 MinMax 校准、QDQ 格式、激活和权重均为 uint8 非对称逐 tensor 量化，并且只量化
Conv。校准目录也支持 `[1,3,H,W]` 或 `[3,H,W]`、数值范围为 `[0,1]` 的 `.npy` 文件。
动态图可通过 `--calib-height` 和 `--calib-width` 指定校准尺寸，但 NPU 性能测试通常应使用与
实际部署一致的固定尺寸模型。量化前会拒绝包含分组卷积或 SiLU/Swish 节点的 ONNX，量化完成后
会运行 `onnx.checker`。

## 本地验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖 SPAN-F ×2 拓扑、输出尺寸、nearest shortcut、Conv3XC/整网重参数化等价性、
tiled inference 拼接和数据集配对。
