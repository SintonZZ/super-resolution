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

当前实现固定用于 ×2 训练。默认先以 L1 训练保真模型，再自动进入 VGG19 感知损失和
GAN 微调阶段；`--resume` 只接受新版两阶段 `train.py` 生成的 SPAN-F ×2 checkpoint。

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
Pillow bicubic；将 `dataset.degradation.type` 设为 `"high_order"` 后，会使用二阶高阶退化：
第一轮依次执行随机 Gaussian/anisotropic blur、随机上采样/下采样/保持尺寸、Gaussian/Poisson
噪声和 JPEG；第二轮再次执行同类操作，最后通过可选的 windowed-sinc 低通滤波精确调整到
目标 ×2 LR。最终 JPEG 与“精确缩放+sinc”的顺序也会随机交换，以模拟多次编辑和转码。

验证集使用相同参数分布，但会根据 `validation_seed` 和图片相对路径固定每张图的完整随机序列，
使不同 step 的 PSNR 可以直接比较；训练集仍会在每次读取时重新随机退化。旧的 `"realistic"`
类型名称保留为 `"high_order"` 的兼容别名；旧版平铺的单阶段参数会自动映射为第一轮参数，
未配置的第二轮和最终处理使用默认值。

如果没有单独验证集，也可以把 `val_hr_dir` 设为 `null`，代码会按 `val_ratio` 从训练 HR 中
固定划分验证集。若使用真实退化，推荐的初始配置如下：

```json
{
  "degradation": {
    "type": "high_order",
    "validation_seed": 1234,
    "first_order": {
      "blur_probability": 0.8,
      "kernel_size": 15,
      "isotropic_probability": 0.5,
      "sigma_range": [0.2, 2.0],
      "rotation_range": [-3.1416, 3.1416],
      "resize_scale_range": [0.5, 1.5],
      "resize_direction_probabilities": [0.7, 0.2, 0.1],
      "resize_modes": ["bicubic", "bilinear", "lanczos"],
      "resize_mode_probabilities": [0.5, 0.25, 0.25],
      "noise_probability": 0.8,
      "gaussian_noise_probability": 0.6,
      "gray_noise_probability": 0.2,
      "gaussian_sigma_range": [1.0, 10.0],
      "poisson_peak_range": [100.0, 1000.0],
      "jpeg_probability": 0.8,
      "jpeg_quality_range": [60, 95],
      "jpeg_subsampling": 2
    },
    "second_order": {
      "blur_probability": 0.4,
      "kernel_size": 15,
      "isotropic_probability": 0.7,
      "sigma_range": [0.2, 1.2],
      "rotation_range": [-3.1416, 3.1416],
      "resize_scale_range": [0.7, 1.2],
      "resize_direction_probabilities": [0.4, 0.3, 0.3],
      "resize_modes": ["bicubic", "bilinear", "lanczos"],
      "resize_mode_probabilities": [0.5, 0.25, 0.25],
      "noise_probability": 0.8,
      "gaussian_noise_probability": 0.6,
      "gray_noise_probability": 0.2,
      "gaussian_sigma_range": [1.0, 8.0],
      "poisson_peak_range": [100.0, 1000.0],
      "jpeg_probability": 1.0,
      "jpeg_quality_range": [60, 95],
      "jpeg_subsampling": 2
    },
    "final": {
      "sinc_probability": 0.8,
      "sinc_kernel_size": 15,
      "sinc_cutoff_range": [0.3333, 1.0],
      "jpeg_before_resize_probability": 0.5,
      "resize_modes": ["bicubic", "bilinear", "lanczos"],
      "resize_mode_probabilities": [0.5, 0.25, 0.25]
    }
  }
}
```

`resize_direction_probabilities` 的顺序是 `[缩小, 放大, 保持]`；第一轮缩放相对 HR 当前尺寸，
第二轮缩放相对目标 LR 尺寸，最后一定会精确对齐到 `HR / scale`。`resize_mode_probabilities`
与同一段里的 `resize_modes` 一一对应。

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

## OV50Q 低照 RAW 数据集制作

`prepare_raw2dnr_dataset.py` 可以把无噪声 OV50Q RAW 制作为与真实部署链一致的
全图 LR/HR 配对。每张 RAW 会先使用 `dataset.py` 相同的亮度重采样模拟低照度，再分别通过
参数化仿真噪声和真实黑帧噪声生成两个版本。噪声图经过 Raw2DNR 和简化 ISP 后降采样为
`1600×1348` LR；未经过 DNR 的低照 Clean RAW 使用同一组白平衡和曝光参数生成
`3200×2696` HR。

先检查输入、黑帧和预期样本数量，不执行推理或写文件：

```bash
.venv/bin/python prepare_raw2dnr_dataset.py --dry-run
```

建议先在 CUDA 上完成一张图的烟测：

```bash
.venv/bin/python prepare_raw2dnr_dataset.py \
  --max-images 1 \
  --output-root /mnt/d/ov50q_sr_dataset_smoke \
  --device cuda
```

完整生成支持断点恢复：

```bash
.venv/bin/python prepare_raw2dnr_dataset.py \
  --output-root /mnt/d/ov50q_sr_dataset \
  --device cuda \
  --resume
```

默认按采集日期隔离数据：`20260617/20260622` 用于训练、`20260624` 用于验证、
`20260701` 用于测试。输入根目录默认为 `/mnt/d/ov50q_real_raw`，黑帧根目录默认为
`/mnt/d/ov50q/dark/processed`。完整参数可通过 `--help` 查看。生成过程会记录异常 RAW，
并把每个样本的噪声方法、gain/K、黑帧来源、亮度比例和 ISP 参数写入 `manifest.jsonl`。

训练时将数据集配置改为：

```json
{
  "manifest_path": "/mnt/d/ov50q_sr_dataset/manifest.jsonl",
  "scale": 2,
  "lr_patch_size": 256,
  "augment": false,
  "max_images": null
}
```

配置 `manifest_path` 后不再需要 `train_hr_dir`、`train_lr_dir` 等目录字段，也不会应用
额外的 `high_order` 合成退化。manifest 中的相对路径按 manifest 文件所在目录解析。

## 训练

先复制一份本机配置并填写数据路径：

```bash
cp config/train.json config/train.local.json
```

从头训练：

```bash
.venv/bin/python train.py --config config/train.local.json
```

训练过程按 optimizer step 计数并自动执行两个阶段：

- Stage 1 使用 USM GT 和 L1，默认 250,000 steps、学习率 `2e-4`。
- Stage 2 从 Stage 1 最终 EMA 权重开始，使用 L1、VGG19 感知损失和
  `UNetDiscriminatorSN`，默认 100,000 steps，G/D 学习率均为 `1e-4`。

两阶段都使用 Adam `(0.9, 0.99)`、无 warmup，并把 MultiStep milestone 放在阶段终点。
训练维护 `ema_decay=0.999` 的 Generator EMA，验证和推理/导出默认使用 EMA 权重。

也可以从命令行覆盖常用参数：

```bash
.venv/bin/python train.py \
  --config config/train.json \
  --train-hr-dir /path/to/train/HR \
  --train-lr-dir /path/to/train/LR \
  --val-hr-dir /path/to/val/HR \
  --val-lr-dir /path/to/val/LR \
  --stage1-steps 250000 \
  --stage2-steps 100000
```

输出目录包含根级 `config.json` 和 `latest_model.pth`，以及 `stage1/`、`stage2/` 两个子目录。
各阶段保存 latest、周期 checkpoint、`best_psnr_model.pth`；启用 LPIPS 验证时 Stage 2 还会保存
`best_lpips_model.pth`。恢复训练使用 `--resume /path/to/latest_model.pth`，checkpoint 中的
`active_stage` 会决定恢复哪个阶段，并恢复对应模型、EMA、optimizer、step 和 AMP scaler。

跳过 Stage 1、从已有 L1 checkpoint 直接运行 Stage 2 时，在配置中关闭 Stage 1 并设置：

```json
{
  "training": {
    "stage1": {"enabled": false},
    "stage2": {
      "enabled": true,
      "pretrained_generator": "checkpoints/run_xxx/best_model.pth"
    }
  }
}
```

也可通过 `--stage2-pretrained` 覆盖路径。该模式优先加载 `params_ema`，Stage 2 的
Generator EMA、判别器、optimizer、step 和最佳指标均从头创建。旧版单阶段 checkpoint
可以作为 `pretrained_generator`，但不能用于新版整体 `--resume`。

仓库提供了针对 OV50Q 真实退化数据的微调配置，可直接启动：

```bash
.venv/bin/python train.py --config config/finetune.json
```

运行前确认 `dataset.manifest_path` 和 `training.stage2.pretrained_generator` 与本机实际路径
一致。该配置跳过 Stage 1，以 `128×128` LR patch、batch size 4 运行 100,000-step GAN 微调；
如显存充足可单独提高 Stage 2 batch size。

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

模型预测的 SR 图像直接保存在 `output-dir`，对应的“LR 双线性放大 vs 模型 SR”对比图保存在
`output-dir/comparisons`。

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
