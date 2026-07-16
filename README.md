# SPAN ×2 超分辨率微调

这个项目用于把官方 SPAN ×4 模型迁移到 ×2 单图超分辨率任务。
模型实现基于论文作者的 [官方 SPAN 代码](https://github.com/hongyuanyu/SPAN)，层名与官方
`span_arch.py` 保持一致，以便直接读取官方 checkpoint。

## ×4 权重如何迁移到 ×2

SPAN 的放大倍率只影响最后的 PixelShuffle head：

```text
x4: upsampler.0 输出通道 = 3 * 4^2 = 48
x2: upsampler.0 输出通道 = 3 * 2^2 = 12
```

因此 `upsampler.0.weight` 和 `upsampler.0.bias` 无法直接复用。训练脚本会：

1. 从官方常见的 `params_ema`、`params` 或 raw state dict 中读取权重；
2. 加载所有名字和 shape 均匹配的 SPAN 主干参数；
3. 保留随机初始化的 ×2 `upsampler.0`；
4. 严格检查是否还有 head 以外的主干缺失或 shape 不匹配。

默认给新 head 使用主干 5 倍学习率，可在 `training.head_lr_multiplier` 中修改。

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

支持两种训练数据。

### 成对 LR/HR

```text
dataset/train/HR/0001.png
dataset/train/LR/0001x2.png
dataset/val/HR/0801.png
dataset/val/LR/0801x2.png
```

配置为：

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

把 `train_lr_dir` 和 `val_lr_dir` 设为 `null`，dataset 会使用 Pillow bicubic 在线生成 ×2 LR。
如果没有单独验证集，也可以把 `val_hr_dir` 设为 `null`，代码会按 `val_ratio` 从训练 HR 中
固定划分验证集。

所有图片按 RGB、`float32 [0, 1]` 读取。成对数据必须严格满足
`HR 高宽 = LR 高宽 × 2`；训练时在 LR 空间随机裁剪，并对 HR 做对应裁剪。

## 训练

先复制一份本机配置：

```bash
cp config/train.json config/train.local.json
```

仓库默认使用已经下载到 `pretrained/spanx4_ch48.pth` 的官方 ×4 CH48 权重；如需使用其他
checkpoint，可在 `config/train.local.json` 中修改：

```json
"pretrained_4x_path": "/path/to/official_span_x4.pth"
```

启动训练：

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
  --val-lr-dir /path/to/val/LR \
  --pretrained-4x /path/to/span_x4.pth
```

输出目录包含 `config.json`、`train.log`、`latest_model.pth`、`best_model.pth` 和周期 checkpoint。
恢复中断训练应使用 `--resume`，它会严格恢复 ×2 模型、optimizer、epoch 和 AMP scaler；不要把
官方 ×4 权重传给 `--resume`。

## 测试与推理

成对测试并计算 RGB/Y 通道 PSNR：

```bash
.venv/bin/python test.py \
  --weights checkpoints/run_xxx/best_model.pth \
  --test-hr-dir /path/to/test/HR \
  --test-lr-dir /path/to/test/LR
```

对任意图片或目录做 ×2 推理：

```bash
.venv/bin/python inference.py \
  --weights checkpoints/run_xxx/best_model.pth \
  --input /path/to/lr_images \
  --output-dir results/inference
```

显存不足时可增加 `--tile-size 256 --tile-pad 24`。`tile-size` 单位是 LR 像素。

## ONNX 导出

导出前会把每个训练态 `Conv3XC` 重参数化成单个 3×3 卷积：

```bash
.venv/bin/python export_onnx.py \
  --weights checkpoints/run_xxx/best_model.pth \
  --output checkpoints/run_xxx/span_x2.onnx
```

默认导出动态 NCHW；增加 `--static-shape --input-height 256 --input-width 256` 可导出固定尺寸。

## 本地验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖 ×2 输出尺寸、Conv3XC 重参数化等价性、×4 主干权重迁移和 tiled inference 拼接。
