# 后端服务说明

简谱图片识别服务 - 上传图片 → 重叠切片 YOLO 检测 → 二维符号关系拼装 → Score JSON。

识别不是普通 OCR：高低音点、附点、升降号和减时线必须挂到对应的
音符数字上。默认拼装器保留检测框二维几何关系；旧的 Transformer 序列
纠错仍可通过 API 参数实验性开启，但不再默认启用。

## 快速开始

### 1. 安装依赖
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备训练数据
```bash
# 在项目根目录
python scripts/gen_training_data.py           # 生成 2009 份训练 JSON
npx tsx scripts/generate_training_pngs.cjs    # 渲染 PNG + YOLO 标注
python backend/scripts/split_dataset.py       # 已生成图片时可单独重建 80/20 划分
python backend/scripts/build_scan_domain_dataset.py \
  --pages 0 --negatives 0 --row-pages 500 --lyric-rows 100  # 扫描谱行+中文文本
python backend/scripts/prepare_pairs.py       # 构造 Transformer 监督对
```

### 3. 训练模型
```bash
python backend/scripts/train_detector.py       # M3/M 系列默认使用 MPS + 960px
python backend/scripts/train_transformer.py   # 训练 Transformer

# HOMR/TrOMR 思路的行图片→多分支事件模型（M3 Pro 使用 MPS）
python backend/scripts/build_jianpu_transformer_dataset.py
python backend/scripts/build_synthetic_transformer_dataset.py
python backend/scripts/train_jianpu_vision_transformer.py \
  --data backend/jianpu_synthetic_transformer_dataset --device mps \
  --epochs 15 --batch 16 --d-model 96 --layers 2 --max-width 1024 \
  --output backend/weights/jianpu_vision_transformer_synthetic.pt
python backend/scripts/train_jianpu_vision_transformer.py --device mps \
  --epochs 30 --batch 8 --lr 0.0001 --d-model 96 --layers 2 --max-width 1024 \
  --init backend/weights/jianpu_vision_transformer_synthetic.pt \
  --output backend/weights/jianpu_vision_transformer_v2.pt
```

设备冒烟测试可限制训练比例和验证张数，并用 `--no-promote` 防止覆盖生产权重：

```bash
python backend/scripts/train_detector.py --device mps --epochs 1 \
  --fraction 0.01 --val-limit 8 --no-promote --name mps_smoke
```

完整链路评估（不只是检测框 mAP）：

```bash
python backend/scripts/evaluate_recognizer.py \
  --weights backend/weights/best.pt \
  --manifest public/training/val.txt

# 不经过检测器，测量二维拼装和标注的理论上限
python backend/scripts/evaluate_recognizer.py \
  --manifest public/training/val.txt --ground-truth-boxes
```

该脚本输出音高、完整音符、完整小节的编辑相似度和整页完全匹配率。真实测试集也使用
相同的图片清单格式，并为每张图片提供同名 Score JSON。

没有人工 Score JSON 的真实图片只能运行健康诊断，不能计算准确率：

```bash
python backend/scripts/analyze_real_images.py /Volumes/jianpu/images \
  --output backend/eval_outputs/nas_real_diagnostics_adaptive.csv \
  --device mps --imgsz 1280
```

训练完成后 `weights/` 下会有 `best.pt` 和 `transformer.pt`。

### 4. 启动服务
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 调用
```bash
curl -X POST http://localhost:8000/recognize -F "file=@test.png"

# 实验性视觉序列模式；当前仍不作为默认生产路径
curl -X POST 'http://localhost:8000/recognize?visual_sequence=true' \
  -F "file=@test.png"
```

返回结构:
```json
{
  "score": { ... Score JSON ... },
  "detections": [...],
  "inference_ms": 234
}
```

大于 1600px 的图片会自动进行 18% 重叠切片，再以 class-aware NMS 合并，
避免整页缩放导致小圆点和细减时线消失。检出异常稀疏的整页会自动以较低
阈值重试，并只在谱行结构得到明显改善时采用重试结果。连续多组五线谱线的
输入会被判定为非简谱并返回空结果。实际照片建议保留原分辨率上传。

## 数据与评估

视觉序列模型采用二维 CNN 编码、横纵位置编码、CTC 单调骨架头和多分支
Transformer 解码器。当前分支包含事件类型、音高、升降、八度、时值和演奏法；
银标没有可靠给出的分支会被 loss mask，而不是错误标成 `NONE`。HOMR 仓库仅作为
架构研究参考，没有复制或链接其 AGPL 源码。

现有实测（页面级隔离）：合成验证集 CTC 音高相似度 0.9786；真实 80 行验证集
调参后为 0.8147；跨来源 101 行测试集为 0.5932。它证明了图像到序列链路，
但尚未超过当前生产检测器，因此 API 默认仍走二维检测与几何拼装。下一轮应增加
至少 1,000 条人工复核的真实谱行，并用真实字体渲染/扫描退化做预训练。

`generate_training_pngs.cjs` 会生成稳定、互不重叠的 `train.txt` 和
`val.txt`（约 80/20）。不要把同一目录同时用作 train/val；此前的指标
存在数据泄漏，不能代表真实照片精度。

合成图只能学习符号形状。要提升真实图片效果，另建 50-100 张真实图片
的 `test` 集（永不参与训练），并逐步标注至少 300-500 张来自目标场景的
手机照片/扫描件用于微调。优先覆盖不同字体、纸张底色、透视、阴影、
压缩、折痕及低清晰度，而不是继续生成同一种渲染字体。

## 目录结构
- `main.py` - FastAPI 入口
- `detector.py` - YOLOv8 推理封装
- `assembler.py` - 检测框 → token → JSON
- `model/transformer.py` - 轻量 Encoder-Decoder
- `model/jianpu_vision_transformer.py` - 图像到多分支事件 Transformer + CTC
- `visual_recognizer.py` - 页面谱行定位与视觉序列 API 适配
- `model/tokenizer.py` - token ↔ Score 字段映射
- `scripts/` - 训练脚本
- `weights/` - 模型权重
