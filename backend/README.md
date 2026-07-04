# 后端服务说明

简谱图片识别服务 - 上传图片 → YOLOv8 检测 → Transformer 拼装 → Score JSON。

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
node scripts/generate_training_pngs.cjs       # 渲染 PNG + YOLO 标注
python backend/scripts/prepare_pairs.py       # 构造 Transformer 监督对
```

### 3. 训练模型
```bash
python backend/scripts/train_detector.py       # 训练 YOLOv8s
python backend/scripts/train_transformer.py   # 训练 Transformer
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
```

返回结构:
```json
{
  "score": { ... Score JSON ... },
  "detections": [...],
  "inference_ms": 234
}
```

## 目录结构
- `main.py` - FastAPI 入口
- `detector.py` - YOLOv8 推理封装
- `assembler.py` - 检测框 → token → JSON
- `model/transformer.py` - 轻量 Encoder-Decoder
- `model/tokenizer.py` - token ↔ Score 字段映射
- `scripts/` - 训练脚本
- `weights/` - 模型权重
