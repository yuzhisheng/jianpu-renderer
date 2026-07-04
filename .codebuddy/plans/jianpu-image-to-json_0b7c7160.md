---
name: jianpu-image-to-json
overview: 基于现有 2009 张训练素材，训练 YOLOv8 符号检测模型 + 轻量 Transformer 序列模型，将图片简谱识别为 JSON，并在现有网站加入「图片识别」Tab 展示原图 vs 渲染结果对比。
design:
  architecture:
    framework: react
    component: shadcn
  styleKeywords:
    - 工具型UI
    - 暗色优先
    - 信息密度高
    - 拖拽上传
    - 并排对比
    - 实时反馈
  fontSystem:
    fontFamily: PingFang-SC
    heading:
      size: 20px
      weight: 600
    subheading:
      size: 14px
      weight: 500
    body:
      size: 13px
      weight: 400
  colorSystem:
    primary:
      - "#3B82F6"
      - "#2563EB"
      - "#1D4ED8"
    background:
      - "#1E1E1E"
      - "#252526"
      - "#F5F5F5"
      - "#FFFFFF"
    text:
      - "#FFFFFF"
      - "#D1D5DB"
      - "#6B7280"
      - "#111827"
    functional:
      - "#10B981"
      - "#EF4444"
      - "#F59E0B"
      - "#8B5CF6"
todos:
  - id: prepare-dataset
    content: Use [subagent:code-explorer] to map layout fields to token ids, then run training data generation (PNG + YOLO txt + pairs.npz)
    status: completed
  - id: train-yolo
    content: Train YOLOv8 detector on data.yaml and export best.pt to backend/weights/
    status: completed
    dependencies:
      - prepare-dataset
  - id: train-transformer
    content: Implement token vocabulary and train lightweight Encoder-Decoder Transformer, export transformer.pt
    status: completed
    dependencies:
      - prepare-dataset
  - id: backend-api
    content: Build FastAPI service with /recognize endpoint that loads both models and returns Score JSON
    status: completed
    dependencies:
      - train-yolo
      - train-transformer
  - id: frontend-recognize
    content: Build RecognizeView, ImageUploader, CompareView components and wire up /recognize call with layout toggle, download, copy
    status: completed
    dependencies:
      - backend-api
  - id: integrate-verify
    content: Add navigation entry from App.tsx and run end-to-end test on sample images to verify recognition quality
    status: completed
    dependencies:
      - frontend-recognize
---

## 产品概述

为简谱渲染器项目增加 **图片简谱识别** 功能：用户上传一张简谱图片，系统通过 YOLOv8 检测符号 + 轻量 Transformer 拼装结构化 token 序列，将结果转换为与项目渲染引擎兼容的 JSON 格式，并在网页上与原图对比展示，便于用户校验识别结果。

## 核心功能

- **图片上传**：网页端选择/拖拽简谱图片，提交到 Python 后端服务
- **YOLOv8 符号检测**：识别数字音符(1-7)、休止符、增时线、减时线、附点、八度点、升降号、小节线、各种竹笛技巧符号、力度记号、连音/圆滑线等的位置和类别
- **Transformer 序列拼装**：将检测结果按行/列分组、按时序编码为结构化 token 序列，预测并组装为合法 JSON
- **结果对比页**：原图与识别渲染结果左右/上下切换对比，可单独下载或拷贝识别出的 JSON
- **渲染回放**：识别出的 JSON 通过现有 `engine` 渲染为图片，验证是否与原图一致

## 技术栈

- **后端**：Python 3.10+ / FastAPI / Uvicorn / Ultralytics YOLOv8 / PyTorch（轻量 Transformer）/ Pillow / OpenCV
- **前端**：现有 React 18 + TypeScript + Vite + Tailwind，新增图像上传、对比页、JSON 拷贝/下载
- **训练**：`scripts/gen_training_data.py` 已能产出 2009 份 JSON，`scripts/generate_training_pngs.cjs` 已能产出 PNG + YOLO 标注 + `data.yaml`，直接复用
- **模型格式**：YOLOv8 导出 `best.pt`（推理用）；Transformer 导出 `transformer.pt`（推理用）

## 架构设计

### 整体流程

```
用户上传图片
    ↓ (multipart/form-data)
FastAPI /recognize 端点
    ↓
YOLOv8 检测 → 列表[(class, x, y, w, h, conf)]
    ↓
几何预处理 → 按 y 分行 → 每行按 x 排序 → 编码为 token 序列
    ↓
Transformer 解码 → 结构化 token 流
    ↓
token 流 → JSON 解析器 → Score JSON
    ↓
返回 JSON 给前端
    ↓
前端用现有 engine 渲染 + 左右/上下对比展示
```

### 组件关系

- `backend/main.py` — FastAPI 入口，挂载 `/recognize` 端点
- `backend/detector.py` — 包装 YOLOv8 模型，加载 `best.pt`
- `backend/assembler.py` — 检测结果 → token 序列 → JSON（用训练好的 Transformer）
- `backend/model/transformer.py` — 轻量 Transformer (Encoder-Decoder)
- `backend/scripts/train_detector.py` — YOLOv8 训练脚本（基于 `data.yaml`）
- `backend/scripts/train_transformer.py` — Transformer 训练脚本
- `backend/scripts/prepare_pairs.py` — 从 `public/training/` 构建 YOLO txt + token 序列训练对
- `src/components/RecognizeView.tsx` — 上传 + 切换对比布局 + 渲染结果组件
- `src/components/ImageUploader.tsx` — 拖拽/选择图片 + 提交后端
- `src/components/CompareView.tsx` — 切换左右/上下 + 下载/拷贝 JSON
- `src/api/recognize.ts` — 调用 `/recognize` 的 fetch 封装

### 关键决策与权衡

- **检测 + 序列 双模型**：YOLO 提供精确符号位置（后续也能反查），Transformer 学习时序结构（避免纯规则的连音/圆滑线/反复跳跃歧义）。相比纯规则更鲁棒，比端到端 OCR 更可控。
- **训练对复用现有素材**：`gen_training_data.py` 生成的 JSON 既能渲染 PNG，又能从 layout 反推 token 序列（事实标签），零成本准备监督数据。
- **后端独立部署**：避免浏览器加载大模型；YOLOv8s + 小 Transformer 在 CPU 上单图推理 < 1s 可接受。
- **JSON 渲染走现有 `engine`**：保证识别 → 渲染闭环完全自洽，便于对比验证。

## 实现细节

### 关键目录结构

```
jianpu-renderer/
├── backend/                                   [NEW] Python 后端服务
│   ├── main.py                                FastAPI 入口，/recognize 端点
│   ├── detector.py                            YOLOv8 推理封装
│   ├── assembler.py                           检测框 → token → JSON
│   ├── model/
│   │   ├── transformer.py                     轻量 Encoder-Decoder
│   │   └── tokenizer.py                       token ↔ Score 字段映射
│   ├── scripts/
│   │   ├── prepare_pairs.py                   从 training JSON + 标注生成监督对
│   │   ├── train_detector.py                  YOLOv8 训练
│   │   └── train_transformer.py               Transformer 训练
│   ├── weights/                               best.pt / transformer.pt
│   └── requirements.txt
├── src/
│   ├── api/
│   │   └── recognize.ts                       [NEW] 后端调用封装
│   └── components/
│       ├── RecognizeView.tsx                  [NEW] 识别主页面
│       ├── ImageUploader.tsx                  [NEW] 上传组件
│       └── CompareView.tsx                    [NEW] 对比组件
├── public/training/                           复用（已含 2009 份 JSON）
└── scripts/generate_training_pngs.cjs         复用（生成 PNG+YOLO 标注）
```

### 训练数据准备关键流程

1. 运行 `python scripts/gen_training_data.py` → 已有 2009 份 JSON
2. 运行 `node scripts/generate_training_pngs.cjs` → 生成 PNG + YOLO txt + `data.yaml`
3. 运行 `backend/scripts/prepare_pairs.py` → 读取每张 PNG 的 YOLO txt + 对应 JSON，按 (行分组, 时序) 编码为 token 序列，存为 `pairs.npz`（输入: token ids, 目标: 结构化 JSON token 流）

### Transformer 设计

- 编码器输入：每行 token 序列（行内按 x 排序的几何 token + 行间分隔符）
- 解码器输出：结构化 token 流（`{"title": "x", "measures": [{"notes": [...]}]}` 的 tokenized 表示）
- 规模：2 层 encoder + 2 层 decoder，d_model=128，~1M 参数
- 训练：从 `pairs.npz` 直接监督

### 推理流程

1. YOLOv8 检测得到 `[(cls, cx, cy, w, h, conf), ...]`
2. 按 `cy` 聚类分行（同一行内 y 差异 < 阈值）
3. 每行按 `cx` 排序，形成 token 序列
4. Transformer 解码出结构化 token 流
5. token 流 → JSON（用状态机解析器按 vocab 还原字段）
6. 用现有 `engine/renderer` 渲染回图片供对比

## 性能与可靠性

- YOLOv8s 输入 640×640，CPU 上单图 ~200ms；Transformer ~50ms
- 单图端到端 < 1s（CPU），GPU < 200ms
- 限制上传图片 ≤ 10MB，超时 30s
- 错误处理：YOLO 无检测结果时返回空 Score 模板；Transformer 输出非法 JSON 时降级为纯规则拼装
- 日志：复用 Python logging，记录请求耗时、检测数量、解析失败原因

## 设计风格

沿用现有项目风格：暗/亮双主题、简洁信息密度、工具型 UI。识别对比页采用现代仪表盘风：

- **顶部工具栏**：标题、主题切换、布局切换（左右/上下）
- **左侧（上方）面板**：原图（带缩放/重置），可拖拽上传或点击选择
- **右侧（下方面板）**：识别 JSON 渲染结果 + 状态指示器（识别中/成功/失败）
- **底部操作区**：下载 JSON、拷贝 JSON、重新识别、查看原始 JSON 文本
- **识别状态**：进度条 + 步骤提示（"检测中" → "拼装中" → "完成"）
- **布局切换**：左右分栏（默认，适合宽屏）/ 上下堆叠（适合竖屏）
- **错误态**：低置信度音符用半透明红色高亮标注（保留能力，初始版本不强制开启）

## 设计原则

- 与现有 Editor/Preview/TrainingViewer 视觉一致（同样的头部栏、配色变量、按钮风格）
- 拖拽上传区域有明显的虚线边框 + hover 动画
- 渲染结果 Canvas 与原图同尺寸，方便并排像素级对比
- 不引入新组件库（沿用 Tailwind + lucide-react，与现有 App.tsx 一致）

## 单页块设计（识别对比页）

### 1. 顶部工具栏

- 左：标题 "图片简谱识别"
- 中：布局切换按钮组（左右/上下）
- 右：主题切换、返回编辑器

### 2. 上传区（识别前）

- 大尺寸虚线边框拖拽区
- 中央图标 + 提示文字 "拖拽简谱图片或点击上传"
- 副提示 "支持 PNG/JPG，最大 10MB"

### 3. 识别进度区

- 步骤进度条：上传 → YOLO 检测 → 序列拼装 → 完成
- 当前步骤文字提示
- 取消按钮

### 4. 对比区（识别后）

- 左/上：原图（带可缩放容器）
- 右/下：渲染结果 Canvas
- 中间：可拖拽分割条
- 底部：操作按钮（下载 JSON、拷贝 JSON、查看原始 JSON、重新识别）

### 5. 原始 JSON 抽屉

- 从右侧滑出，显示识别出的 JSON 文本（只读）
- 顶部有关闭按钮和"复制全部"按钮

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 深入调研 src/engine/renderer.ts 和 layout.ts 的边界框生成逻辑细节，确保 prepare_pairs.py 能从 layout 反推出 token 序列的事实标签
- Expected outcome: 拿到每个 NoteLayout 字段对应的 token id 映射表，作为 tokenizer 的真值来源