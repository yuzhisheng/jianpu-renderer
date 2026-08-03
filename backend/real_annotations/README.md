# 真实简谱银标数据

这里的标签全部由模型生成，不能称为人工金标准。不同目录代表不同监督强度：

- `reviews/`：YOLO 检测器底稿和逐内容带裁剪索引，状态默认为待视觉复核。
- `vlm_reviews/`：当前视觉大模型逐行读取后的完整 token 标注，包含不确定项。
- `ground_truth_silver/`：由 `vlm_reviews/` 编译的 Score JSON，可用于真实集评估。
- `local_vlm_pitch_reviews/`：本机 Qwen3-VL-8B 生成的数字/休止符/延音/小节线骨架；
  暂不包含可靠的高低音点、附点和减时线。
- `labels/`：检测器框底稿。未与视觉 token 对齐前不得直接当成完整 YOLO 真值训练。

## 当前覆盖

- 原始图片：49 张。
- 简谱：47 张；五线谱排除：2 张。
- 固定测试页：15 张；训练银标候选：32 张。
- 完整视觉 Score 银标：2 张（`昆仑道`、`Canto Della Terra`）。
- 本地 8B 音高骨架：47 张简谱全部完整；测试集 162 带、训练集 458 带，残留解析错误 0。
- 测试集包含 3635 个骨架 token，训练集包含 8365 个骨架 token。
- 高置信对齐数据集：168 条谱线、3276 个纠正框、110 训练图/58 验证图。
- 8 类模型权重：`backend/weights/pitch8_silver_v1.pt`；银标验证 mAP50 0.935、
  mAP50-95 0.767。独立测试音高相似度从旧模型的 0.4405 提升到 0.7662。

两张完整视觉银标显示，当前 YOLO 在真实页上的平均音高编辑相似度为 0.1695，
完整音符为 0.0855；这比合成验证集低得多。本地 Qwen3-VL-8B 在两张校准页上的
纯音高 token 相似度分别为 0.8455 和 0.9134，但包含小节线和增时线后分别为
0.6199 和 0.8655，因此它适合生成银标候选，不应作为无人复核的测试真值。

## 重建与恢复

```bash
# 检测器底稿、完整原图副本和内容带
backend/.venv/bin/python backend/scripts/bootstrap_real_annotations.py \
  --device mps --imgsz 1280

# 编译已经复核的完整 Score 标签
backend/.venv/bin/python backend/scripts/compile_vlm_reviews.py

# 计算当前模型在视觉银标页上的真实指标
backend/.venv/bin/python backend/scripts/evaluate_vlm_annotations.py

# 本地 8B 继续批量标测试页；逐内容带写检查点，可安全中断和恢复
HF_HOME=backend/.cache/huggingface \
backend/.venv-vlm/bin/python backend/scripts/annotate_rows_with_local_vlm.py \
  --split test

# 仅从缓存检测框安全刷新指定页面的内容带，不重跑检测器
backend/.venv/bin/python backend/scripts/refresh_annotation_rows.py \
  --only nas_tianbian_jp

# 从完整 VLM 银标构建经过过滤和单调对齐的 8 类训练集
backend/.venv/bin/python backend/scripts/build_pitch_silver_dataset.py
```

安装本地视觉环境：

```bash
python3.10 -m venv backend/.venv-vlm
backend/.venv-vlm/bin/pip install -r backend/requirements-vlm.txt
```
