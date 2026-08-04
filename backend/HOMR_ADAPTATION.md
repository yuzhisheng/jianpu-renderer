# HOMR 思路的简谱适配方案

## 结论

HOMR 的核心方向适合简谱，但其西方五线谱模型和词表不能直接复用。当前实现采用
同类的“谱行图像编码器 + 自回归多分支事件解码器”，并增加小数据更稳定的 CTC
单调对齐头。HOMR 源码许可证为 AGPL-3.0，本项目只参考公开架构说明，新增代码为
独立实现，`references/homr` 不进入产品源码。

参考资料：

- HOMR: <https://github.com/liebharc/homr>
- HOMR vocabulary: <https://github.com/liebharc/homr/blob/main/Vocabulary.md>
- TrOMR: <https://arxiv.org/abs/2308.09370>

## 当前架构

1. 现有检测器只负责简谱判别和乐谱行定位。
2. 轻量 CNN 将 128×1024 灰度谱行编码为二维特征图。
3. 可分离横/纵位置编码分别表达时间方向与上下修饰符位置。
4. CTC 头输出 `P1..P7 / R0 / - / 小节线`，采用 prefix beam search。
5. Transformer 解码器的每个事件同时包含 `kind / pitch / accidental /
   octave / duration / articulation` 六个分支。
6. 分支标签缺失时使用 `IGNORE_ID`，不把“银标未知”训练成 `NONE`。
7. 输出复用 `parse_tokens_to_score`，直接进入现有网页 Score JSON 渲染链路。

模型共 627,353 个参数，可在 M3 Pro 36GB 上以 MPS 训练。Apple MPS 尚未原生支持
CTCLoss，因此只有 loss 运算回退 CPU，视觉网络和 Transformer 前后向仍在 MPS。

## 数据与结果

| 数据 | 行数 | 事件数 | 用途 |
|---|---:|---:|---|
| 合成训练 | 2,362 | 51,907 | 精确预训练 |
| 合成验证 | 603 | 13,089 | 页面隔离验证 |
| 真实训练 | 237 | 5,823 | 扫描域微调 |
| 真实验证 | 80 | 1,811 | 页面隔离调参 |
| 跨来源测试 | 101 | 2,675 | 不参与 v2 训练 |

自由解码结果：

- 合成验证音高相似度：0.9786。
- 真实验证音高相似度：0.8147。
- 跨来源测试音高相似度：0.5932，计数相似度：0.7652。
- 8 页最终域外盲测音高相似度：约 0.591。

纯视觉序列模型尚未超过生产检测路线，因此 `/recognize` 默认行为没有改变；只有
显式传入 `visual_sequence=true` 才使用新模型。正式切换前需要至少 1,000 条人工
复核真实谱行，并给八度点、减时线、附点和连音建立 gold 多分支标签。

## 复现

完整命令见 `backend/README.md`。关键产物：

- `backend/weights/jianpu_vision_transformer_synthetic.pt`
- `backend/weights/jianpu_vision_transformer_v2.pt`
- `backend/eval_outputs/jianpu_vision_transformer_v2_beam_tuned_test.json`
- `backend/eval_outputs/user_rehuishou_visual_v2.json`
