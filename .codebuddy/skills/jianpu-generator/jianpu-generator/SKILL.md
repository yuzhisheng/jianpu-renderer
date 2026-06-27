---
name: jianpu-generator
description: |
  随机简谱生成器。当用户需要生成简谱测试数据、随机乐谱、
  批量创建示例乐谱时使用此技能。支持控制小节数、拍号、
  调号、速度等参数。
---

# 随机简谱生成器

使用 `scripts/generate_jianpu.py` 脚本生成合法的简谱 Score JSON 数据。

## 用法

```bash
python3 scripts/generate_jianpu.py [选项]
```

### 可选参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--measures N` | 小节数 | 随机 12-24 |
| `--beats N` | 每小节拍数 | 4 |
| `--key C` | 调号 | 随机 |
| `--tempo N` | 速度 BPM | 随机 60-120 |
| `--title "xxx"` | 标题 | 自动生成 |
| `--intro` | 添加前奏括号 | 否 |
| `--seed N` | 随机种子 | 否 |
| `--output file.json` | 输出文件 | stdout |

### 示例

```bash
# 生成 16 小节 4/4 拍乐谱
python3 scripts/generate_jianpu.py --measures 16 --beats 4 --key C --output test.json

# 生成带前奏括号的乐谱
python3 scripts/generate_jianpu.py --measures 20 --intro --seed 42

# 生成 3/4 拍乐谱
python3 scripts/generate_jianpu.py --beats 3 --measures 12
```

## 生成内容

生成的 JSON 包含随机组合的：

- **节奏型**：1/4、1/2、1、附点、2 等
- **装饰符号**：波音、倚音、重音、保持音、延长记号
- **连音线**：相邻同节奏音符随机添加圆滑线
- **小节线**：随机 double/repeat/end
- **休止符**和**增时线**
- **多行歌词**：汉字+拼音
- **小房子**：随机 1./2. ending

输出格式可直接用于 `src/data/examples.ts` 的 Score 类型。
