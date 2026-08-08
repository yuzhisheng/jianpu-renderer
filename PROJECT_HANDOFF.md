# 简谱网页编辑器｜项目交接说明

最后更新：2026-08-08

这份文档用于在另一台电脑上快速恢复项目背景、启动开发环境并继续实现功能。

## 1. 项目目标

将 Windows 简谱制谱工具 JP-Word 的核心能力逐步实现为网页版本，重点覆盖：

- 所见即所得的简谱编辑
- 音符、时值、拍号、小节线和歌词编辑
- 自动排版和规则检查
- 图片简谱识别
- 图片、JSON，以及后续 SVG/PDF/MIDI 等格式的输入输出

当前项目名称是 jianpu-renderer，前端已经从“JSON → Canvas 预览”发展为一个带基础可视化编辑能力的 MVP。

## 2. 当前 Git 状态

当前本地分支：master

当前远程状态：本地 master 与 origin/master 已同步，工作区干净。

最近三个重要 commit：

    ca93705 chore: 提交当前识谱与渲染改动
    9123aab docs: 添加项目交接与开发指南
    259c02e feat: 增加可视化简谱编辑器 MVP

ca93705 包含了此前暂存的识谱、渲染、识谱面板、测试和生成示例图片改动。模型权重和训练数据仍按 .gitignore 规则不进入 Git。

## 3. 快速启动前端

要求：Node.js、npm。

    cd jianpu-renderer
    npm ci
    npm run dev

Vite 默认开发地址：

    http://localhost:5173/

其他常用命令：

    npm run build
    npm run preview
    npm run lint

当前 npm run build 已验证通过。全仓库 npm run lint 仍有历史遗留问题，主要集中在后端识谱接口、识谱面板和 Canvas 渲染代码；本次编辑器新增文件的针对性 ESLint 检查已通过。

## 4. 前端当前功能

打开网页后，左侧是编辑器，右侧是简谱预览。

### 可视化编辑

- 点击右侧音符选中
- 修改音高 0–7
- 修改十六分、八分、四分、二分、全音符时值
- 添加升号、降号、还原号
- 上下调整八度
- 切换附点
- 编辑当前音符歌词
- 在当前音符后插入音符
- 删除音符
- 添加小节
- 修改标题、调号、速度、拍号
- 修改当前小节线类型
- 撤销/重做，最多保留 50 步

### JSON 模式

左侧顶部可以在“谱面”和“JSON”之间切换。JSON 模式使用 Monaco Editor，适合直接编辑完整 Score JSON 或调试渲染结果。

### 其他已有功能

- 示例乐谱切换
- 暗色/亮色主题
- 音符间距、行间距和预览缩放
- PNG 导出
- 图片简谱识别页面
- 训练素材查看页面
- 浏览器本地草稿保存

## 5. 前端架构

### 入口和状态

src/App.tsx

- 管理当前 Score
- 管理选中音符地址 { measureIndex, noteIndex }
- 管理撤销/重做历史
- 管理 JSON 同步和 localStorage 草稿
- 连接编辑器、预览和工具栏

### 编辑器面板

src/components/ScoreEditorPanel.tsx

负责可视化控件，包括音高、时值、升降号、八度、歌词、拍号和小节线。

### 编辑操作和规则检查

src/editor/score.ts

集中放置不可变编辑操作：

- updateNote
- insertNoteAfter
- deleteNote
- addMeasure
- updateMeasure
- updateScoreMeta
- validateScore

后续新增编辑功能应优先放在这里，不要把数据修改逻辑散落到 UI 组件中。

### 排版和渲染

    src/engine/layout.ts    Score → 坐标布局
    src/engine/renderer.ts  Canvas 绘制
    src/engine/symbols.ts   符号路径和图形
    src/engine/export.ts    PNG 导出

当前 Canvas 渲染器已经支持音符、休止符、减时线、增时线、八度点、附点、升降号、歌词、技巧符号、连音线、圆滑线、反复记号等。

src/components/Preview.tsx 负责 Canvas 生命周期、命中检测和选中框显示。

## 6. Score 数据模型

类型定义位于 src/types/jianpu.ts，核心结构如下：

    Score
     ├─ title / key / timeSignature / tempo
     └─ measures[]
         └─ notes[]
             ├─ Note
             └─ Dash

目前 Note 已支持：

- pitch
- octave
- duration
- beamLevel
- dot
- accidental
- techniques
- tieId
- slurId
- tripletId
- lyric / lyrics
- 多种力度和演奏记号

当前编辑地址仍然主要依赖小节索引和音符索引。下一阶段应该给音符和附属对象增加稳定 ID，以便支持复杂拖拽、复制粘贴和多声部编辑。

## 7. 图片识谱后端

后端是 FastAPI 服务，代码位于 backend/。

安装和启动：

    cd backend
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Windows 激活虚拟环境的命令是：

    .venv\Scripts\activate

主要接口：

    GET  /health
    POST /recognize

前端默认请求地址在 src/api/recognize.ts 中定义：

    http://localhost:8000

可以通过环境变量覆盖：

    VITE_API_BASE=http://localhost:8000 npm run dev

后端需要模型权重才能正常识谱。backend/weights/ 被 .gitignore 忽略，不能依靠普通 Git clone 自动恢复；换电脑时需要另外准备模型文件或重新训练。

后端详细训练和评估说明见 backend/README.md。

## 8. 下一阶段建议

### P0：编辑体验

- 键盘输入 1–7、0
- 空格切换音值组合
- Tab 根据拍号自动插入小节线
- 选区和连续音符操作
- 浏览器下载/导入 JSON
- 更准确的选中区域和歌词命中检测

### P1：专业排版

- 根据音符时值计算前进宽度，而不是固定音符宽度
- 手动锁定音符间距
- SVG 渲染和导出
- PDF/打印预览
- 多段歌词和自动歌词对位
- 稳定 ID 的 Attachments 模型

### P2：复杂乐谱

- 房子和反复跳跃
- 连音线、圆滑线的可视化编辑
- 多声部和连谱号
- 特殊文字、力度、技法对象
- 页面文本框和分页

### P3：音乐能力

- Web Audio 播放
- MIDI 导入导出
- JPW-ABC 或自有文本格式解析
- 动态谱播放指示条
- 协作和云端保存

## 9. JP-Word 调研资料

实现参考：

- JP-Word 官网介绍：https://www.happyeo.com/intro_jpw.htm
- JP-Word 4.0 使用手册：https://happyeo.gitbooks.io/jp-word-manual/content/01/00.html
- 基本符号：https://happyeo.gitbooks.io/jp-word-manual/content/04/03.html
- 附属符号：https://happyeo.gitbooks.io/jp-word-manual/content/04/04.html
- 排版操作：https://happyeo.gitbooks.io/jp-word-manual/content/04/13.html
- JPW-ABC 记谱法：https://happyeo.gitbooks.io/jp-word-manual/content/06/01.html

注意：公开手册主要对应 JP-Word 4.0，官网当前版本更高。因此手册适合参考功能和数据模型，不应视为最新版的完整协议说明。

## 10. 建议的继续工作流程

在新电脑上：

    git clone https://github.com/yuzhisheng/jianpu-renderer.git
    cd jianpu-renderer
    git checkout master
    npm ci
    npm run dev

开始新功能前先检查：

    git status
    git log -5 --oneline

完成一个相对独立的功能后建议：

    npm run build
    git diff --check
    git add <明确的文件>
    git commit -m "feat: ..."

避免把模型权重、训练数据、生成图片和无关实验文件混入前端功能 commit。
