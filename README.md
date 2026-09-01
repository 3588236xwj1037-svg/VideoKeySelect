# 基于视觉语言模型的长视频问答 QATSS 关键帧选择方法研究

面向长视频多选问答任务，探索如何在固定的视觉输入预算下选择更有价值的关键帧，并将选中帧按时间顺序输入视觉语言模型完成推理。

项目以 NExT-QA 为评测任务，构建了从视频解码、候选帧生成、关键帧选择到 Qwen2.5-VL 推理和结果记录的完整流水线。核心方法为 **QATSS**（Query-Aware Temporal Semantic Sampling）。

## 项目目标

直接将长视频的全部帧输入视觉语言模型会带来较高的计算开销，也容易引入大量重复画面。本项目将视频理解问题拆分为两个阶段：

1. 从视频中以固定频率生成候选帧。
2. 在固定帧预算内选出与问题相关、视觉内容有变化且时间覆盖较合理的帧，再交给视觉语言模型作答。

## 方法概览

### 基线方法

| 方法 | 选帧规则 | 作用 |
| --- | --- | --- |
| Uniform-K | 在整段视频中均匀选择 K 帧 | 时间覆盖基线 |
| Random-K | 固定随机种子，从候选帧中无放回随机选择 K 帧 | 无语义随机基线 |
| CLIP Top-K | 按问题文本与候选帧的 CLIP 语义相似度选择前 K 帧 | 查询相关性基线 |
| QATSS | 融合查询相关性、视觉新颖性与时间约束 | 项目核心方法 |

### Uniform-K：均匀时间采样

设视频经粗采样后得到按时间排序的候选帧序列：

$$
\mathcal{C} = (x_i)_{i=0}^{N-1}
$$

其中 `N` 为候选帧数，帧预算为 `K`，实际选择数量为 `K' = min(K, N)`。当 `K' > 1` 时，Uniform-K 的第 `j` 个采样索引为：

$$
i_j = \lfloor \frac{j(N - 1)}{K' - 1} \rfloor,
\quad j = 0, 1, \ldots, K' - 1
$$

若视频帧率为 `f`，对应时间戳为：

$$
t_{i_j} = \frac{i_j}{f}
$$

该方法不使用问题文本或图像语义信息，但能保证从视频开始到结束的时间覆盖，因此作为最基础的时间采样基线。

### Random-K：随机无放回采样

Random-K 从 `N` 个候选帧中均匀地、不重复地选取 `K'` 个帧索引：

$$
S \sim \mathrm{Uniform}(\mathcal{A}_{N,K'}),
\qquad
|\mathcal{A}_{N,K'}| = \binom{N}{K'}
$$

其中，`\mathcal{A}_{N,K'}` 表示从 `N` 个候选帧中选择 `K'` 个帧的全部合法组合。

任意一组合法帧组合被选中的概率相同：

$$
P(S = A) = \frac{1}{\binom{N}{K'}}
$$

单个候选帧被选中的边际概率为：

$$
P(i \in S) = \frac{K'}{N}
$$

实现中固定随机种子以保证单次实验可复现，选中索引会在输入视觉语言模型前重新按时间升序排序。Random-K 不观察问题和视觉内容，用于衡量无语义选帧的表现。

### CLIP Top-K：问题-帧语义相关性

对第 `i` 个候选帧 `x_i` 使用 CLIP 图像编码器得到特征，并对问题 `q` 使用文本编码器得到文本特征。先进行 L2 归一化：

$$
\mathbf{f}_i = \frac{E_I(x_i)}{\|E_I(x_i)\|_2},
\qquad
\mathbf{u} = \frac{E_T(q)}{\|E_T(q)\|_2}
$$

帧与问题的相关性分数为归一化特征的点积，也就是余弦相似度：

$$
r_i = \mathbf{f}_i^\top \mathbf{u}
$$

CLIP Top-K 按 `r_i` 从高到低取前 `K'` 个索引：

$$
S_{\mathrm{clip}} = \mathrm{TopK}((r_i)_{i=0}^{N-1}, K')
$$

最后将 `S_clip` 中的帧按时间排序。该方法能针对问题优先保留语义相关画面，但可能选择多个相邻的相似帧。

### QATSS：查询感知时序语义采样

对候选帧提取归一化 CLIP 图像特征，同时提取问题文本特征。QATSS 使用问题-帧语义相关性与相邻候选帧间的视觉变化度组成综合分数：

$$
score = alpha * relevance + (1 - alpha) * novelty
$$
其中相关性直接复用 CLIP Top-K 的 `r_i`。对于第 `i` 帧，视觉新颖性定义为其与前一候选帧特征的余弦距离：

$$
v_0 = 0,
\qquad
v_i = 1 - \mathbf{f}_i^\top \mathbf{f}_{i-1}, \quad i \geq 1
$$

为了让相关性和新颖性处于同一数值范围，分别进行 min-max 归一化。对于任意分数序列 `z`：

$$
\mathrm{norm}(z_i) =
\begin{cases}
\dfrac{z_i - \min(z)}{\max(z) - \min(z)}, & \max(z) > \min(z) \\
0, & \max(z) = \min(z)
\end{cases}
$$

综合得分为：

$$
s_i = \alpha \cdot \mathrm{norm}(r_i)
      + (1 - \alpha) \cdot \mathrm{norm}(v_i)
$$

当前实现使用 `alpha = 0.75`，使问题相关性占主要权重。随后按 `s_i` 降序贪心选择帧；若 `S` 为已选集合，候选帧 `i` 只有在满足最小时间间隔 `\Delta` 时才加入：

$$
\forall j \in S, \quad |t_i - t_j| \geq \Delta
$$

实验中使用 `\Delta = 1.5` 秒。若短视频或时间约束导致候选不足 `K'` 帧，则按综合得分依次补齐未选帧。该过程同时考虑问题语义、画面变化和时序覆盖。

最终选中的帧会按时间升序输入 Qwen2.5-VL，以保留事件顺序信息。

## 系统流程

```text
视频文件
  -> OpenCV / Decord 解码与候选帧采样
  -> CLIP 图像特征与问题文本特征
  -> Uniform / Random / CLIP Top-K / QATSS 选帧
  -> 按时间排序的多帧输入
  -> Qwen2.5-VL 多选问答推理
  -> 逐题预测、选帧记录与评测汇总
```

为减少重复计算，候选帧的 CLIP 特征、时间戳和文本特征可缓存；实验中记录选帧索引、时间戳、预测答案和原始输出，便于复核和复现。

## 实验设置

- 数据集：NExT-QA 视频多选问答任务。
- 推理模型：Qwen2.5-VL-3B-Instruct。
- 表征模型：CLIP ViT-B/32（LAION 预训练权重）。
- 候选帧：默认按约 1 FPS 粗采样，并在固定帧预算内选择 4 帧。
- 公平对比：不同选帧方法使用相同视频样本、模型版本、Prompt、帧数、生成参数和答案解析规则。
- 数据划分：以视频 ID 为粒度构建并检查开发/评测集，避免同一源视频跨集合出现。

## 实验结果

### 关键帧选择对比

在固定 4 帧预算下，使用同一套 Qwen2.5-VL 推理配置进行评测。

| 评测集 | 方法 | 正确数 | 准确率 |
| --- | --- | ---: | ---: |
| dev500 | Uniform-K | 369 / 500 | 73.8% |
| dev500 | CLIP Top-K | 370 / 500 | 74.0% |
| dev500 | QATSS v1 | 376 / 500 | 75.2% |
| dev500 | QATSS v2 | 369 / 500 | 73.8% |
| dev1000 | Uniform-K | 724 / 1000 | 72.4% |
| dev1000 | CLIP Top-K | 736 / 1000 | 73.6% |
| dev1000 | QATSS v1 | 740 / 1000 | 74.0% |
| dev1000 | QATSS v2 | 730 / 1000 | 73.0% |

Random-K 在 dev50 上以 3 个随机种子运行，平均准确率为 71.33%，用于观察无语义随机选帧的结果波动。该小规模检查与 dev500/dev1000 的评测集不同，因此不与后两者的数值合并比较。

### QLoRA 对照

项目还将 Uniform-K、CLIP Top-K 与 QATSS 选出的时序帧序列转换为多图 JSONL 数据，使用 4-bit QLoRA 对 Qwen2.5-VL-3B 训练三组适配器，并在 dev1000 上完成对照评测：

| 选帧策略 | 准确率 |
| --- | ---: |
| Uniform-K + QLoRA | 72.8% |
| CLIP Top-K + QLoRA | 72.9% |
| QATSS v1 + QLoRA | 73.4% |

## 目录说明

```text
VideoKeySelect/
├── src/
│   ├── samplers/           # 均匀采样、候选帧与随机采样
│   ├── selectors/          # CLIP Top-K、QATSS 等关键帧选择器
│   └── experiments/        # 选帧、推理与批量评测入口
├── scripts/                # 数据清单、下载、QLoRA 数据构造与结果分析脚本
├── tests/                  # 采样与选择器单元测试
├── data/nextqa/manifests/  # 可复现的数据划分 ID 清单
├── results/                # 小型实验汇总与分析结果
└── external/NExT-QA/       # NExT-QA 数据集代码或说明
```

原始视频、模型缓存、特征缓存、训练检查点和批量生成的帧图像不纳入版本库，相关路径由 `.gitignore` 排除。

## 环境安装

实验在 AutoDL 的 NVIDIA RTX 4090D 24GB GPU 环境上完成，使用 Python 3.10、PyTorch 与 CUDA。可参考下面的基础环境：

```bash
conda create -n videoqa python=3.10 -y
conda activate videoqa

pip install --upgrade pip
pip install \
  transformers accelerate qwen-vl-utils \
  open-clip-torch decord opencv-python-headless \
  numpy pandas pyyaml tqdm scikit-learn matplotlib \
  pillow pytest
```

运行 QLoRA 训练还需要：

```bash
pip install -U ms-swift peft bitsandbytes datasets
```

## 复现说明

1. 按 NExT-QA 的官方许可获取数据集及视频文件，不将受限原始数据上传到本仓库。
2. 使用 `scripts/` 下的数据清单和下载脚本准备指定的视频 ID 与评测集。
3. 使用 `src/experiments/` 下的选帧和推理脚本运行评测。运行前可查看参数说明：

```bash
python -m src.experiments.run_dev500_selection --help
python -m src.experiments.run_dev500_qwen_eval --help
python scripts/build_dev1000_manifest.py --help
```

4. 运行单元测试：

```bash
pytest -q
```

> 提示：首次运行 CLIP 或 Qwen2.5-VL 会下载模型权重；建议将 Hugging Face 缓存目录设在有足够空间的持久化数据盘。

## 局限与后续工作

- CLIP 的图文语义相似度对细粒度动作、计数、文字识别和抽象因果问题存在局限。
- 较低的候选帧采样频率可能遗漏持续时间很短的关键事件。
- 后续可探索分层时间预算、字幕/ASR 融合，以及在严格视频级隔离条件下训练轻量级排序器。

## 数据与模型声明

本仓库不分发 NExT-QA 原始视频、预训练模型权重或训练检查点。使用数据集、模型和第三方代码时，请遵循其各自的许可协议与使用条款。
