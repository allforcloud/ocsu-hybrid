# OCSU-Hybrid

**OCSU 论文中 OCSR-based 路线的模块化、可运行实现。**

[English](README.md) | **简体中文**

输入一张分子结构图，输出 SMILES、官能团、IUPAC 名称和英文描述。整条流水线只使用公开的专用模型，可在单张 8 GB 显卡上运行，不需要 7B 视觉语言模型。

```
图像 ──► MolScribe ──► SMILES ──┬──► PubChem ──────► IUPAC 名称
        (95.5 M)               ├──► BioT5+ ───────► 分子描述
                                │    (252 M)
                                └──► RDKit 字典 ──► 官能团（备用）
        图像 ───────────────► ConvNeXt-T ────────► 官能团
                               (27.8 M，默认)
```

---

## 结果

全量 Vis-CheBI20 test split，指标定义与论文一致。下表把论文的两条路线与本项目放在一起比较。

| 任务 | 指标 | **OCSU-Hybrid** | OCSR-based：MolScribe | OCSR-based：DoubleCheck | OCSR-free：Mol-VL-2B | OCSR-free：Mol-VL-7B |
|---|---|---:|---:|---:|---:|---:|
| SMILES | Acc_s | **84.00** | 80.50 | 82.40 | — | — |
| SMILES | Acc_c | **80.55** | 75.10 | 77.60 | — | — |
| SMILES | Tanimoto | 92.03 | 90.80 | **92.70** | — | — |
| 官能团 | F1 | **99.07** | 89.60 | 93.63 | 95.40 | 97.32 |
| IUPAC | BLEU-4 | **65.98** | 61.03 | 64.23 | 72.79 | **76.95** |
| IUPAC | ROUGE-1 | **85.39** | 82.73 | 84.45 | — | 81.88 |
| 分子描述 | BLEU-4 | **46.83** | 44.64 | 45.56 | 40.26 | 46.14 |
| 分子描述 | METEOR | **58.99** | 57.09 | 58.07 | — | 58.14 |

**如何理解 IUPAC 这一行。** 本项目从 PubChem **检索**名称，而不是生成名称。在找到记录的 87.72% 测试分子上，BLEU-4 为 **99.01**，**91.20%** 的名称与 ground truth 逐字一致；按全部 2,680 条计算，端到端逐字一致率为 80.0%。与 Mol-VL-7B 的 76.95 相比，差距主要在 **覆盖率**，不在检索质量：剩余 329 条未命中样本以 OCSR 错误为主（预测 SMILES 与 GT 的 Tanimoto 中位数为 0.794）。

四个头条指标（SMILES Acc_s、官能团 F1、IUPAC BLEU-4、描述 BLEU-4）均达到或超过论文最佳 OCSR-based 列；官能团超过两种规模的 Mol-VL，IUPAC 超过两列 OCSR-based。唯一没有超过的是 Mol-VL-7B 的生成式 IUPAC 分数；SMILES Tanimoto 比 DoubleCheck 低 0.67（92.03 vs 92.70）。详见[限制](#限制)。

---

## 论文两条路线与本项目的技术对比

| | **论文：OCSR-based** | **论文：OCSR-free** | **本项目** |
|---|---|---|---|
| 核心思想 | `P(T\|I) = P(S\|I) · P(T\|S)`：先识别结构，再理解结构 | 单个 VLM 端到端学习 `P(T\|I)` | OCSR-based，官能团改用图像分类头 |
| 图像 → SMILES | DoubleCheck 或 MolScribe | Mol-VL 直接生成 | **MolScribe** 公开权重 |
| 官能团 | RDKit + OCSU 字典 | Mol-VL 生成 | **ConvNeXt-T 图像分类头**（默认）；字典备用 |
| IUPAC | PubChem 检索 | Mol-VL 生成 | **PubChem 检索** |
| 分子描述 | BioT5+ | Mol-VL 生成 | **BioT5+**，输入前先做 RDKit 规范化 |
| 名称 grounding | 数据库记录 | 无 | **数据库记录** |
| 训练 | OCSR 单独训练（DoubleCheck 4×A100×4 天；MolScribe 4×A100×1 天）；下游不训练 | 4×A100×10 天，50 epoch，全参数 SFT | **只训练 27.8 M 的官能团头**，RTX 4060 约 30 分钟；OCSR 与 BioT5+ 使用公开权重 |
| 推理栈 | OCSR + BioT5+ + PubChem | 单个 2B / 7B VLM | OCSR + 官能团头 + BioT5+ + PubChem |
| 主要优势 | IUPAC 名称是真实数据库记录；模块可替换 | 单模型，生成文本较强 | 真实记录 + 图像官能团头；小、可复现 |
| 主要劣势 | IUPAC 覆盖率受 OCSR 限制 | 名称无 grounding；训练和推理开销大 | IUPAC 覆盖率仍受 OCSR 限制 |

### 资源对比

| 项目 | **本项目** | 论文：OCSR-based | 论文：OCSR-free |
|---|---|---|---|
| 流水线参数量 | **约 0.38 B**（MolScribe 95.5 M + ConvNeXt-T 27.8 M + BioT5+ 252 M） | OCSR（MolScribe 级，约 95 M）+ BioT5+ 252 M | 2 B / 8.29 B（7B） |
| 需要训练的组件 | **仅官能团头，27.8 M** | OCSR backbone | 完整 VLM |
| 训练算力 | **1× RTX 4060，约 30 分钟** | DoubleCheck 4×A100×4 天；MolScribe 4×A100×1 天 | 4×A100×10 天 |
| 推理显存 | **RTX 4060 8 GB 可跑全部模块** | OCSR + 文本模型 | bf16 需 24 GB+；4-bit 约 6 GB，但更慢 |
| 权重磁盘占用 | **约 2.2 GB** | 未报告 | 7B 约 16.6 GB |
| 全量 test（四个任务） | **单卡 4060 约 2 小时** | 未报告 | 7B 4-bit、3 秒/图时，单任务约 2.75 小时 |
| 外部依赖 | IUPAC 需要 PubChem 网络 | IUPAC 需要 PubChem 网络 | 生成过程无外部依赖 |

8 GB 显存是实测结论：3,300 张 SMILES、3,269 张官能团、2,680 次 PubChem 查询和 3,300 条 BioT5+ 描述均在单张 RTX 4060 上完成。

---

## 已实现与未实现

**已实现**

- MolScribe 图像 → SMILES，已 vendor 到仓库并包含兼容补丁。
- ConvNeXt-T 140 类官能团分类头，使用 Vis-CheBI20 caption 训练。
- PubChem IUPAC 检索：canonical SMILES、InChIKey、Kekulé、CID、最大片段多级 fallback；只缓存命中结果。
- BioT5+ 分子描述，输入 SELFIES 前先做 RDKit canonicalisation。
- 可选 OPSIN 名称 → 结构校验。
- 四个任务的论文口径评测。

**未实现**

- **Mol-VL / OCSR-free 路线。** 不随项目发布 7B 模型。在 100 张 test 图上，它的图像 → SMILES 为 **0.00 Acc_s**（Tanimoto 0.177），同子集 MolScribe 为 0.81。作为 IUPAC 兜底时，生成名称 **0/329** 结构正确。详见 [`docs/negative_results.md`](docs/negative_results.md)。
- **MolSight 后端。** 已移除；MolScribe 是唯一 OCSR 后端。
- **OCSR 微调权重。** MolScribe 和 MolSight 在 25.9k 训练图上的微调均崩溃；保留公开 checkpoint。
- **MolScribe beam search。** 上游 decoder 未实现该分支。

---

## 快速开始

### 1. 安装

需要 Python 3.10+；OPSIN 校验需要 Java 11+（可选）。

```bash
git clone https://github.com/allforcloud/ocsu-hybrid.git
cd ocsu-hybrid

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 先安装与驱动匹配的 CUDA 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -e ".[all]"
```

> 本项目以源码 checkout 方式运行，请使用 `pip install -e .`。MolScribe worker 和 vendored MolScribe/OpenNMT 源码位于 `tools/` 与 `third_party/`，不在 Python 包内；普通 wheel 安装不是受支持的分发方式。

仅 CPU 也可以运行：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

### 2. 下载公开权重

```bash
# MolScribe、BioT5+、OPSIN、NLTK 数据
python scripts/download_models.py --all

# 官能团头（发布在 Hugging Face，不提交到 git）
python scripts/download_models.py --fg \
  --fg-repo allforcloud9527/ocsu-hybrid-fg-convnext-t
```

检查：

```bash
python scripts/download_models.py --check
```

期望目录：

```
models/
├── molscribe/swin_base_char_aux_1m680k.pth
├── biot5-plus-chebi20/
├── fg_classifier/model.pt
└── opsin/opsin-cli.jar
```

### 3. 单图分析

```bash
ocsu analyze examples/vischebi20_000000.png
```

真实输出（描述已截断）：

```json
{
  "image": ".../examples/vischebi20_000000.png",
  "smiles": "[H][C@@]12CCC3=CC(=O)CC[C@]3(C)[C@@]1([H])C(=O)C[C@]1(C)/C(=C\\C(=O)OC)CC[C@@]21[H]",
  "canonical_smiles": "COC(=O)/C=C1/CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3C(=O)C[C@]12C",
  "valid": true,
  "functional_groups": [
    "alkenes", "carbonyl methyl ester", "cyclohexane", "cyclopentane",
    "esters", "ketones", "methoxy", "side-chain aldehydes"
  ],
  "iupac": {
    "name": "methyl (2Z)-2-[(8S,9S,10R,13S,14S)-10,13-dimethyl-3,11-dioxo-1,2,6,7,8,9,12,14,15,16-decahydrocyclopenta[a]phenanthren-17-ylidene]acetate",
    "source": "inchikey",
    "status": "ok"
  },
  "description": {
    "text": "The molecule is a 3-oxo Delta(4)-steroid that is ...",
    "status": "ok"
  }
}
```

常用选项：

```bash
# 跳过较慢的阶段
ocsu analyze molecule.png --no-iupac --no-description

# 不下载官能团头，改用 RDKit 字典（91.11 F1）
ocsu analyze molecule.png --fg-backend dictionary

# 用 OPSIN 校验检索到的名称
ocsu analyze molecule.png --verify-iupac

# 强制 CPU
ocsu analyze molecule.png --device cpu
```

### 4. 批量处理

```bash
ocsu batch examples/ -o outputs/examples.jsonl
ocsu batch images/ -o outputs/images.jsonl --recursive
ocsu batch list.txt -o outputs/list.jsonl      # 每行一个路径
```

批处理会对整块图像只加载一次 MolScribe，然后批量跑官能团头，并并行查询 PubChem。

### 5. 全量 test

```bash
# 从 https://huggingface.co/datasets/PharMolix/Vis-CheBI20 下载数据，
# 将 --data-dir 指向包含 test.json 与 imgs/ 的目录。

ocsu eval --data-dir /path/to/vischebi20 --out results/full_test
```

先跑 smoke test（每个任务 20 条）：

```bash
ocsu eval --data-dir /path/to/vischebi20 --out results/smoke --limit 20
```

等价脚本：

```bash
python scripts/run_full_test.py --data-dir /path/to/vischebi20 \
  --out results/full_test --iupac-workers 4
```

runner 支持断点续跑：OCSR、官能团、IUPAC、描述结果会追加到 `results/full_test/preds/`，PubChem 中断不会重跑 GPU 阶段。续跑只按图像去重；如果更换 OCSR checkpoint 或官能团后端，请删除对应的预测文件（或整个 `preds/` 目录），避免混用不同配置的结果。

输出结构：

```
results/full_test/
├── metrics.json
├── summary.txt
├── preds/{ocsr,smiles,fg,iupac,desc}.jsonl
└── details/{smiles,fg,iupac,desc}.jsonl
```

### 6. Python API

```python
from ocsu_hybrid.pipeline import Pipeline

pipeline = Pipeline()                       # 默认启用全部四个阶段
result = pipeline.analyze("examples/vischebi20_000000.png")
print(result["smiles"])
print(result["functional_groups"])
print(result["iupac"]["name"])
print(result["description"]["text"])

rows = pipeline.analyze_batch(["a.png", "b.png"], use_description=False)
```

---

## 复现论文对比结果

README 中的指标使用仓库内的 `src/ocsu_hybrid/evaluate/` 评测代码。全量复现：

```bash
ocsu eval --data-dir /path/to/vischebi20 --out results/full_test --iupac-workers 4
```

RTX 4060 8 GB 上的期望结果：

```
SMILES      Acc_s 84.00  Acc_g 86.94  Acc_c 80.55  Tanimoto 92.03
FG          F1 99.07  P 99.31  R 99.00  exact 92.87
IUPAC       coverage 87.72%  BL-4 65.98  R-1 85.39  METEOR 85.41
Description BL-4 46.83  R-2 47.96  R-L 55.89  METEOR 58.99
```

这些数值已用本仓库 evaluator 对全量预测文件独立复算，与表格完全一致。指标定义、端到端与条件分口径、SciBERT tokenizer、论文 F1 表 P/R 列互换问题，以及 BioT5+ tokenizer artifact，见 [`docs/evaluation.md`](docs/evaluation.md)。

---

## 模型与数据

| 组件 | 来源 | 大小 | 许可 | 是否提交到 git |
|---|---|---|---:|---|
| MolScribe checkpoint | [`yujieq/MolScribe`](https://huggingface.co/yujieq/MolScribe) | 1.1 GB | MIT | 否 |
| BioT5+ | [`QizhiPei/biot5-plus-base-chebi20`](https://huggingface.co/QizhiPei/biot5-plus-base-chebi20) | 0.96 GB | MIT | 否 |
| 官能团 ConvNeXt-T 头 | [`allforcloud9527/ocsu-hybrid-fg-convnext-t`](https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t) | 107 MB | Apache-2.0 | 否（HF） |
| OPSIN 2.9.0 | [dan2097/opsin](https://github.com/dan2097/opsin) | 14 MB | MIT | 否 |
| SciBERT 词表 | [allenai/scibert](https://github.com/allenai/scibert) | 223 KB | Apache-2.0 | **是** |
| MolScribe 源码 | [thomas0809/MolScribe](https://github.com/thomas0809/MolScribe) | 14 MB | MIT | **是**，已 vendor |
| OpenNMT-py 2.2.0 最小包 | [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) | 1 MB | MIT | **是**，已 vendor |
| Vis-CheBI20 | [`PharMolix/Vis-CheBI20`](https://huggingface.co/datasets/PharMolix/Vis-CheBI20) | 3,300 张 test 图 | Apache-2.0 | 仅一张示例图 |

官能团 checkpoint 发布在 Hugging Face，不提交到仓库。模型卡为 [`docs/fg_model_card.md`](docs/fg_model_card.md)，训练脚本为 `scripts/train_fg_classifier.py`。

### 自己训练官能团头

```bash
python scripts/build_fg_dataset.py --data-json /path/to/train.json
python scripts/train_fg_classifier.py \
  --train-json /path/to/vischebi20/train.json \
  --images-root /path/to/vischebi20 \
  --out models/fg_classifier
```

即可复现 99.07 F1 checkpoint：ConvNeXt-T、384²、140 类、batch 32、12 epoch，RTX 4060 约 30 分钟。

### 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 各阶段设计、数据流、失败行为、运行时 |
| [`docs/evaluation.md`](docs/evaluation.md) | 指标定义、论文 evaluator 复现、覆盖率与条件分 |
| [`docs/negative_results.md`](docs/negative_results.md) | 被否决的 7B、微调、beam search、2B 实验 |
| [`docs/fg_model_card.md`](docs/fg_model_card.md) | 官能团头的 Hugging Face 模型卡 |

---

## 发布官能团头与 GitHub 仓库（可选）

凭据放在 `configs/local.yaml`，该文件已被 **gitignore**。Hugging Face 使用 write token；GitHub 默认使用本地 SSH key 推送，只有需要脚本自动创建仓库时才需要 GitHub token。`HF_TOKEN`、`GITHUB_TOKEN` 环境变量会覆盖文件中的值。

```bash
cp configs/local.example.yaml configs/local.yaml
# 填写用户名、token 和仓库名
```

配置格式：

```yaml
huggingface:
  username: "your-hf-username"
  token: "hf_..."                       # write token；`hf auth login` 后可为空
  fg_repo: "your-hf-username/ocsu-hybrid-fg-convnext-t"

github:
  username: "your-github-username"
  auth: "ssh"                           # ssh | token
  token: ""                             # 仅 auth: token 或 API 建仓时需要
  repo: "your-github-username/ocsu-hybrid"
  branch: "main"
  private: false                        # 公开仓库
  create_repo: "auto"                   # auto | gh | api | never
```

`fg_repo` 与 `repo` 既可以是 `owner/name`，也可以是完整浏览器/SSH URL，脚本会自动规范化。

上传官能团头：

```bash
python scripts/publish_fg_model.py --dry-run
python scripts/publish_fg_model.py --model /path/to/model.pt
```

脚本会按需创建公开 HF 模型仓库，上传 `model.pt` 与模型卡 `README.md`。不传 `--model` 时默认读取 `models/fg_classifier/model.pt`。

创建并推送 GitHub 仓库：

```bash
python scripts/publish_github.py --dry-run
python scripts/publish_github.py
```

`auth: ssh` 时，脚本用 `git ls-remote` 检查仓库，确认 `configs/local.yaml` 已被忽略且未跟踪，提交项目并通过 SSH 推送。**不会 force push**。仓库不存在时，会优先用已登录的 `gh repo create`，其次用配置的 token 调 API，否则停止并提示手动创建空仓库。SSH key 不会出现在 URL 中。

本仓库已发布地址：

- GitHub：https://github.com/allforcloud/ocsu-hybrid
- Hugging Face：https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t

---

## 环境变量

所有路径都可以不修改代码直接覆盖：

| 变量 | 默认值 |
|---|---|
| `OCSU_HOME` | 仓库根目录 |
| `OCSU_MODELS` | `<root>/models` |
| `OCSU_DATA` | `<root>/data/vischebi20` |
| `OCSU_CACHE` | `<root>/.cache` |
| `MOLSCRIBE_CKPT` | `<models>/molscribe/swin_base_char_aux_1m680k.pth` |
| `MOLSCRIBE_SRC` | `<root>/third_party/MolScribe` |
| `OCSU_FG_CKPT` | `<models>/fg_classifier/model.pt` |
| `BIOT5_PATH` | `<models>/biot5-plus-chebi20` |
| `OCSU_OPSIN_JAR` | `<models>/opsin/opsin-cli.jar` |
| `MOLSCRIBE_PY` | 当前 Python 解释器 |
| `HF_TOKEN` | 覆盖 `configs/local.yaml` 中的 `huggingface.token` |
| `GITHUB_TOKEN` | 覆盖 `configs/local.yaml` 中的 `github.token` |

默认情况下，MolScribe worker 使用当前 Python 解释器。如果 OCSR 栈在独立环境，可设置 `MOLSCRIBE_PY=/path/to/python` 或传入 `--molscribe-python`。

---

## 限制

1. **IUPAC 覆盖率受 OCSR 限制。** 87.72% 的测试分子能拿到 PubChem 名称。检索不是瓶颈：命中时 BLEU-4 为 99.01。剩余 329 条未命中主要是 OCSR 错误；提升覆盖率需要更强的 OCSR，而不是更好的查询层。
2. **IUPAC 是检索，不是生成。** 项目报告诚实的端到端分数（65.98），没有用生成名称填补未命中。7B 兜底可把 BLEU-4 抬到 77.40，但新增结构正确名称为 **0**；该变体只记录在文档中，不随项目发布。
3. **官能团标签来自 OCSU 字典。** 分类头学习的是“图像 → 字典中的官能团”，不是自由化学命名，也不做官能团定位。不同渲染管线的图像可能需要域适应。
4. **分子描述继承 BioT5+ tokenizer artifact。** 多位数字会被拆开解码（`17-oxo` → `1 7 -oxo`）。输出按原样评分，不做后处理。
5. **不包含 Mol-VL / 7B / OCSR-free 路径。** 它只作为论文第二条路线在文档中说明。
6. **BERTScore 可选，且不可直接比较。** 论文未说明是否使用 baseline rescaling。

---

## 负结果（为什么这些路线没有进入项目）

| 路线 | 结果 | 结论 |
|---|---|---|
| Mol-VL-7B 作为 OCSR | 100 张图上 0.00 Acc_s、0.177 Tanimoto（同子集 MolScribe：0.81 / 0.929） | 不能替代 MolScribe |
| Mol-VL-7B 作为 IUPAC 兜底 | 0/329 名称通过 OPSIN 结构校验；BLEU-4 77.40 只是指标虚高 | 不发布 |
| MolScribe 在 25.9k 图上微调 | 第一个 epoch 验证集 `post_smiles` 0.254 | 灾难性遗忘 |
| MolSight 在 25.9k 图上微调 | Acc_s 69.58 → 65.12，空输出 157 → 248 | 灾难性遗忘 |
| MolScribe beam search | decoder 分支抛 `IndexError`；上游注明 "not supported yet" | 不可用 |
| 2B VLM 官能团 | 59.19 F1，对比字典 91.11 / 分类头 99.07 | 无竞争力 |

完整分析见 [`docs/negative_results.md`](docs/negative_results.md)。

---

## 引用

如果本项目对你有帮助，请引用 OCSU 论文及相关组件论文：

```bibtex
@article{fan2025ocsu,
  title   = {OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific Discovery},
  author  = {Fan, Siqi and Xie, Yuguang and Cai, Bowen and Xie, Ailin and Liu, Gaochao and Qiao, Mu and Xing, Jie and Nie, Zaiqing},
  journal = {arXiv preprint arXiv:2501.15415},
  year    = {2025}
}

@article{qian2023molscribe,
  title   = {MolScribe: Robust Molecular Structure Recognition with Image-to-Graph Generation},
  author  = {Qian, Yujie and Guo, Jiang and Tu, Zhengkai and Li, Zhen and Coley, Connor W. and Barzilay, Regina},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2023}
}

@article{pei2024biot5plus,
  title   = {BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning},
  author  = {Pei, Qizhi and Wu, Lijun and Gao, Kaiyuan and Liang, Xiaozhuan and Fang, Yin and Zhu, Jinhua and Xie, Shufang and Qin, Tao and Yan, Rui},
  journal = {arXiv preprint arXiv:2402.17810},
  year    = {2024}
}
```

## 致谢

MolScribe（MIT）、BioT5+（MIT）、OPSIN（MIT）、RDKit（BSD-3-Clause）、SciBERT（Apache-2.0）、OpenNMT-py（MIT），以及 OCSU 作者提供的 Vis-CheBI20 数据集和官能团字典（Apache-2.0）。完整归属与许可文本见 [`NOTICE`](NOTICE)。

## 许可

代码：**Apache-2.0**。第三方模型、数据与 vendored 源码保留各自许可；见 [`NOTICE`](NOTICE)。
