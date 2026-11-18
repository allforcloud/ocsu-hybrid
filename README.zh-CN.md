# OCSU-Hybrid

**OCSU 论文中 OCSR-based 路线的模块化、可运行实现。**

[English](README.md) | **简体中文**

OCSU-Hybrid 将二维分子结构图转换为 SMILES、官能团、IUPAC 名称和英文描述。它组合了 MolScribe、ConvNeXt-T、PubChem 与 BioT5+，只使用公开的专用模型，可在单张 8 GB 显卡上运行。

**Hugging Face 权重：** 官能团分类头托管在
[`allforcloud9527/ocsu-hybrid-fg-convnext-t`](https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t)。

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

全量 Vis-CheBI20 test split，指标定义与论文一致。下表把论文的两条路线与本项目的三个版本放在一起比较。

| 任务 | 指标 | Baseline（FG 微调） | TTA（三视图） | TTA + synonym | 论文 MolScribe | 论文 DoubleCheck | 论文 Mol-VL-2B | 论文 Mol-VL-7B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMILES | Acc_s | 84.00 | 85.42 | **85.42** | 80.50 | 82.40 | — | — |
| SMILES | Acc_c | 80.55 | 82.07 | **82.07** | 75.10 | 77.60 | — | — |
| SMILES | Tanimoto | 92.03 | 94.28 | **94.28** | 90.80 | 92.70 | — | — |
| 官能团 | F1 | **99.07** | **99.07** | **99.07** | 89.60 | 93.63 | 95.40 | 97.32 |
| IUPAC | BLEU-4 | 65.98 | 71.97 | **72.01** | 61.03 | 64.23 | 72.79 | **76.95** |
| IUPAC | ROUGE-1 | 85.39 | 88.44 | **88.50** | 82.73 | 84.45 | — | 81.88 |
| 分子描述 | BLEU-4 | 46.83 | 48.26 | **48.26** | 44.64 | 45.56 | 40.26 | 46.14 |
| 分子描述 | METEOR | 58.99 | 60.39 | **60.39** | 57.09 | 58.07 | — | 58.14 |

**如何理解 IUPAC 这一行。** 本项目从 PubChem **检索**名称，而不是生成名称。在找到记录的 90.56% 测试分子上，BLEU-4 为 **99.09**，**92.21%** 的名称与 ground truth 逐字一致；按全部 2,680 条计算，端到端逐字一致率为 83.5%。

Baseline 和 TTA 在四个头条指标（SMILES Acc_s、官能团 F1、IUPAC BLEU-4、描述 BLEU-4）上均达到或超过论文最佳 OCSR-based 列；TTA 的 SMILES Acc_s、Acc_c 与 Tanimoto 也超过两列 OCSR-based；官能团超过两种规模的 Mol-VL，IUPAC 超过两列 OCSR-based。

Baseline 只训练 27.8 M 的 FG 头，OCSR 单视图。TTA 使用 `base`、8% pad、+3° 旋转三个确定性视图；图像 FG 头选视图；选中视图无 PubChem 记录时回退其他视图。TTA + synonym 增加名称风格归一化。OCSR 单阶段 Acc_s 为 **86.70**；最终 pipeline 为 85.42，因为名称一致性兜底会把少量结构切到有 PubChem 记录的视图。

当前基准：`4b039ae`（2026-11-18）。

### 版本历史

| 版本 | 提交 | 日期 | 类型 | SMILES Acc_s | Acc_c | Tanimoto | IUPAC coverage | IUPAC exact | BLEU-4 | 描述 BLEU-4 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline（FG 微调） | `efa7f3f` | 2026-11-17 | 基线 | 84.00 | 80.55 | 92.03 | 87.72% | 80.00%（2,144） | 65.98 | 46.83 |
| TTA（三视图） | `6094586` | 2026-11-18 | 重大 | 85.42 | 82.07 | 94.28 | 90.56% | 82.54%（2,212） | 71.97 | 48.26 |
| TTA + synonym | `4b039ae` | 2026-11-18 | 次要 | 85.42 | 82.07 | 94.28 | 90.56% | **83.51%**（2,238） | **72.01** | 48.26 |

| 对比 | Δ Acc_s | Δ exact | Δ BLEU-4 | Δ 描述 |
|---|---:|---:|---:|---:|
| TTA 相对 Baseline | +1.42 | +2.54 | +5.99 | +1.43 |
| Synonym 相对 TTA | 0 | +0.97 | +0.04 | 0 |
| Synonym 相对 Baseline | +1.42 | +3.51 | +6.03 | +1.43 |

MolScribe 三视图调整就是 TTA 行（`6094586`）；2026-11-18 的第二次提交（`4b039ae`）是 synonym 归一化，不是 OCSR 训练微调。FG F1 三版均为 99.07。

复现三行：Baseline 省略 `--molscribe-views` 并加 `--no-normalize-iupac`；TTA 加三视图并加 `--no-normalize-iupac`；TTA + synonym 使用“复现论文对比结果”中的命令。

---

## 论文两条路线与本项目的技术对比

| | **论文：OCSR-based** | **论文：OCSR-free** | **本项目** |
|---|---|---|---|
| 核心思想 | `P(T\|I) = P(S\|I) · P(T\|S)`：先识别结构，再理解结构 | 单个 VLM 端到端学习 `P(T\|I)` | OCSR-based，官能团改用图像分类头 |
| 图像 → SMILES | DoubleCheck 或 MolScribe | Mol-VL 直接生成 | **MolScribe** 公开权重，多视图 TTA |
| 官能团 | RDKit + OCSU 字典 | Mol-VL 生成 | **ConvNeXt-T 图像分类头**（默认）；字典备用 |
| IUPAC | PubChem 检索 | Mol-VL 生成 | **PubChem 检索**，synonym 门控名称归一化 |
| 分子描述 | BioT5+ | Mol-VL 生成 | **BioT5+**，输入前先做 RDKit 规范化 |
| 名称 grounding | 数据库记录 | 无 | **数据库记录** |
| 训练 | OCSR 单独训练（DoubleCheck 4×A100×4 天；MolScribe 4×A100×1 天）；下游不训练 | 4×A100×10 天，50 epoch，全参数 SFT | **只训练 27.8 M 的官能团头**，RTX 4060 约 30 分钟；OCSR 与 BioT5+ 使用公开权重 |
| 推理栈 | OCSR + BioT5+ + PubChem | 单个 2B / 7B VLM | OCSR + 官能团头 + BioT5+ + PubChem |
| 主要优势 | IUPAC 名称是真实数据库记录；模块可替换 | 单模型，生成文本较强 | 真实记录 + 图像官能团头；小、可复现 |
| 主要权衡 | 名称是 PubChem 记录；OCSR 是唯一的学习式感知阶段 | 单模型覆盖所有任务；名称是生成而非检索 | OCSR + 检索；模块可独立替换 |

### 资源对比

| 项目 | **本项目** | 论文：OCSR-based | 论文：OCSR-free |
|---|---|---|---|
| 流水线参数量 | **约 0.38 B**（MolScribe 95.5 M + ConvNeXt-T 27.8 M + BioT5+ 252 M） | OCSR（MolScribe 级，约 95 M）+ BioT5+ 252 M | 2 B / 8.29 B（7B） |
| 需要训练的组件 | **仅官能团头，27.8 M** | OCSR backbone | 完整 VLM |
| 训练算力 | **1× RTX 4060，约 30 分钟** | DoubleCheck 4×A100×4 天；MolScribe 4×A100×1 天 | 4×A100×10 天 |
| 推理显存 | **RTX 4060 8 GB 可跑全部模块** | OCSR + 文本模型 | bf16 需 24 GB+；4-bit 约 6 GB，但更慢 |
| 权重磁盘占用 | **约 2.2 GB** | 未报告 | 7B 约 16.6 GB |
| 全量 test（四个任务） | **单卡 4060 约 2.3 小时** | 未报告 | 7B 4-bit、3 秒/图时，单任务约 2.75 小时 |
| 外部依赖 | IUPAC 需要 PubChem 网络 | IUPAC 需要 PubChem 网络 | 生成过程无外部依赖 |

8 GB 显存是实测结论：3,300 张 SMILES、3,269 张官能团、2,680 次 PubChem 查询和 3,300 条 BioT5+ 描述均在单张 RTX 4060 上完成。

---

## 功能范围

- MolScribe 图像 → SMILES，已 vendor 到仓库并包含兼容补丁；可选多视图 TTA，由图像官能团头选视图。
- ConvNeXt-T 140 类官能团分类头，使用 Vis-CheBI20 caption 训练。
- PubChem IUPAC 检索：canonical SMILES、InChIKey、Kekulé、CID、最大片段多级 fallback；只缓存命中结果。
- 数据集风格同义词归一化：仅当目标名称也是同一 InChIKey 的 PubChem synonym 时才切换。
- BioT5+ 分子描述，输入 SELFIES 前先做 RDKit canonicalisation。
- 可选 OPSIN 名称 → 结构校验。
- 四个任务的论文口径评测。

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

# 官能团头（托管在 Hugging Face）
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

ocsu eval --data-dir /path/to/vischebi20 --out results/full_test \
  --molscribe-views "base,pad:0.08,rot:3"
```

`--molscribe-views` 启用多视图 OCSR 测试时增强：图像官能团头选择最一致的视图，base 有效时保留。省略该参数即单视图推理，OCSR 约快 2.3 倍。`--no-normalize-iupac` 可关闭数据集风格同义词归一化。

先跑 smoke test（每个任务 20 条）：

```bash
ocsu eval --data-dir /path/to/vischebi20 --out results/smoke --limit 20
```

等价脚本：

```bash
python scripts/run_full_test.py --data-dir /path/to/vischebi20 \
  --out results/full_test --iupac-workers 4 \
  --molscribe-views "base,pad:0.08,rot:3"
```

runner 支持断点续跑：OCSR、官能团、IUPAC、描述结果会追加到 `results/full_test/preds/`，PubChem 中断不会重跑 GPU 阶段。续跑只按图像去重；如果更换 OCSR checkpoint、`--molscribe-views` 设置或官能团后端，请删除对应的预测文件（或整个 `preds/` 目录），避免混用不同配置的结果。

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
ocsu eval --data-dir /path/to/vischebi20 --out results/full_test \
  --iupac-workers 4 --molscribe-views "base,pad:0.08,rot:3"
```

RTX 4060 8 GB 上的期望结果：

```
SMILES      Acc_s 85.42  Acc_g 88.79  Acc_c 82.07  Tanimoto 94.28
FG          F1 99.07  P 99.31  R 99.00  exact 92.87
IUPAC       coverage 90.56%  BL-4 72.01  R-1 88.50  METEOR 88.44
Description BL-4 48.26  R-2 49.21  R-L 57.27  METEOR 60.39
```

这些数值已用本仓库 evaluator 对全量预测文件独立复算，与表格完全一致。指标定义、端到端与条件分口径、SciBERT tokenizer、论文 F1 表 P/R 列互换问题，以及 BioT5+ tokenizer artifact，见 [`docs/evaluation.md`](docs/evaluation.md)。

---

## 模型与数据

| 组件 | 来源 | 大小 | 许可 | 获取方式 |
|---|---|---|---:|---|
| MolScribe checkpoint | [`yujieq/MolScribe`](https://huggingface.co/yujieq/MolScribe) | 1.1 GB | MIT | Hugging Face 下载 |
| BioT5+ | [`QizhiPei/biot5-plus-base-chebi20`](https://huggingface.co/QizhiPei/biot5-plus-base-chebi20) | 0.96 GB | MIT | Hugging Face 下载 |
| 官能团 ConvNeXt-T 头 | [`allforcloud9527/ocsu-hybrid-fg-convnext-t`](https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t) | 107 MB | Apache-2.0 | Hugging Face 下载 |
| OPSIN 2.9.0 | [dan2097/opsin](https://github.com/dan2097/opsin) | 14 MB | MIT | 下载 |
| SciBERT 词表 | [allenai/scibert](https://github.com/allenai/scibert) | 223 KB | Apache-2.0 | 内置 |
| MolScribe 源码 | [thomas0809/MolScribe](https://github.com/thomas0809/MolScribe) | 14 MB | MIT | 内置（vendored） |
| OpenNMT-py 2.2.0 最小包 | [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) | 1 MB | MIT | 内置（vendored） |
| Vis-CheBI20 | [`PharMolix/Vis-CheBI20`](https://huggingface.co/datasets/PharMolix/Vis-CheBI20) | 3,300 张 test 图 | Apache-2.0 | 单独下载；内置一张示例图 |

官能团 checkpoint 托管在 Hugging Face。模型卡为 [`docs/fg_model_card.md`](docs/fg_model_card.md)，训练脚本为 `scripts/train_fg_classifier.py`。

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
| [`docs/fg_model_card.md`](docs/fg_model_card.md) | 官能团头的 Hugging Face 模型卡 |

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

默认情况下，MolScribe worker 使用当前 Python 解释器。如果 OCSR 栈在独立环境，可设置 `MOLSCRIBE_PY=/path/to/python` 或传入 `--molscribe-python`。

---

## 范围与保证

1. **每个 IUPAC 名称都是 PubChem 记录。** 没有记录时返回空名称，不做猜测。
2. **多视图 OCSR 由图像头选择。** MolScribe 运行三个确定性视图；ConvNeXt-T 官能团头保留与图像最一致的视图；选中视图无 PubChem 记录时再尝试其他视图；名称只在目标也是同一 InChIKey 的 PubChem synonym 时才做风格归一化。IUPAC 覆盖率达到 90.56%，命中时 BLEU-4 为 99.09。
3. **官能团标签遵循 140 类 OCSU 字典。** 分类头学习“图像 → 字典官能团”，不做官能团定位，也不生成自由化学文本。
4. **描述使用公开的 BioT5+ tokenizer。** 输出按原样评分，包括其多位数切分行为。

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
