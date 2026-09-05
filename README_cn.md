<h1 align="center">SANMS: 面向视觉检索中主体检测的结构感知后NMS精修</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/License-Apache-yellow.svg" alt="License: Apache" />
  <a href="tests/test_sanms.py"> <img src="https://img.shields.io/badge/tests-25%2F25-green.svg" alt="License: Apache" /> </a>
</p>

[English](./README.md) | 简体中文

SANMS（Structure-Aware Non-Maximum Suppression）是一个 **NMS 后精修模块**，专为 **以图搜图中的主体检测** 设计。它解决了一个 IoU 类 NMS 无法检测的几何包含盲区：当检测器对同一物体同时输出宽松裁剪框（有利于检索）和紧凑裁剪框（不利于检索但置信度更高）时，由于两者的 IoU 低于抑制阈值，NMS 会保留错误的框。

**SANMS 不是 NMS 的替代品。** 它在任何 NMS 变体（Greedy、Soft、DIoU、Matrix）之后作为轻量精修步骤使用，仅依赖一个超参数（alpha，面积比例门控）。

---

## 目录

- [快速开始](#快速开始)
- [算法概述](#算法概述)
- [安装](#安装)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [实验](#实验)
- [论文](#论文)
- [引用](#引用)
- [许可证](#许可证)

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/TommysLee/SANMS.git
cd SANMS

# 安装核心依赖（仅需 numpy + scipy）
pip install -r requirements.txt

# 运行示例（无需数据集）
python examples/quickstart.py

# 运行单元测试
python -m pytest tests/ -v
```

**最小代码示例：**

```python
import numpy as np
from sanms import greedy_nms, sanms

# 模拟检测器输出：包含框 + 被包含框
boxes  = np.array([[50, 50, 250, 250], [90, 90, 160, 160]])
scores = np.array([0.65, 0.92])

# 步骤1：标准 NMS（因 IoU 低，无法抑制被包含框）
nms_boxes, nms_scores = greedy_nms(boxes, scores, iou_thresh=0.5)

# 步骤2：SANMS 精修（检测几何包含关系，抑制内部框）
final_boxes, final_scores = sanms(nms_boxes, nms_scores, alpha=0.75)
```

---

## 算法概述

SANMS 解决 IoU 类 NMS 的一个盲区：**几何包含关系**。

当检测器对同一物体同时输出宽松框（包含框）和紧凑框（被包含框）时，两者的 IoU 可能低至 0.3，远低于典型 NMS 抑制阈值（0.45-0.5）。NMS 因此保留两个框。被包含框通常置信度更高，会被选为最终检测结果，产生过度紧凑的裁剪区域，损害下游检索质量。

```
Container box (loose crop, score=0.65)     Contained box (tight crop, score=0.92)
  ┌──────────────────────────────┐            ┌──────────────────────────────┐
  │                              │            │   ┌──────────┐                │
  │                              │            │   │          │                │
  │                              │            │   │  IoU≈0.3 │                │
  │                              │            │   │          │                │
  │                              │            │   └──────────┘                │
  └──────────────────────────────┘            └──────────────────────────────┘
  IoU < 0.5: NMS keeps both                    is_contained = True: SANMS fixes this
```

**SANMS 算法（3 步）：**

1. **排序**：按面积降序排列（最大框优先处理）
2. **包含检测**：对每个框，通过直接坐标比较（非 IoU）检查是否被已保留的框几何包含
3. **面积比例门控**：若被包含且 `area(内框)/area(外框) < alpha`，则抑制内框，并将其分数继承给外框（分数继承机制）

**核心特性：**
- **免训练**：无需学习参数
- **单一超参数**：`alpha`（面积比例门控，默认 0.75）
- **后 NMS**：在任何 NMS 变体之后应用
- **最坏 O(N^2)，平均 O(N)**
- **运行时**：仅为 Soft-NMS 的 10-15%

---

## 安装

```bash
# 方式一：从源码安装（推荐，含实验代码）
git clone https://github.com/TommysLee/SANMS.git
cd SANMS
pip install -r requirements.txt          # 核心：numpy, scipy
pip install -r requirements-optional.txt # 可选：torch, faiss, PIL, pycocotools

# 方式二：pip 安装（仅包）
pip install git+https://github.com/TommysLee/SANMS.git

# 方式三：开发模式
pip install -e ".[all]"
```

### 依赖说明

| 包 | 必需 | 用途 |
|---|------|------|
| numpy | 是 | 核心数组运算 |
| scipy | 是 | 数值工具 |
| torch | 否 | ResNet-50 特征提取（完整流程） |
| faiss-cpu | 否 | 快速近邻搜索 |
| Pillow | 否 | 图像加载 |
| pycocotools | 否 | COCO 格式评测 |

核心算法和合成实验仅需 numpy + scipy。torch/faiss/PIL 仅在完整检索流程中使用，为可选依赖。

---

## 项目结构

```
SANMS/
|-- sanms/                          # 核心 Python 包
|   |-- __init__.py                 # 公共 API 导出
|   |-- sanms.py                    # SANMS 算法（核心贡献）
|   |-- box_ops.py                  # IoU、DIoU、包含检测、框投票
|   |-- nms_baselines.py            # Greedy NMS、Soft-NMS、DIoU-NMS、Matrix NMS
|   |-- metrics.py                  # COCO mAP、Recall@K、Oxford mAP、Top-1、ECE
|   |-- feature_extractor.py        # ResNet-50 特征提取（可选）
|   `-- retriever.py                # 基于 Faiss 的检索引擎（可选）
|-- experiments/                    # 实验脚本
|   |-- pipeline.py                 # 端到端流程编排
|   |-- run_all.py                  # 复现全部表格（合成数据）
|   |-- run_runtime.py              # 运行时基准测试（Table 12）
|   `-- results/                    # 预计算结果
|       |-- all_results_v2.json     # Tables 1-11, 13 数据
|       `-- runtime_with_std.json   # Table 12 数据（100次试验）
|-- tests/
|   `-- test_sanms.py               # 25 个单元测试（无外部依赖）
|-- examples/
|   `-- quickstart.py               # 最小示例（仅需 numpy）
|-- paper/
|   |-- SANMS Resolving Geometric Containment in Non-Maximum Suppression for Visual Retrieval_en.pdf  # 论文（英文）
|-- pyproject.toml                  # 现代打包配置
|-- setup.py                        # 向后兼容 setup
|-- requirements.txt                # 核心依赖
|-- requirements-optional.txt       # 可选依赖
|-- LICENSE
`-- .gitignore
```

---

## 使用方法

### 运行示例

```bash
python examples/quickstart.py
```

### 运行单元测试

```bash
python -m pytest tests/ -v
# 或直接运行：
python tests/test_sanms.py
```

### 复现实验

所有实验使用受控合成数据，建模包含场景。无需外部数据集。

```bash
# 复现 Tables 1-11, 13（约 2-5 分钟）
python -m experiments.run_all

# 复现 Table 12: 运行时基准（约 2 分钟）
python -m experiments.run_runtime
```

结果保存到 `experiments/results/`。

### 在你的代码中使用

```python
import numpy as np
from sanms import greedy_nms, soft_nms, diou_nms, matrix_nms, sanms

# 检测器输出
boxes = np.array([...], dtype=np.float64)   # (N, 4): [x1, y1, x2, y2]
scores = np.array([...], dtype=np.float64)   # (N,)

# 步骤1：运行任意 NMS 变体
nms_boxes, nms_scores = greedy_nms(boxes, scores, iou_thresh=0.5)

# 步骤2：应用 SANMS 精修
final_boxes, final_scores = sanms(nms_boxes, nms_scores, alpha=0.75)
```

---

## 实验

### 预计算结果

预计算实验结果已包含在 `experiments/results/` 中：

| 文件 | 内容 | 对应表格 |
|------|------|----------|
| `all_results_v2.json` | COCO mAP、Recall@K、Oxford mAP、Top-1、ECE、消融 | Tables 1-11, 13 |
| `runtime_with_std.json` | 运行时（mean +/- std, 100次试验） | Table 12 |

### 运行时基准

在 Intel Core i7-8550U 上测量（每组 100 次试验）：

| 方法 | N=50 | N=100 | N=300 |
|------|------|-------|-------|
| Greedy NMS | 3.4 +/- 1.0 ms | 6.3 +/- 1.7 ms | 15.5 +/- 2.7 ms |
| Soft-NMS | 6.6 +/- 1.7 ms | 16.6 +/- 6.9 ms | 79.2 +/- 14.2 ms |
| DIoU-NMS | 6.5 +/- 1.9 ms | 12.9 +/- 2.6 ms | 30.0 +/- 4.6 ms |
| Matrix NMS | 0.5 +/- 0.2 ms | 1.6 +/- 0.5 ms | 9.7 +/- 1.9 ms |
| Box Voting | 4.3 +/- 2.1 ms | 7.8 +/- 1.7 ms | 21.9 +/- 4.2 ms |
| **SANMS** | **0.9 +/- 0.3 ms** | **2.4 +/- 0.7 ms** | **10.8 +/- 1.9 ms** |

SANMS 仅增加 Soft-NMS 10-15% 的开销。

---

## API 参考

### `sanms.sanms(boxes, scores, alpha=0.75, eps=1e-3, sort_criterion="area")`

核心 SANMS 算法。

**参数：**
- `boxes`: (N, 4) numpy 数组，格式 [x1, y1, x2, y2]
- `scores`: (N,) numpy 数组，置信度分数
- `alpha`: float，面积比例门控（默认 0.75）。当被包含框面积/外框面积 < alpha 时被抑制。
- `eps`: float，几何容差（像素，默认 1e-3）
- `sort_criterion`: str，排序方式。可选 `"area"`、`"diagonal"`、`"perimeter"`、`"confidence"`（默认 `"area"`）

**返回：** `(kept_boxes, kept_scores)`，按置信度降序排列。

### NMS 基线

- `greedy_nms(boxes, scores, iou_thresh=0.5)`
- `soft_nms(boxes, scores, sigma=0.5, mode="gaussian")`
- `diou_nms(boxes, scores, diou_thresh=0.5)`
- `matrix_nms(boxes, scores, iou_thresh=0.5, sigma=0.5)`

### 框操作

- `compute_area(boxes)` -> 面积数组
- `compute_iou(boxes_a, boxes_b)` -> IoU 矩阵
- `compute_diou(boxes_a, boxes_b)` -> DIoU 矩阵
- `is_contained(box_inner, box_outer, eps)` -> bool
- `box_voting(boxes, scores, iou_thresh)` -> (投票框, 投票分数)

### 评测指标

- `coco_map(...)` -> {mAP, AP50, AP75, per_class}
- `recall_at_k(...)` -> {R@1, R@10, R@100}
- `oxford_map(...)` -> mAP
- `top1_accuracy(...)` -> Top-1 准确率
- `expected_calibration_error(...)` -> ECE

---

## 论文

完整论文包含在 [`paper/`](paper/) 目录中：

- **英文版**：`SANMS Resolving Geometric Containment in Non-Maximum Suppression for Visual Retrieval_en.pdf`

### 核心发现

- SANMS 通过抑制 NMS 遗漏的几何包含框来提升检测 mAP
- 改进可迁移到检索任务：应用 SANMS 后 Recall@K 和 Top-1 准确率均有提升
- SANMS 仅有单一超参数（alpha=0.75），且在较宽范围内鲁棒
- 算法免训练，计算开销极低（Soft-NMS 的 10-15%）

### 局限性

本仓库中的实验使用受控合成数据，建模包含场景。在真实基准（MS COCO、Stanford Online Products、Oxford Buildings）上的验证仍是后续工作。

---

## 引用

如在研究中使用本代码，请引用：

```bibtex
@misc{sanms2026,
    author = {Li Zongjin},
    title = {SANMS: Structure-Aware Post-NMS Refinement for Subject Detection in Visual Retrieval},
    year = {2026},
    url = {https://github.com/TommysLee/SANMS},
    orcid = {0009-0009-9121-0268}
}
```

---

## 许可证

[Apache-2.0](LICENSE)

## 作者

**Li Zongjin**

- ORCID: [0009-0009-9121-0268](https://orcid.org/0009-0009-9121-0268)
- Email: li_zongjin@alumni.hust.edu.cn
- GitHub: [TommysLee](https://github.com/TommysLee)

