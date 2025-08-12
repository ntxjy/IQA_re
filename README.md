# MVQA Hub (2024–2025, non-text, 6×RGB + 6×Depth)

本仓库聚合 **2024–2025 顶会/顶刊** 与本项目密切相关的实现，均为**真克隆**（非 submodule），并配套训练/可视化/评测工具。

- `iqa_methods/`：图像/多视图质量评估（IQA）
- `backbones/`：视觉骨架/低层复原主干（ViT/Mamba/CNN/KAN）
- `other_methods/`：深度/立体/光场/超分/去噪/去雾
- `tools/`：训练可视化、画图、特征可视化、数据/MOS 预处理、排序损失与评测协议
- `eval_compare/`：IQA 指标库整合 + 一键对比/消融脚本

## IQA 统一口径（摘要）
- 主任务：组内 ΔMOS（`MOS_ref − MOS_i`）；可选组内 z-score 仅用于训练稳性（汇报用原始 MOS）
- 采样：Group-balanced（“参考 + 若干失真”，并显式抽到跨场景 **同参数编号** 样本）
- 损失：`1 − PLCC + L1 + 组内 list-wise 排序 (+ L_delta)`；做“同参数一致性”再加 `L_param_align`
- 评测：验证/测试统一用 **4/5 参数 Logistic** 拟合后再算 PLCC/RMSE/GoF；SRCC/KRCC **不做映射**

---

## A. IQA Methods
| 名称 | 做什么 | Venue/Year | 论文 | 上游代码 | 本地路径 |
|---|---|---|---|---|---|
| **LoDa** | ViT 高效适配 + 局部增强的 NR-IQA | CVPR 2024 | CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.pdf | https://github.com/NeosXu/LoDa | `iqa_methods/LoDa` |
| **CrossScore** | 多视图 cross-attention 相互参照 | ECCV 2024 | ECCV: https://eccv.ecva.net/virtual/2024/poster/1491/ | https://github.com/ActiveVisionLab/CrossScore | `iqa_methods/CrossScore` |
| **QCN** | 几何顺序学习的 BIQA | CVPR 2024 | CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Shin_Blind_Image_Quality_Assessment_Based_on_Geometric_Order_Learning_CVPR_2024_paper.pdf | https://github.com/nhshin-mcl/QCN | `iqa_methods/QCN` |
| **SaTQA** | 监督对比 + Transformer 的 NR-IQA | AAAI 2024 | AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/28285 | https://github.com/Srache/SaTQA | `iqa_methods/SaTQA` |

> 可选（若 `SKIP_OPTIONAL=0`）：DepictQA（ECCV’24）、FineVQ（CVPR’25, VQA）、SAMA（AAAI’24）

---

## B. Backbones
| 名称 | 做什么 | Venue/Year | 论文/说明 | 上游代码 | 本地路径 |
|---|---|---|---|---|---|
| **Fast-iTPN** | 分层 ViT + Token Migration | TPAMI 2024 | 见仓库 | https://github.com/sunsmarterjie/iTPN | `backbones/iTPN` |
| **VMamba** | 视觉 SSM（线性复杂度） | NeurIPS 2024 | Proc | https://github.com/MzeroMiko/VMamba | `backbones/VMamba` |
| **ViM** | Vision Mamba 变体 | ICML 2024 | 见仓库 | https://github.com/hustvl/Vim | `backbones/Vim` |
| **2DMamba** | 2D 选择性扫描 Mamba | CVPR 2025 | 见仓库 | https://github.com/AtlasAnalyticsLab/2DMamba | `backbones/2DMamba` |
| **MambaIR** | 面向复原的 Mamba 框架 | ECCV 2024 | 见仓库 | https://github.com/csguoh/MambaIR | `backbones/MambaIR` |
| **EVSSM** | VSSM 去模糊骨干/框架 | CVPR 2025 | 见仓库 | https://github.com/kkkls/EVSSM | `backbones/EVSSM` |
| **MS-VMamba** | VMamba 多尺度改进 | NeurIPS 2024 | 见仓库 | https://github.com/YuHengsss/MSVMamba | `backbones/MSVMamba` |
| **ConvNeXt-V2** | 强力 CNN 基线组件 | — | 见仓库 | https://github.com/facebookresearch/ConvNeXt-V2 | `backbones/ConvNeXt-V2` |
| **timm** | 预训练模型/模块库 | — | 见仓库 | https://github.com/rwightman/pytorch-image-models | `backbones/timm` |
| **pykan** | KAN 模块 | ICLR 2024 | 见仓库 | https://github.com/KindXiaoming/pykan | `backbones/pykan` |

---

## C. Other Methods（深度/立体、光场、SR、去噪、去雾）
- **Depth/Stereo**：`other_methods/depth_stereo/*`（Depth-Anything-V2、Video-Depth-Anything、DAC、DepthPro、Selective-Stereo、TC-Stereo、BiDAStereo、Stereo-Anywhere、Murre、Metric3D、FoundationStereo）
- **Light Field**：`other_methods/light_field/*`（NeLF-Pro、PDistgNet、PSWPP、BasicLFSR）
- **Super-Resolution**：`other_methods/super_resolution/*`（PFT-SR、PiSA-SR、PASD、StableVSR、InvSR）
- **Denoising**：`other_methods/denoising/*`（SplitterNet、RDDM、DualDn、TAP）
- **Weather Removal**：`other_methods/weather_removal/*`（Histoformer、MWFormer、MetaWeather、DCL、DarkIR）

各条目的论文链接与上游地址，见子目录 `SOURCE_URL.txt` 或上游仓库 README。

---

## D. Tools（训练/可视化/预处理）
| 名称 | 类别 | 说明 | 上游代码 | 本地路径 |
|---|---|---|---|---|
| **pytorch-grad-cam** | 特征可视化 | 支持 CNN/ViT 的 CAM 可视 | https://github.com/jacobgil/pytorch-grad-cam | `tools/vis/pytorch-grad-cam` |
| **torch-cam** | 特征可视化 | 钩子机制 CAM 工具 | https://github.com/frgfm/torch-cam | `tools/vis/torch-cam` |
| **tensorboardX** | 训练可视化 | 标量/图像/网络/Embedding | https://github.com/lanpa/tensorboardX | `tools/train/tensorboardX` |
| **Albumentations** | 数据增强 | 高性能 CV 增强 | https://github.com/albumentations-team/albumentations | `tools/data/albumentations` |
| **local_utils** | 本仓自带 | `mos_utils.py`（z-score/ΔMOS/4/5PL/PLCC/SRCC/KRCC）、`plotting_utils.py`（训练曲线/IQA散点+拟合）、`rank_losses.py`（ListNet/Pairwise）、`data_utils.py`（GroupBalancedBatchSampler）、`eval_protocol.py`（统一评测） | — | `tools/local_utils/` |

---

## E. eval_compare（对比/消融）
| 名称 | 说明 | 上游代码 | 本地路径 |
|---|---|---|---|
| **PIQ** | PyTorch Image Quality：SSIM/PSNR/FSIM/LPIPS 等 | https://github.com/photosynthesis-team/piq | `eval_compare/piq` |
| **LPIPS** | Learned Perceptual Image Patch Similarity | https://github.com/richzhang/PerceptualSimilarity | `eval_compare/LPIPS` |
| **pyiqa** | IQA 工具箱（FR/NR） | https://github.com/chaofengc/IQA-PyTorch | `eval_compare/pyiqa` |
| **run_eval.py** | 一键读取 `pred.csv`/`gt.csv` → 4/5PL 标定 → PLCC/SRCC/KRCC/RMSE | （本仓脚本） | `eval_compare/run_eval.py` |

---

## 快速检索
- 训练入口：`rg -n -e 'def main\(|if __name__ == .__main__|argparse'`
- 数据/采样：`rg -n -e 'class .*Dataset|DataLoader|__getitem__|__len__'`
- 排序/相关性/PLCC：`rg -n -e 'list.?wise|rank|PLCC|SRCC|KRCC|logistic'`
- 深度/立体：`rg -n -e 'stereo|cost volume|disparit|depth|mvs' other_methods/depth_stereo`
- 工具函数：`rg -n 'tools/local_utils'`
