---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
- synccraft
- block-dsl
- blockbench
- minecraft
- procedural-generation
- lora
- bf16
- rtx-pro-6000
- geometry-ast
---

# SyncCraft v3.0 — 4B/7B BF16 LoRA End-to-End Model

Fine-tuned full-precision `bfloat16` PEFT LoRA adapter for generating typed, kinematically valid **SyncCraft v3.0 `Block-DSL`** 3D articulated block assets.

## Model & Dataset Overview
- **Source Repository**: [https://github.com/BaiTian6641/SyncCraft](https://github.com/BaiTian6641/SyncCraft)
- **Companion Public Dataset**: [https://huggingface.co/datasets/BaiTian6641/synccraft-v3-blockdsl-dataset](https://huggingface.co/datasets/BaiTian6641/synccraft-v3-blockdsl-dataset)
- **Dataset Manifest SHA-256**: `1b9ad1acc489f4379eabe571bbe69c01f6250ebee7c7d71b73d287bec5d69390`
- **Base Architecture**: `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Fine-Tuning Technique**: Full-Precision `bfloat16` PEFT LoRA on Google Colab NVIDIA RTX PRO 6000 (`r=32`, `alpha=64`, `quantization=none`)
- **License**: Apache-2.0
- **Target Export Profiles**: Blockbench 5.0 (`bbmodel_5`) / Java Modern (`1.21.11+`)

## Training Metrics
- **Final Training Loss**: `7.4832`
- **Total Generated Tokens**: `5335`
- **Supervised Training Samples**: `16` SAT-verified `Block-DSL` records

## Credits, Attributions & Open-Source Notices
- **SyncCraft v3.0 Architecture, Compiler, SAT Auditor & Training Pipeline**: Copyright 2026 **SyncCraft Contributors** ([BaiTian6641/SyncCraft](https://github.com/BaiTian6641/SyncCraft)), licensed under **Apache-2.0**.
- **Base Pretrained Model**: `Qwen/Qwen2.5-Coder-7B-Instruct` by the **Qwen Team** (verify exact revision license per `configs/model_candidates.yaml`).
- **Parameter-Efficient Fine-Tuning (LoRA)**: Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (`peft` / Hugging Face).
- **Perceptual Color Space (Oklab / OKLCH)**: Palette conversions follow **Björn Ottosson** (2020), [https://bottosson.github.io/posts/oklab/](https://bottosson.github.io/posts/oklab/).
- **Compatibility Disclaimer**: Trained exclusively on original procedural `Block-DSL` geometry verified by SyncCraft's 15-axis SAT auditor. Zero Mojang / Minecraft game assets or proprietary third-party models/textures were used.

## Citation

```bibtex
@misc{synccraft2026model,
  title        = {SyncCraft v3.0: Full-Precision BF16 LoRA for Typed Kinematic Block-DSL Generation},
  author       = {BaiTian6641 and SyncCraft Contributors},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/BaiTian6641/synccraft-v3-bf16-lora},
  note         = {Dataset: https://huggingface.co/datasets/BaiTian6641/synccraft-v3-blockdsl-dataset | Code: https://github.com/BaiTian6641/SyncCraft}
}
```
