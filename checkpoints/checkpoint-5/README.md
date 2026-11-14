---
license: apache-2.0
tags:
- synccraft
- blockbench
- minecraft
- procedural-generation
- lora
- bf16
- rtx-pro-6000
- geometry-ast
---

# SyncCraft v3.0 - 4B/7B BF16 LoRA End-to-End Model

## Model Overview
- **Base Architecture**: `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Fine-Tuning Technique**: Full BF16 LoRA on Colab RTX PRO 6000 (r=32, alpha=64, quantization=none)
- **Dataset Manifest SHA-256**: `a1e301fc94e4932f8c1902baaed10e40be02823ac19151933fa57df83825299b`
- **License**: Apache-2.0
- **Target Profile**: Blockbench 5.0 / Java Modern 1.21.11+

## Training Metrics
- Final Training Loss: `5.8185`
- Total Generated Tokens: `730`
- Supervised Samples: `16` verified Block-DSL records
