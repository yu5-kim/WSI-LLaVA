#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# llava는 WSI_LLAVA 루트에 있음 (v1_5 스크립트의 PYTHONPATH와 동일)
export PYTHONPATH="$(cd "${SCRIPT_DIR}/.." && pwd)"

# python scripts/merge_lora_weights.py \
#  --model-path /dataset/personal/yu5kim/WSI-LLaVA/checkpoints/wsi_llava_lora_penultimate_stage3only \
#  --model-base /dataset/data/raw/WSIBench/vicuna-7b-v1.5 \
#  --save-model-path /dataset/personal/yu5kim/WSI-LLaVA/merged_checkpoints/wsi_llava_lora_penultimate_stage3only

CUDA_VISIBLE_DEVICES=7 python "${SCRIPT_DIR}/merge_lora_weights.py" \
    --model-path /dataset/personal/yu5kim/WSI-Qwen/WSI-LLaVA/checkpoints_4gpu_3e_0421/wsi_llava_qwen3_8b_lora_last_stage3only \
    --model-base /dataset/model/Qwen3/Qwen3-8B \
    --save-model-path /dataset/personal/yu5kim/WSI-Qwen/WSI-LLaVA/merged_checkpoints/wsi_llava_qwen3_8b_lora_last_stage3only
