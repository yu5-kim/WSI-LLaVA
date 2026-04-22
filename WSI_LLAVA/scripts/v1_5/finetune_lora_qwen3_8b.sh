#!/usr/bin/env bash
# Stage3만 (LoRA). Stage2 mm_projector.bin 미로드. LLM: Qwen3-8B (WSI-LLaVA Qwen 분기).
# 데이터·설정: finetune_lora_qwen3.sh 와 동일.
# 레포 루트: bash WSI_LLAVA/scripts/v1_5/finetune_lora_qwen3_8b.sh
#
# GPU 지정 예)
#   CUDA_VISIBLE_DEVICES=0 bash WSI_LLAVA/scripts/v1_5/finetune_lora_qwen3_8b.sh
#   CUDA_VISIBLE_DEVICES=4 MASTER_PORT=29518 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/WSI_LLAVA"
export WANDB_MODE=offline
REPORT_TO="${REPORT_TO:-wandb}"

IMAGE_FOLDER="/dataset/data/slide_spatial_features/ps512/conch_v1_5_titan/TCGA_yu5kim_WSI_LLaVA"
DATA_PATH="/dataset/personal/yu5kim/WSI-LLaVA/WSI-Bench/WSI-Bench-train_filtered_llava576_paths_last.json"
OUTPUT_DIR="${REPO_ROOT}/checkpoints_4gpu_3e/wsi_llava_qwen3_8b_lora_last_stage3only"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
# NUM_TRAIN_EPOCHS=0.1
SAVE_TOTAL_LIMIT=1
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/dataset/model/Qwen3/Qwen3-8B}"
VISION_TOWER="${VISION_TOWER:-/dataset/data/raw/WSIBench/clip-vit-large-patch14-336}"

cd "${REPO_ROOT}" || exit 1

deepspeed --master_port "${MASTER_PORT:-29517}" "${REPO_ROOT}/WSI_LLAVA/llava/train/train_mem.py" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed "${REPO_ROOT}/WSI_LLAVA/scripts/zero3.json" \
    --llm_backbone qwen3 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}"
