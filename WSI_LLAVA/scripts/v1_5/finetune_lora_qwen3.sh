#!/usr/bin/env bash
set -euo pipefail

# Qwen3 기반 WSI-LLaVA LoRA 학습 예시.
#
# 중요:
#   이 스크립트는 "Qwen3 모델만"으로는 실행되지 않습니다.
#   아래 항목이 모두 필요합니다.
#   1) Qwen3 텍스트 백본 체크포인트
#   2) vision_tower(예: CLIP) 경로
#   3) WSI feature(.pt) 폴더
#   4) 학습 JSON 데이터
#   5) deepspeed 실행 환경

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
VISION_TOWER="${VISION_TOWER:-./LLaVA-main/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/path/to/WSI-Bench-train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/wsi_features_pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output/wsi-llava-qwen3-lora}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./WSI_LLAVA/scripts/zero3.json}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-./WSI_LLAVA}"

missing=0
for required_cmd in deepspeed; do
  if ! command -v "$required_cmd" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: $required_cmd"
    missing=1
  fi
done

for required_path in "$VISION_TOWER" "$DATA_PATH" "$IMAGE_FOLDER" "$(dirname "$OUTPUT_DIR")" "$DEEPSPEED_CONFIG"; do
  if [ ! -e "$required_path" ]; then
    echo "[ERROR] required path not found: $required_path"
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  echo ""
  echo "Please set these env vars to valid paths before running:"
  echo "  MODEL_NAME_OR_PATH, VISION_TOWER, DATA_PATH, IMAGE_FOLDER, OUTPUT_DIR, DEEPSPEED_CONFIG"
  exit 1
fi

PYTHONPATH="$PYTHONPATH_ROOT" WANDB_MODE=offline deepspeed --include localhost:0 --master_port 29507 \
  ./WSI_LLAVA/llava/train/train_mem.py \
  --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --llm_backbone qwen3 \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --version v1 \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --vision_tower "$VISION_TOWER" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 2 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb
