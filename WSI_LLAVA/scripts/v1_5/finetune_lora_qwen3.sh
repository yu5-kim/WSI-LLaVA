#!/usr/bin/env bash
set -euo pipefail

# Qwen3 기반 WSI-LLaVA LoRA 학습 예시
# 사용 전 아래 경로를 환경에 맞게 수정하세요.

PYTHONPATH=./WSI_LLAVA WANDB_MODE=offline deepspeed --include localhost:0 --master_port 29507 \
  ./WSI_LLAVA/llava/train/train_mem.py \
  --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
  --deepspeed ./WSI_LLAVA/scripts/zero3.json \
  --llm_backbone qwen3 \
  --model_name_or_path Qwen/Qwen3-4B \
  --version v1 \
  --data_path /path/to/WSI-Bench-train.json \
  --image_folder /path/to/wsi_features_pt \
  --vision_tower ./LLaVA-main/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir /path/to/output/wsi-llava-qwen3-lora \
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
