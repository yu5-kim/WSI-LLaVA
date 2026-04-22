export MASTER_PORT=29501
export PYTHONPATH=./WSI_LLAVA
export CUDA_VISIBLE_DEVICES=6

# 리포 루트(WSI-LLaVA)에서 실행 권장: ./WSI_LLAVA/scripts/wsi-vqa.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_FOLDER="${IMAGE_FOLDER:-/dataset/data/slide_spatial_features/ps512/conch_v1_5_titan/TCGA_yu5kim_WSI_LLaVA}"
QUESTION_FILE="${QUESTION_FILE:-/dataset/personal/yu5kim/WSI-LLaVA/WSI-Bench/WSI-Bench-Report-only_TCGA_yu5kim_WSI_LLaVA_last.jsonl}"

# MODEL_PATH 기본(checkpoints/wsi_llava_stage2_projection_penultimate)과 맞춘 출력명
ANSWERS_FILE="${ANSWERS_FILE:-WSI-Bench/WSI-Bench-Report-only-wsi_llava_stage2_projection_penultimate.jsonl}"


IMAGE_FOLDER=/dataset/data/slide_spatial_features/ps512/conch_v1_5_titan/TCGA_yu5kim_WSI_LLaVA


# [대조군] 예: 아래 세 줄을 쓰고, 맨 아래 python에서 (1)을 막고 (2)만 쓴다.
# MODEL_PATH="/dataset/data/raw/WSIBench/wsi-llava-v1.5-7b-14"
# MODEL_PATH="/dataset/data/raw/WSIBench/wsi-llava-v1.5-7b-e1"
# MODEL_PATH="/dataset/personal/yu5kim/WSI-LLaVA/checkpoints/wsi_llava_lora_penultimate_stage3only/checkpoint-257"
# MODEL_PATH="/dataset/personal/yu5kim/WSI-LLaVA/checkpoints/wsi_llava_lora_penultimate_stage3only/checkpoint-515"

# QUESTION_FILE=/dataset/personal/yu5kim/WSI-LLaVA/WSI-Bench/WSI-Bench-Report-only_TCGA_yu5kim_WSI_LLaVA_penultimate.jsonl


# MODEL_PATH=/dataset/personal/yu5kim/WSI-LLaVA/merged_checkpoints/wsi_llava_lora_penultimate_stage3only_1e \
# MODEL_PATH=/dataset/personal/yu5kim/WSI-LLaVA/merged_checkpoints/wsi_llava_lora_penultimate_after_stage2_stage3_1e \
# MODEL_PATH=/dataset/personal/yu5kim/WSI-LLaVA/merged_checkpoints_1gpu/wsi_llava_lora_last_after_stage2_stage3_1e
# MODEL_PATH=/dataset/personal/yu5kim/WSI-LLaVA/merged_checkpoints_1gpu/wsi_llava_lora_last_stage3only_1e

MODEL_PATH=/dataset/personal/yu5kim/WSI-Qwen/WSI-LLaVA/merged_checkpoints/wsi_llava_qwen3_4b_instruct_2507_lora_last_stage3only_3e



# ANSWERS_FILE="WSI-Bench/WSI-Bench-Report-only-wsi-llava-v1.5-7b-14.jsonl"
# ANSWERS_FILE="WSI-Bench/WSI-Bench-Report-only-wsi-llava-v1.5-7b-e1.jsonl"
# ANSWERS_FILE="WSI-Bench/WSI-Bench-Report-only-wsi_llava_lora_penultimate_stage3only_checkpoint-515.jsonl"
# ANSWERS_FILE="WSI-Bench/WSI-Bench-Report-only-wsi_llava_lora_penultimate_stage3only_1e.jsonl"
# ANSWERS_FILE="WSI-Bench/WSI-Bench-Report-only-1gpu-wsi_llava_lora_last_after_stage2_stage3_1e.jsonl"
ANSWERS_FILE="WSI-Bench/Results/WSI-Bench-Report-only-Qwen3_4B_Instruct_2507_lora_last_stage3only_3e.jsonl"




# 한 번만 쓸 비율 (기본 1.0 = 전체 패치). 예: PATCH_SAMPLE_RATIO=0.05 ./WSI_LLAVA/scripts/wsi-vqa.sh
PATCH_SAMPLE_RATIO="${PATCH_SAMPLE_RATIO:-1.0}"

echo "patch_sample_ratio=${PATCH_SAMPLE_RATIO}"
echo "answers: ${ANSWERS_FILE}"

cd "${REPO_ROOT}" || exit 1

VQA_EXIT=0
if [[ "${SKIP_VQA:-0}" != "1" ]]; then
    CUDA_LAUNCH_BLOCKING=1 python WSI_LLAVA/llava/eval/model_vqa.py \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --question-file "${QUESTION_FILE}" \
        --answers-file "${ANSWERS_FILE}" \
        --conv-mode llava_v1 \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --patch-sample-ratio "${PATCH_SAMPLE_RATIO}" || VQA_EXIT=$?
else
    echo "SKIP_VQA=1: model_vqa.py 건너뜀 (기존 ANSWERS_FILE로 메트릭만 계산)"
fi

# VQA 출력(jsonl)에 대해 NLP 메트릭 평가 → TSV 저장
# 기본: ANSWERS_FILE과 같은 stem (예: ...Report-only-1gpu-foo.jsonl → ...Report-only-1gpu-foo.tsv)
# METRICS_USE_MODEL_PATH=1 이면: .../모델/checkpoint-N → 모델_checkpoint-N.tsv
METRICS_OUT_DIR="${METRICS_OUT_DIR:-WSI-Bench/NLP_metric_results}"
if [[ "${METRICS_USE_MODEL_PATH:-0}" == "1" ]]; then
    MODEL_PATH_NORM="${MODEL_PATH%/}"
    METRICS_STEM="$(basename "${MODEL_PATH_NORM}")"
    if [[ "${METRICS_STEM}" == checkpoint-* ]]; then
        METRICS_STEM="$(basename "$(dirname "${MODEL_PATH_NORM}")")_${METRICS_STEM}"
    fi
else
    METRICS_STEM="$(basename "${ANSWERS_FILE%.jsonl}")"
fi
METRICS_TSV="${METRICS_OUT_DIR}/${METRICS_STEM}.tsv"
mkdir -p "${METRICS_OUT_DIR}"

if [[ "${VQA_EXIT}" -ne 0 ]]; then
    echo "VQA가 실패했거나 중단되어 NLP 메트릭을 건너뜁니다 (exit=${VQA_EXIT})."
    echo "이미 jsonl이 있으면: SKIP_VQA=1 ./WSI_LLAVA/scripts/wsi-vqa.sh"
    exit "${VQA_EXIT}"
fi
if [[ ! -f "${ANSWERS_FILE}" ]]; then
    echo "ANSWERS_FILE이 없습니다: ${ANSWERS_FILE}"
    exit 1
fi

echo "NLP metrics TSV: ${METRICS_TSV}"
python NLP_Metric.py \
    --input "${ANSWERS_FILE}" \
    --output "${METRICS_TSV}"


# export MASTER_PORT=29501
# export PYTHONPATH=./WSI_LLaVA
# export CUDA_VISIBLE_DEVICES=0
# ./miniconda3/envs/llava/bin/python ./WSI_LLaVA/llava/eval/model_vqa.py \
#     --model-path "" \
#     --image-folder  \
#     --question-file .jsonl \
#     --answers-file .jsonl \
#     --conv-mode llava_v1 \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0.2 \
#     # --top_p 0.9 \
#     # --num_beams 4 
