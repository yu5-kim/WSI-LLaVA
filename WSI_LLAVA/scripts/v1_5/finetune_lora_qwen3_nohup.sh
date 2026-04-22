#!/usr/bin/env bash
# 4B → 4B-Instruct-2507 → 8B LoRA 학습을 nohup으로 순차 실행합니다.
# (이전 단계 완료 후 다음 단계 시작, 동일 GPU 상속)
#
# 사용 예:
#   cd /dataset/personal/yu5kim/WSI-Qwen/WSI-LLaVA
#   CUDA_VISIBLE_DEVICES=5,6 bash WSI_LLAVA/scripts/v1_5/finetune_lora_qwen3_nohup.sh
#
# 로그: <레포루트>/logs/nohup/finetune_lora_qwen3_seq_YYYYMMDD_HHMMSS.log
# 진행 확인: tail -f 해당 로그

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/nohup"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/finetune_lora_qwen3_seq_${TS}.log"
PID_FILE="${LOG_FILE%.log}.pid"

cd "${REPO_ROOT}" || exit 1

export SCRIPT_DIR REPO_ROOT

nohup bash -c '
  set -euo pipefail
  echo "========== $(date -Is) 1/3: finetune_lora_qwen3.sh (4B) =========="
  bash "${SCRIPT_DIR}/finetune_lora_qwen3.sh"
  echo "========== $(date -Is) 2/3: finetune_lora_qwen3_4b_instruct_2507.sh (4B-Instruct-2507) =========="
  bash "${SCRIPT_DIR}/finetune_lora_qwen3_4b_instruct_2507.sh"
  echo "========== $(date -Is) 3/3: finetune_lora_qwen3_8b.sh (8B) =========="
  bash "${SCRIPT_DIR}/finetune_lora_qwen3_8b.sh"
  echo "========== $(date -Is) sequential run finished =========="
' >>"${LOG_FILE}" 2>&1 &

echo $! | tee "${PID_FILE}"

echo "log: ${LOG_FILE}"
echo "pid: $(cat "${PID_FILE}")"
echo "tail -f \"${LOG_FILE}\""
