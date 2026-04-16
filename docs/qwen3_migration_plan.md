# WSI-LLaVA에서 LLaVA 텍스트 백본을 Qwen3로 대체하는 실전 가이드

이 문서는 **현재 WSI-LLaVA 파이프라인(WSI `.pt` feature + mm projector)** 를 유지하면서, 텍스트 백본을 LLaVA-LLaMA 계열에서 Qwen3로 전환하는 방법을 정리합니다.

## 1. 결론 먼저: 어떤 경로가 가장 맞는가

WSI-LLaVA는 이미 `vision_tower + projector + language_model` 구조로 분리되어 있으므로,
Qwen-VL 네이티브(비전 내장) 경로보다 **LLaVA-style Qwen 이식 경로**가 맞습니다.

- 권장 기준선:
  1) LLaVA PR #1573의 Qwen2 이식 패턴 (`llava_qwen.py`, `finetune_qwen_2.sh`)을 구조 참고
  2) TobyYang7/Llava_Qwen2에서 실제 스크립트 운용 방식 참고
  3) Qwen3는 HF Transformers의 `Qwen3*` 클래스로 교체

즉, **Qwen2 이식 패턴을 그대로 따르되 Qwen3 API로 치환**하는 전략이 가장 리스크가 낮습니다.

---

## 2. 공개 레퍼런스 정합성 체크

### 2.1 LLaVA PR #1573 (Qwen2)
- URL: https://github.com/haotian-liu/LLaVA/pull/1573
- 정합 포인트:
  - LLaVA 내부에 Qwen 계열 language adapter를 추가하는 방식
  - 기존 LLaVA 학습 루프를 크게 건드리지 않고 확장

### 2.2 TobyYang7/Llava_Qwen2
- URL: https://github.com/TobyYang7/Llava_Qwen2
- 정합 포인트:
  - `pretrain_qwen2.sh` / `ft_qwen2.sh`처럼 stage를 분리한 운용
  - LLaVA 스타일 코드베이스에서 Qwen 텍스트 백본이 동작하는 공개 사례

### 2.3 Qwen3 Transformers 문서
- URL: https://huggingface.co/docs/transformers/model_doc/qwen3
- 정합 포인트:
  - `Qwen3Config`, `Qwen3Model`, `Qwen3ForCausalLM` 클래스 제공
  - HF Auto 계열과 함께 사용 가능

### 2.4 Qwen-VL 파인튜닝 레포들(비교군)
- Qwen-VL-Series-Finetune: https://github.com/2U1/Qwen-VL-Series-Finetune
- lmms-finetune: https://github.com/zjysteven/lmms-finetune
- HF Cookbook(Qwen2-VL + TRL): https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

> 위 레포들은 "비전 인코더가 모델 내부" 전제를 두는 경우가 많아, WSI-LLaVA의 `.pt feature` 직결 파이프라인과는 입력 스키마가 다릅니다.

---

## 3. 이 저장소에서 적용한 코드 변경(이번 반영)

### 3.1 Qwen3 language adapter 추가
- 파일: `WSI_LLAVA/llava/model/language_model/llava_qwen3.py`
- 내용:
  - `LlavaQwen3Config`, `LlavaQwen3Model`, `LlavaQwen3ForCausalLM` 추가
  - 프로젝트 레벨 가독성을 위해 `WSIQwen3Config`, `WSIQwen3ForCausalLM` 별칭 제공
  - 멀티모달 입력 준비(`prepare_inputs_labels_for_multimodal`) 경로를 기존 LLaVA와 동일하게 유지
  - Transformers 버전에 따라 `Qwen3*` 클래스가 없으면 `Qwen2*`로 fallback

### 3.2 모델 import/export 확장
- 파일: `WSI_LLAVA/llava/model/__init__.py`
- 내용:
  - `LlavaQwen3ForCausalLM`, `LlavaQwen3Config` + `WSIQwen3*` 별칭 export 추가

### 3.3 추론 로더 분기 확장
- 파일: `WSI_LLAVA/llava/model/builder.py`
- 내용:
  - `model_name`에 `qwen3/qwen2/qwen` 포함 시 Qwen 분기 로드
  - projector-only 로딩(model_base 제공) 경로에서도 Qwen 분기 지원

### 3.4 학습 루프 분기 확장
- 파일: `WSI_LLAVA/llava/train/train.py`
- 내용:
  - `--llm_backbone` 인자 추가 (`auto`, `qwen3`, `mpt`, `mistral`, `llama` 등)
  - auto 감지 함수(`infer_backbone`) 추가
  - vision_tower 사용 시 Qwen3 백본으로 `WSIQwen3ForCausalLM` 로드
  - tokenizer fast 경로를 Qwen 계열에 활성화

### 3.5 Qwen3 LoRA 학습 스크립트 추가
- 파일: `WSI_LLAVA/scripts/v1_5/finetune_lora_qwen3.sh`
- 내용:
  - `--llm_backbone qwen3`
  - Qwen3 모델 경로 기반 deepspeed 학습 템플릿

---

## 4. 실제 학습 절차 (WSI-LLaVA 기준)

### Stage A: Projector warm-up (선택)
- 목적: 새 백본 임베딩 공간에 projector를 빠르게 정렬
- 방법:
  - `--tune_mm_mlp_adapter True`
  - LLM 본체 고정 + projector만 학습

### Stage B: Qwen3 LoRA 본학습
- 방법:
  - `bash WSI_LLAVA/scripts/v1_5/finetune_lora_qwen3.sh`
  - LoRA + projector 동시 업데이트
- 전제:
  - Qwen3 체크포인트만으로는 부족하며, `vision_tower`, WSI feature(.pt), 학습 JSON, deepspeed 환경이 모두 필요

### Stage C: 평가
- WSI-Bench 결과를 기존 baseline과 비교
- 메트릭:
  - `WSI-Metric/WSI-Precision_stage*.py`
  - `WSI-Metric/WSI-Relevance_stage_*.py`

---

## 5. 권장 하이퍼파라미터 시작점

- precision: bf16
- learning rate: 2e-4 (LoRA), mm_projector_lr 2e-5
- batch: GPU 메모리에 맞춰 `batch_size 8 + grad_acc 16`에서 시작
- max length: 2048

> Qwen3 계열은 컨텍스트를 길게 가져갈 수 있지만, WSI 파이프라인에서는 이미지 토큰/텍스트 토큰 동시 메모리 사용량을 먼저 확인하세요.

---

## 6. 자주 깨지는 지점 (실전 체크리스트)

1) tokenizer pad/eos 미정의
- 증상: 학습 초반 loss 이상
- 조치: train.py의 tokenizer 기본값 로직 유지

2) LoRA target 모듈 자동탐색 품질
- 증상: 학습은 되는데 성능 정체
- 조치: `find_all_linear_names` 결과를 로깅해 Qwen3 블록이 포함되는지 점검

3) Qwen3 클래스 미지원 transformers 버전
- 증상: import 오류
- 조치: 현재 구현은 Qwen2 클래스 fallback을 포함

---

## 7. 운영 전략 제안

- 1차: LLaVA baseline vs Qwen3(동일 데이터/동일 seed) A/B
- 2차: 성능 동급 이상이면 Qwen3를 default 백본으로 승격
- 3차: 실패 시 즉시 롤백 가능하도록 `--llm_backbone` 스위치로 공존 운용

이 방식이면 연구 재현성과 운영 안정성을 동시에 지킬 수 있습니다.

---

## 8. Qwen3 사용 시에도 LLaVA를 쓰는가?

네. 현재 이식 방식은 **LLaVA 프레임워크 위에 Qwen3 텍스트 백본만 교체**하는 구조입니다.

- `llava_qwen3.py`는 `LlavaMetaModel`, `LlavaMetaForCausalLM`을 상속해 LLaVA 멀티모달 결합 경로를 그대로 사용합니다.
- `builder.py`, `train.py`의 Qwen3 분기는 가독성을 위해 `WSIQwen3ForCausalLM` 별칭을 사용하며, 내부 구현은 `LlavaQwen3ForCausalLM`과 동일합니다.
- 즉, WSI feature 주입/프로젝터/이미지 토큰 처리 등은 LLaVA 방식을 그대로 유지합니다.

정적 검증은 아래 스크립트로 바로 확인할 수 있습니다.

```bash
python WSI_LLAVA/scripts/check_qwen3_llava_usage.py
```

추가로, 학습 실행 전 요구사항 확인:

```bash
bash WSI_LLAVA/scripts/v1_5/finetune_lora_qwen3.sh
```

스크립트는 필수 경로/명령(deepspeed)이 없으면 에러를 내고 종료합니다.
