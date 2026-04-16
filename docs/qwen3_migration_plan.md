# WSI-LLaVA에서 Qwen3 기반으로 전환하기 위한 개선 설계

## 1) 목표와 배경

현재 저장소는 LLaVA 계열(`llava_llama`, `llava_mistral`, `llava_mpt`) 중심의 로딩/학습 경로를 사용합니다. Qwen3 계열로 이식하면 다음을 기대할 수 있습니다.

- 한국어/다국어 질의 대응력 개선
- 긴 컨텍스트 질의에서의 안정성 향상(병리 리포트/복합 질문)
- 동일 WSI feature 입력(576 토큰)에서 응답 정밀도 개선

핵심 원칙은 **WSI 인코더/프로젝터 파이프라인은 최대한 재사용**하고, **LLM 백본 추상화 계층을 확장**하는 것입니다.

---

## 2) 현재 구조 요약 (이식 관점)

- 모델 로딩 분기:
  - `WSI_LLAVA/llava/model/builder.py`에서 `model_name` 문자열 기준으로 LLaVA 계열 분기
- 언어모델 구현체:
  - `WSI_LLAVA/llava/model/language_model/llava_llama.py`
  - `WSI_LLAVA/llava/model/language_model/llava_mistral.py`
  - `WSI_LLAVA/llava/model/language_model/llava_mpt.py`
- 멀티모달 결합 공통부:
  - `WSI_LLAVA/llava/model/llava_arch.py`
- 학습 진입점:
  - `WSI_LLAVA/llava/train/train.py`
  - `WSI_LLAVA/scripts/v1_5/finetune_lora.sh`

즉, Qwen3 이식은 **(A) Qwen3용 language_model 어댑터 추가** + **(B) builder/train 분기 확장** + **(C) 학습 스크립트 인자 표준화**의 3축으로 진행하면 됩니다.

---

## 3) 제안 아키텍처

### 3.1 백본 추상화 레이어 추가

`language_model` 하위에 Qwen3 전용 파일을 추가합니다.

- 신규: `WSI_LLAVA/llava/model/language_model/llava_qwen3.py`
  - 역할:
    - Qwen3 CausalLM 래핑
    - `LlavaMetaModel`, `LlavaMetaForCausalLM`와 결합
    - 기존 LLaVA forward/generate 인터페이스와 동일 시그니처 유지

권장 클래스명 예시:

- `LlavaQwen3Config`
- `LlavaQwen3Model`
- `LlavaQwen3ForCausalLM`

### 3.2 Builder 분기 확장

`builder.py`에서 다음을 추가합니다.

- `model_name` 혹은 `config.model_type`에 `qwen3` 포함 시:
  - `AutoTokenizer.from_pretrained(..., use_fast=True)` 우선
  - `LlavaQwen3ForCausalLM.from_pretrained(...)`로 로드
- LoRA 병합 경로에서도 Qwen3 분기 지원
- `torch_dtype`, `attn_implementation`, quantization 옵션은 기존 정책 재사용

### 3.3 Train 인자 확장

`train.py` 및 `finetune_lora.sh`에서 아래를 명시적으로 노출합니다.

- `--llm_backbone {llama,mistral,mpt,qwen3}`
- `--model_name_or_path`에 Qwen3 체크포인트 전달
- tokenizer 옵션:
  - pad/eos 토큰 미정의 시 안전 기본값 지정

---

## 4) 구현 단계 (실행 순서)

### Phase 0. 호환성 확인

- Transformers/PEFT 버전에서 Qwen3 로딩 가능 여부 검증
- FlashAttention2와의 조합 검증

### Phase 1. 최소 이식(MVP)

1. `llava_qwen3.py` 추가 (학습/추론 공통 forward 경로 우선)
2. `builder.py` Qwen3 분기 추가
3. `__init__.py` export 갱신
4. 단일 배치 추론 스모크 테스트

완료 기준:
- 기존 WSI feature(.pt) 입력으로 에러 없이 답변 생성

### Phase 2. 학습 안정화

1. `finetune_lora.sh`에 Qwen3 preset 추가
2. LoRA target module를 Qwen3 구조에 맞춰 재정의
3. gradient checkpointing + bf16/fp16 조합 벤치

완료 기준:
- 1~3 epoch에서 loss 하강 및 NaN 없음

### Phase 3. 성능 최적화

1. projector learning rate / warmup 별도 튜닝
2. max length 및 대화 템플릿 튜닝
3. WSI-Bench dev set에서 task별 ablation

완료 기준:
- WSI-Precision, WSI-Relevance 모두 baseline(LLaVA) 대비 개선

---

## 5) 핵심 리스크와 대응

### 리스크 A: 토크나이저/특수토큰 불일치

- 증상: 이미지 토큰 삽입 후 시퀀스 어긋남, loss 폭증
- 대응:
  - special token 추가 직후 `resize_token_embeddings`
  - train/infer 동일 conversation template 강제

### 리스크 B: LoRA 타깃 모듈 미스매치

- 증상: 학습은 되지만 성능 정체
- 대응:
  - Qwen3 블록 네이밍 기준으로 target module 명시
  - projector-only / llm-only / joint LoRA 3조건 비교

### 리스크 C: 컨텍스트 길이와 VRAM 병목

- 증상: OOM, 배치 스루풋 급락
- 대응:
  - sequence length 스케줄링
  - gradient accumulation 증가
  - 필요 시 4bit QLoRA fallback

---

## 6) 평가 설계 (권장)

### 6.1 오프라인 자동평가

- 데이터: WSI-Bench test split 고정
- 메트릭:
  - `WSI-Metric/WSI-Precision_stage*.py`
  - `WSI-Metric/WSI-Relevance_stage_*.py`
- 비교군:
  1. 기존 LLaVA baseline
  2. Qwen3 + 동일 projector
  3. Qwen3 + projector 재학습

### 6.2 운영 관점 지표

- 평균 추론 지연(ms/sample)
- GPU 메모리 peak
- 장문 질의에서 답변 중단율

---

## 7) 바로 실행 가능한 작업 항목 (체크리스트)

- [ ] `llava_qwen3.py` 스캐폴드 추가
- [ ] `builder.py` Qwen3 분기 및 LoRA 병합 분기 추가
- [ ] `train.py` 인자(`--llm_backbone`) 추가
- [ ] `finetune_lora.sh`에 qwen3 preset 추가
- [ ] 스모크 추론(1 샘플) 통과
- [ ] 소규모 학습(수백 step) loss 안정성 확인
- [ ] WSI-Precision/WSI-Relevance 회귀 비교표 생성

---

## 8) 권장 마이그레이션 전략 (요약)

- 단번 교체보다 **듀얼 백본(LLAVA/Qwen3) 공존 기간**을 두고,
- 동일 데이터/동일 평가셋에서 회귀를 확인한 뒤,
- 성능 + 비용(지연/메모리) 기준을 동시에 만족하면 기본 백본을 Qwen3로 승격합니다.

이 방식이 연구 재현성과 운영 안정성을 모두 확보하는 가장 안전한 경로입니다.
