#!/usr/bin/env python3
"""Static verification that Qwen3 path still uses LLaVA multimodal stack."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
checks = {
    "adapter_inherits_llava_meta": (
        ROOT / "WSI_LLAVA/llava/model/language_model/llava_qwen3.py",
        ["class LlavaQwen3Model(LlavaMetaModel", "class LlavaQwen3ForCausalLM(_QwenForCausalLM, LlavaMetaForCausalLM)"],
    ),
    "builder_routes_qwen_to_llava_adapter": (
        ROOT / "WSI_LLAVA/llava/model/builder.py",
        ["model = WSIQwen3ForCausalLM.from_pretrained"],
    ),
    "train_routes_qwen_to_llava_adapter": (
        ROOT / "WSI_LLAVA/llava/train/train.py",
        ["model = WSIQwen3ForCausalLM.from_pretrained"],
    ),
}

failed = []
for name, (path, patterns) in checks.items():
    text = path.read_text(encoding="utf-8")
    missing = [p for p in patterns if p not in text]
    if missing:
        failed.append((name, path, missing))
    else:
        print(f"[PASS] {name}: {path}")

if failed:
    print("\n[FAIL] Some checks failed:")
    for name, path, missing in failed:
        print(f"- {name} ({path})")
        for pattern in missing:
            print(f"  missing: {pattern}")
    sys.exit(1)

print("\nConclusion: Qwen3 경로에서도 LLaVA 멀티모달 구성요소를 사용합니다.")
