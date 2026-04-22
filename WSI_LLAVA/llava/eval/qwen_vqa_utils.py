# WSI / Qwen3 LLaVA: inference-time alignment with train.py preprocess_qwen_chat_template + _normalize_qwen_multimodal_user_content
import copy
import json
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from llava.mm_utils import tokenizer_image_token
from llava.train.train import (
    _normalize_qwen_multimodal_user_content,
    normalize_qwen_turn_content,
    ensure_tokenizer_pad_token,
    get_tokenizer_pad_token_id,
)
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)


def build_qwen_wsi_vqa_inputs(
    tokenizer: PreTrainedTokenizer,
    question: str,
    mm_use_im_start_end: bool,
) -> Tuple[str, torch.Tensor]:
    """Match LazySupervisedDataset + preprocess_qwen_chat_template: single user turn, then add_generation_prompt=True.
    Returns the serialized string (for debugging) and input_ids.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Qwen VQA needs tokenizer.apply_chat_template")

    user_text = (question or "").strip()
    if DEFAULT_IMAGE_TOKEN not in user_text:
        user_text = f"{DEFAULT_IMAGE_TOKEN}\n{user_text}"

    source: List[dict] = [{"from": "human", "value": user_text}]
    source = copy.deepcopy(source)
    data_args = SimpleNamespace(mm_use_im_start_end=mm_use_im_start_end)
    _normalize_qwen_multimodal_user_content(source, data_args)
    if not source:
        raise ValueError("empty Qwen WSI VQA source")

    uval = normalize_qwen_turn_content("user", source[0]["value"])
    messages = [{"role": "user", "content": uval}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if mm_use_im_start_end:
        image_token = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    input_ids = tokenizer_image_token(
        prompt, tokenizer, image_token=image_token, return_tensors="pt"
    ).unsqueeze(0)
    return prompt, input_ids


def build_qwen_wsi_vqa_input_ids(
    tokenizer: PreTrainedTokenizer,
    question: str,
    mm_use_im_start_end: bool,
) -> torch.Tensor:
    return build_qwen_wsi_vqa_inputs(tokenizer, question, mm_use_im_start_end)[1]


def resolve_eos_and_pad_for_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Union[int, List[int]], int]:
    """Prefer model.generation_config; add im_end if encodes to a single id distinct from EOS."""
    ensure_tokenizer_pad_token(tokenizer)
    pad_id = int(get_tokenizer_pad_token_id(tokenizer))

    eos_set: List[int] = []
    gen_cfg = getattr(model, "generation_config", None)
    g_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if g_eos is not None:
        if isinstance(g_eos, (list, tuple)):
            eos_set.extend(int(x) for x in g_eos)
        else:
            eos_set.append(int(g_eos))
    t_eos = getattr(tokenizer, "eos_token_id", None)
    if t_eos is not None and int(t_eos) not in eos_set:
        eos_set.append(int(t_eos))

    for marker in ("<|im_end|>", "<|endoftext|>"):
        tid = _try_single_eos_id(tokenizer, marker)
        if tid is not None and tid not in eos_set:
            eos_set.append(tid)

    if not eos_set:
        eos_set.append(int(t_eos) if t_eos is not None else pad_id)

    eos_for_generate: Union[int, List[int]] = (
        eos_set[0] if len(eos_set) == 1 else list(dict.fromkeys(eos_set))
    )
    return eos_for_generate, pad_id


def _try_single_eos_id(tokenizer: PreTrainedTokenizer, text: str) -> Optional[int]:
    """Return token id only if the string encodes to exactly one subword (for eos_token_id list)."""
    try:
        e = tokenizer(text, add_special_tokens=False)
        ids = e["input_ids"] if isinstance(e, dict) else e
        if not ids or len(ids) != 1:
            return None
        tid = int(ids[0])
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is not None and tid == int(unk):
            return None
        return tid
    except Exception:  # noqa: BLE001
        return None


def qwen_extra_stop_strs() -> List[str]:
    """When model does not hit EOS, stop on next-turn / report boilerplate (substring match in KeywordsStoppingCriteria)."""
    return [
        "<|im_start|>",
        "<|im_start|>user",
        "<|im_start|>User",
        "<|im_start|>system",
        "<|im_start|>assistant",
        "Slide 2:",  # noqa: S105
        "Slide 3:",
        "Additional Slide:",
        "Final Diagnosis:",  # noqa: S105
        "\n## ",
        "ASSISTANT:",
    ]


def strip_qwen_decoded_artifacts(
    text: str,
) -> str:
    """String-level cleanup when EOS/stop did not cut before next control token (after token-sliced decode)."""
    if not text:
        return text
    out = text
    cut = len(out)
    for anchor in (
        "<|im_start|>",
    ):
        p = out.find(anchor)
        if p != -1 and p < cut:
            cut = p
    out = out[:cut].rstrip()
    for suf in (
        "<|im_end|>",
        "<|endoftext|>",
    ):
        if out.endswith(suf):
            out = out[: -len(suf)].rstrip()
    return out


def trim_wsi_bench_artifacts(text: str) -> str:
    """Extra leak trimming for WSI / narrative-report style generations (applies to any backbone)."""
    if not text:
        return text
    stop_markers = [
        "\nUSER:",
        "\nASSISTANT:",
        "\nHuman:",
        "\nQuestion:",
        "\nQUESTION:",
        "\nTASK:",
        "\nASK:",
        "\nQ:",
        "\n<|im_start|>",
        "\nSlide 2:",  # noqa: S105
        "\nSlide 3:",
        "\nAdditional Slide:",
        "\nFinal Diagnosis:",  # noqa: S105
        "\n## Treatment",  # noqa: S105
        "\n## Follow-up",  # noqa: S105
        "\n## Prognosis",  # noqa: S105
        "\n## Patient",  # noqa: S105
        "\nPathologist's",
        "\nPathologist’s",
    ]
    cut_pos = len(text)
    for marker in stop_markers:
        pos = text.find(marker)
        if pos != -1:
            cut_pos = min(cut_pos, pos)
    text = text[:cut_pos].strip()

    cleaned_lines: List[str] = []
    prev: Optional[str] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        numeric_like = sum(ch.isdigit() for ch in line) > max(16, int(0.7 * len(line)))
        if numeric_like:
            break
        if line == prev:
            continue
        cleaned_lines.append(line)
        prev = line
    return "\n".join(cleaned_lines).strip()


def write_debug_decode_line(
    path: str,
    record: dict,
) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
