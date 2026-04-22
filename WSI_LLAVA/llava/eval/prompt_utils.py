import re
from typing import List, Tuple

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


def trim_generated_answer(text: str) -> str:
    """Trim leaked multi-turn prefixes and noisy numeric tails."""
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
    ]
    cut_pos = len(text)
    for marker in stop_markers:
        pos = text.find(marker)
        if pos != -1:
            cut_pos = min(cut_pos, pos)
    text = text[:cut_pos].strip()

    cleaned_lines = []
    prev = None
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


def is_qwen_family(model_name: str, tokenizer) -> bool:
    lowered_model_name = (model_name or "").lower()
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    tokenizer_class = tokenizer.__class__.__name__.lower()
    has_qwen_marker = any("qwen" in src for src in (lowered_model_name, tokenizer_name, tokenizer_class))
    return has_qwen_marker and hasattr(tokenizer, "apply_chat_template")


def build_prompt_and_stop_words(cur_prompt: str, model, model_name: str, tokenizer, conv_mode: str, include_image: bool = True) -> Tuple[str, List[str], bool]:
    qwen_mode = is_qwen_family(model_name, tokenizer)
    if qwen_mode:
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{cur_prompt}" if include_image else cur_prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        stop_words = [
            "\nUSER:",
            "\nASSISTANT:",
            "\nHuman:",
            "\nQUESTION:",
            "\nTASK:",
            "USER:",
            "ASSISTANT:",
        ]
        return prompt, stop_words, qwen_mode

    if include_image:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + cur_prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + cur_prompt
    else:
        qs = cur_prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    stop_words = []
    if conv.sep2:
        stop_words.append(conv.sep2)
    stop_words.extend([
        f"{conv.roles[0]}:",
        "\nUSER:",
        "\nASSISTANT:",
        "\nHuman:",
        "\nQUESTION:",
        "\nTASK:",
    ])
    return prompt, stop_words, qwen_mode


def resolve_generation_eos_and_pad(model, tokenizer):
    eos_ids = []
    generation_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(generation_eos, (list, tuple)):
        eos_ids.extend([int(x) for x in generation_eos if x is not None])
    elif generation_eos is not None:
        eos_ids.append(int(generation_eos))
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_ids = list(dict.fromkeys(eos_ids))

    if not eos_ids:
        eos_token_id = None
    elif len(eos_ids) == 1:
        eos_token_id = eos_ids[0]
    else:
        eos_token_id = eos_ids

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[0]

    return eos_token_id, pad_token_id


def postprocess_generated_text(outputs: str, qwen_mode: bool) -> str:
    outputs = (outputs or "").strip()
    if qwen_mode:
        outputs = re.sub(r"^\s*(assistant|ASSISTANT|Assistant)\s*:\s*", "", outputs)
    return trim_generated_answer(outputs)
