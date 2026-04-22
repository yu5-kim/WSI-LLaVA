import re

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


def is_qwen_family(model_name: str, tokenizer) -> bool:
    lowered = (model_name or "").lower()
    if any(k in lowered for k in ("qwen3", "qwen2", "qwen")):
        return True
    tok_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    tok_class = tokenizer.__class__.__name__.lower()
    return "qwen" in tok_name or "qwen" in tok_class


def build_prompt(cur_prompt: str, model, model_name: str, tokenizer, conv_mode: str):
    qwen_mode = is_qwen_family(model_name, tokenizer) and hasattr(tokenizer, "apply_chat_template")
    if qwen_mode:
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{cur_prompt}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, qwen_mode

    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + cur_prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + cur_prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), qwen_mode


def postprocess_output(text: str, qwen_mode: bool = False) -> str:
    output = (text or "").strip()
    if qwen_mode:
        output = re.sub(r"^\s*(assistant|ASSISTANT|Assistant)\s*:\s*", "", output)
    return output.strip()
