import re
import torch

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


def extract_generated_ids(output_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Robustly extract newly generated token ids.

    Some multimodal wrappers call HF generate() with inputs_embeds instead of
    input_ids, and the returned sequences can be generation-only (without prompt
    prefix). In that case, do not slice by prompt length.
    """
    if output_ids.ndim != 2:
        return output_ids
    prompt_len = int(input_ids.shape[1]) if input_ids is not None and input_ids.ndim == 2 else 0
    if prompt_len > 0 and output_ids.shape[1] > prompt_len:
        # Slice only when output actually contains prompt prefix.
        # Some multimodal wrappers return generation-only sequences.
        prefix_len = min(prompt_len, output_ids.shape[1])
        if prefix_len > 0 and torch.equal(output_ids[:, :prefix_len], input_ids[:, :prefix_len]):
            return output_ids[:, prompt_len:]
    return output_ids
