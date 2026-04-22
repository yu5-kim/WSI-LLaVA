import argparse
import torch
import os
import json
import re
from tqdm import tqdm
import shortuuid
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_files):
    image = torch.load(image_files)
    return image


def sample_patch_features(image, patch_sample_ratio):
    if patch_sample_ratio is None or patch_sample_ratio >= 1.0:
        return image

    if patch_sample_ratio < 0 or patch_sample_ratio > 1:
        raise ValueError(f"patch_sample_ratio must be in [0, 1], got {patch_sample_ratio}")

    if not isinstance(image, torch.Tensor) or image.ndim == 0 or image.shape[0] == 0:
        return image

    num_patches = image.shape[0]
    sampled_patch_count = max(1, math.ceil(num_patches * patch_sample_ratio))
    if sampled_patch_count >= num_patches:
        return image

    sampled_indices = torch.randperm(num_patches, device=image.device)[:sampled_patch_count]
    sampled_indices, _ = torch.sort(sampled_indices)
    return image.index_select(0, sampled_indices)


def trim_generated_answer(text):
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


def trim_at_stop_strings(text, stop_strings):
    if not text:
        return text
    cut_pos = len(text)
    for marker in stop_strings or []:
        if not marker:
            continue
        pos = text.find(marker)
        if pos != -1:
            cut_pos = min(cut_pos, pos)
    return text[:cut_pos].strip()


def is_qwen_family(model_name: str, tokenizer) -> bool:
    lowered = (model_name or "").lower()
    if any(k in lowered for k in ("qwen3", "qwen2", "qwen")):
        return True
    tok_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    tok_class = tokenizer.__class__.__name__.lower()
    return "qwen" in tok_name or "qwen" in tok_class


def _build_user_content(cur_prompt, model):
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + cur_prompt
    return DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt


def is_llava_style_model(model_name: str, model) -> bool:
    lowered = (model_name or "").lower()
    if "llava" in lowered or "wsi-llava" in lowered:
        return True
    arch = getattr(getattr(model, "config", None), "architectures", None) or []
    joined_arch = " ".join(str(x).lower() for x in arch)
    if "llava" in joined_arch:
        return True
    return hasattr(getattr(model, "config", None), "mm_use_im_start_end")


def resolve_prompt_format(model_name, model, tokenizer, args):
    if args.prompt_format == "qwen":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("--prompt-format qwen requires tokenizer.apply_chat_template")
        return "qwen"
    if args.prompt_format == "llava":
        return "llava"
    # auto mode: prefer llava-style prompt for llava checkpoints even when backbone tokenizer is qwen.
    if is_llava_style_model(model_name, model):
        return "llava"
    if is_qwen_family(model_name, tokenizer) and hasattr(tokenizer, "apply_chat_template"):
        return "qwen"
    return "llava"


def build_prompt_and_stop_words(cur_prompt, model, model_name, tokenizer, args):
    prompt_format = resolve_prompt_format(model_name, model, tokenizer, args)
    qwen_mode = prompt_format == "qwen"
    user_content = _build_user_content(cur_prompt, model)

    if qwen_mode:
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
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
            "<|im_end|>",
            "<|endoftext|>",
        ]
        return prompt, stop_words, qwen_mode, prompt_format

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], user_content)
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
    return prompt, stop_words, qwen_mode, prompt_format


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


def slice_generated_tokens(output_ids, input_ids):
    if output_ids.ndim != 2 or input_ids.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors for output_ids/input_ids, got {output_ids.shape}, {input_ids.shape}"
        )
    if output_ids.shape[0] != input_ids.shape[0]:
        raise ValueError(
            f"Batch mismatch between output_ids ({output_ids.shape[0]}) and input_ids ({input_ids.shape[0]})"
        )
    prompt_token_len = input_ids.shape[1]
    if output_ids.shape[1] < prompt_token_len:
        raise ValueError(
            f"Generated sequence shorter than prompt: output={output_ids.shape[1]}, prompt={prompt_token_len}"
        )
    return output_ids[:, prompt_token_len:]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # ===== 1. Load question file =====
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # ===== 2. Check and load existing output file =====
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    decode_dump_file = None
    if args.dump_decode_stages:
        decode_dump_path = args.decode_dump_file
        if decode_dump_path is None:
            decode_dump_path = f"{answers_file}.decode_stages.jsonl"
        decode_dump_path = os.path.expanduser(decode_dump_path)
        os.makedirs(os.path.dirname(decode_dump_path), exist_ok=True)
        decode_dump_file = open(decode_dump_path, "a", encoding="utf-8")
        print(f"Decode-stage dump enabled: {decode_dump_path}")

    def _load_existing_records(path):
        processed = set()
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return processed, 0

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        decoder = json.JSONDecoder()
        pos = 0
        malformed = 0
        n = len(raw)
        while pos < n:
            while pos < n and raw[pos].isspace():
                pos += 1
            if pos >= n:
                break
            try:
                obj, next_pos = decoder.raw_decode(raw, pos)
            except json.JSONDecodeError:
                malformed += 1
                next_nl = raw.find("\n", pos)
                if next_nl == -1:
                    break
                pos = next_nl + 1
                continue
            qid = obj.get("question_id") if isinstance(obj, dict) else None
            if qid:
                processed.add(qid)
            pos = next_pos
        return processed, malformed

    processed_ids, malformed_records = _load_existing_records(answers_file)
    if processed_ids:
        print(f"Existing results file detected: {answers_file}")
        print(f"Skipping {len(processed_ids)} already processed samples.")
        if malformed_records:
            print(f"Warning: skipped {malformed_records} malformed JSON fragments in existing answers file.")
        ans_file = open(answers_file, "a+", encoding="utf-8")
        ans_file.seek(0, os.SEEK_END)
        if ans_file.tell() > 0:
            ans_file.seek(ans_file.tell() - 1)
            tail = ans_file.read(1)
            if tail != "\n":
                ans_file.write("\n")
        ans_file.seek(0, os.SEEK_END)
    else:
        ans_file = open(answers_file, "w", encoding="utf-8")
        print(f"Creating new results file: {answers_file}")

    # ===== 3. Iterate over question list =====
    prompt_format_logged = False
    for line in tqdm(questions, desc="Inference"):
        idx = line["question_id"]
        if idx in processed_ids:
            continue  # skip already processed samples

        image_file = line["image"]
        qs = line["question"]
        metadata = line["metadata"]
        Tanswer = line["T-answer"]

        cur_prompt = qs
        prompt, stop_words, qwen_mode, prompt_format = build_prompt_and_stop_words(
            cur_prompt, model, model_name, tokenizer, args
        )
        if not prompt_format_logged:
            print(f"[prompt-format] selected={prompt_format}")
            prompt_format_logged = True

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        image_path = os.path.join(args.image_folder, image_file)
        image = load_image(image_path)
        image = sample_patch_features(image, args.patch_sample_ratio)
        image_tensor = image.to(model.device, dtype=torch.float16)
        # preserve order while removing duplicates/empties
        stop_words = list(dict.fromkeys([w for w in stop_words if w]))
        stopping_criteria = [KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)]
        eos_token_id, pad_token_id = resolve_generation_eos_and_pad(model, tokenizer)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = slice_generated_tokens(output_ids, input_ids)
        raw_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        decoded_no_special = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        outputs = trim_at_stop_strings(decoded_no_special, stop_words)
        if qwen_mode:
            outputs = re.sub(r"^\s*(assistant|ASSISTANT|Assistant)\s*:\s*", "", outputs)
        outputs = trim_generated_answer(outputs)
        if decode_dump_file is not None:
            decode_dump_file.write(json.dumps({
                "question_id": idx,
                "image": image_file,
                "prompt_format": prompt_format,
                "prompt": prompt,
                "input_token_count": int(input_ids.shape[1]),
                "generated_token_count": int(generated_ids.shape[1]),
                "raw_decoded": raw_decoded,
                "decoded_no_special": decoded_no_special,
                "post_stop_trim": trim_at_stop_strings(decoded_no_special, stop_words),
                "final_output": outputs,
            }, ensure_ascii=False) + "\n")
            decode_dump_file.flush()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "image": image_file,
            "question": cur_prompt,
            "answer": outputs,
            "T-answer": Tanswer,
            "metadata": metadata
        }) + "\n")
        ans_file.flush()
        processed_ids.add(idx)

    ans_file.close()
    if decode_dump_file is not None:
        decode_dump_file.close()
    print("All samples inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="auto",
        choices=["auto", "llava", "qwen"],
        help="Prompt format selection. auto prefers llava format for llava-style checkpoints, otherwise uses qwen template when detected.",
    )
    parser.add_argument(
        "--dump-decode-stages",
        dest="dump_decode_stages",
        action="store_true",
        help="Dump raw decode -> post-stop-trim -> final output stages to JSONL for debugging.",
    )
    parser.add_argument(
        "--no-dump-decode-stages",
        dest="dump_decode_stages",
        action="store_false",
        help="Disable decode-stage debug dump.",
    )
    parser.set_defaults(dump_decode_stages=True)
    parser.add_argument(
        "--decode-dump-file",
        type=str,
        default=None,
        help="Optional path for decode-stage dump JSONL. Defaults to <answers-file>.decode_stages.jsonl",
    )
    parser.add_argument("--patch-sample-ratio", type=float, default=1.0,
                        help="Ratio of patch features to sample per slide during evaluation. "
                             "Use 1.0 to keep all patches.")
    args = parser.parse_args()

    eval_model(args)
