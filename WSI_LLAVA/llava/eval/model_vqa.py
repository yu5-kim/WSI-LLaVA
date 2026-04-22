import argparse
import os
import json
import warnings
from tqdm import tqdm
import torch
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.qwen_vqa_utils import (
    build_qwen_wsi_vqa_inputs,
    resolve_eos_and_pad_for_generate,
    qwen_extra_stop_strs,
    strip_qwen_decoded_artifacts,
    trim_wsi_bench_artifacts,
    write_debug_decode_line,
)
from llava.train.train import is_qwen_family_tokenizer, ensure_tokenizer_pad_token


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


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    ensure_tokenizer_pad_token(tokenizer)

    use_qwen = is_qwen_family_tokenizer(tokenizer)
    if use_qwen and args.conv_mode not in ("qwen", "auto", None):
        warnings.warn(
            f"Qwen checkpoint detected: --conv-mode={args.conv_mode!r} is ignored; "
            "using apply_chat_template like train preprocess_qwen_chat_template (no system in messages).",
            stacklevel=1,
        )

    debug_path = args.debug_decode
    if debug_path and args.debug_decode_max > 0:
        _dd = os.path.dirname(os.path.abspath(debug_path))
        if _dd:
            os.makedirs(_dd, exist_ok=True)

    # ===== 1. Load question file =====
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # ===== 2. Check and load existing output file =====
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)

    processed_ids = set()
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        print(f"Existing results file detected: {answers_file}")
        with open(answers_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["question_id"])
                except Exception:  # noqa: BLE001
                    continue
        print(f"Skipping {len(processed_ids)} already processed samples.")
        ans_file = open(answers_file, "a")  # append mode
    else:
        ans_file = open(answers_file, "w")
        print(f"Creating new results file: {answers_file}")

    debug_written = 0
    # ===== 3. Iterate over question list =====
    for line in tqdm(questions, desc="Inference"):
        idx = line["question_id"]
        if idx in processed_ids:
            continue  # skip already processed samples

        image_file = line["image"]
        qs = line["question"]
        metadata = line["metadata"]
        Tanswer = line["T-answer"]

        cur_prompt = qs
        mm_use = getattr(model.config, "mm_use_im_start_end", False)

        qwen_template_str = None
        if use_qwen:
            qwen_template_str, input_ids = build_qwen_wsi_vqa_inputs(
                tokenizer, cur_prompt, mm_use
            )
            input_ids = input_ids.to(model.device)
            stop_words = qwen_extra_stop_strs()
        else:
            if mm_use:
                user_segment = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                user_segment = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], user_segment)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            stop_words = []
            if conv.sep2:
                stop_words.append(conv.sep2)
            stop_words.extend(
                [
                    f"{conv.roles[0]}:",
                    "\nUSER:",
                    "\nASSISTANT:",
                    "\nHuman:",
                    "\nQUESTION:",
                    "\nTASK:",
                ]
            )

        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        image_path = os.path.join(args.image_folder, image_file)
        image = load_image(image_path)
        image = sample_patch_features(image, args.patch_sample_ratio)
        image_tensor = image.to(model.device, dtype=torch.float16)

        stop_words = list(dict.fromkeys([w for w in stop_words if w]))
        stopping_criteria = [KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)]

        if use_qwen:
            eos_token_id, pad_token_id = resolve_eos_and_pad_for_generate(model, tokenizer)
        else:
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = int(tokenizer.pad_token_id)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.unsqueeze(0).half().to(model.device),
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

        generated_ids = output_ids[:, input_ids.shape[1]:]
        raw_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if use_qwen:
            after_control_strip = strip_qwen_decoded_artifacts(raw_decoded)
        else:
            after_control_strip = raw_decoded
        outputs = trim_wsi_bench_artifacts(after_control_strip)

        if (
            debug_path
            and args.debug_decode_max > 0
            and debug_written < args.debug_decode_max
        ):
            if use_qwen and qwen_template_str is not None:
                prompt_str = (qwen_template_str or "")[:2000]
            else:
                if mm_use:
                    pseg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + cur_prompt
                else:
                    pseg = DEFAULT_IMAGE_TOKEN + "\n" + cur_prompt
                c = conv_templates[args.conv_mode].copy()
                c.append_message(c.roles[0], pseg)
                c.append_message(c.roles[1], None)
                prompt_str = c.get_prompt()[:2000]
            write_debug_decode_line(
                debug_path,
                {
                    "question_id": idx,
                    "stage_prompt_note": prompt_str,
                    "stage_raw_new_tokens": raw_decoded,
                    "stage_after_qwen_strip": after_control_strip if use_qwen else None,
                    "stage_final": outputs,
                },
            )
            debug_written += 1

        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "image": image_file,
                    "question": cur_prompt,
                    "answer": outputs,
                    "T-answer": Tanswer,
                    "metadata": metadata,
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()
    print("All samples inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
        help="Non-Qwen: LLaVA conv template. Qwen: value is ignored; chat template is used.",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--patch-sample-ratio",
        type=float,
        default=1.0,
        help="Ratio of patch features to sample per slide during evaluation. "
        "Use 1.0 to keep all patches.",
    )
    parser.add_argument(
        "--debug-decode",
        type=str,
        default=None,
        help="If set, append up to --debug-decode-max JSONL lines with prompt note + decode stages.",
    )
    parser.add_argument(
        "--debug-decode-max",
        type=int,
        default=0,
        help="Max debug records (0 = disabled even if --debug-decode is set).",
    )
    args = parser.parse_args()

    eval_model(args)
