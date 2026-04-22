import argparse
import torch
import os
import json
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


def trim_after_stop_markers(text):
    """Cut obvious leaked next-turn markers that generation stop may miss."""
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
    return text[:cut_pos].strip()


def minimal_output_cleanup(text, separator=None):
    """Keep post-processing minimal: trim whitespace and optional trailing separator."""
    if not text:
        return text

    text = text.strip()
    if separator:
        separator = separator.strip()
        if separator and text.endswith(separator):
            text = text[:-len(separator)].rstrip()
    return text


def aggressive_trim_generated_answer(text):
    """Legacy aggressive cleaner kept optional for ablation/debug purposes."""
    if not text:
        return text

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

    processed_ids = set()
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        print(f"Existing results file detected: {answers_file}")
        with open(answers_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["question_id"])
                except:
                    continue
        print(f"Skipping {len(processed_ids)} already processed samples.")
        ans_file = open(answers_file, "a")  # append mode
    else:
        ans_file = open(answers_file, "w")
        print(f"Creating new results file: {answers_file}")

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
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        image_path = os.path.join(args.image_folder, image_file)
        image = load_image(image_path)
        image = sample_patch_features(image, args.patch_sample_ratio)
        image_tensor = image.to(model.device, dtype=torch.float16)
        # Prefer preventing multi-turn leakage at generation-time (template + stop/eos),
        # and keep text post-processing minimal/transparent.
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
        # preserve order while removing duplicates/empties
        stop_words = list(dict.fromkeys([w for w in stop_words if w]))
        stopping_criteria = [KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)]

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
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        raw_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        post_stop_trim = trim_after_stop_markers(raw_decoded)
        final_output = minimal_output_cleanup(post_stop_trim, separator=conv.sep2)
        if args.enable_aggressive_post_clean:
            final_output = aggressive_trim_generated_answer(final_output)
        outputs = final_output

        ans_id = shortuuid.uuid()
        answer_record = {
            "question_id": idx,
            "image": image_file,
            "question": cur_prompt,
            "answer": outputs,
            "T-answer": Tanswer,
            "metadata": metadata
        }
        if args.debug_output_stages:
            answer_record["debug_output_stages"] = {
                "raw_decoded": raw_decoded,
                "post_stop_trim": post_stop_trim,
                "final_output": final_output,
            }
        ans_file.write(json.dumps(answer_record) + "\n")
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
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--patch-sample-ratio", type=float, default=1.0,
                        help="Ratio of patch features to sample per slide during evaluation. "
                             "Use 1.0 to keep all patches.")
    parser.add_argument(
        "--enable-aggressive-post-clean",
        action="store_true",
        help="Enable legacy aggressive cleanup rules (numeric-tail/duplicate-line removal). "
             "Disabled by default to avoid masking generation stop failures.",
    )
    parser.add_argument(
        "--debug-output-stages",
        action="store_true",
        help="Store raw_decoded, post_stop_trim, and final_output for quality comparison.",
    )
    args = parser.parse_args()

    eval_model(args)
