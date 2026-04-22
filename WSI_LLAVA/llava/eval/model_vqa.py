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
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
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
                max_new_tokens=2048,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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
    parser.add_argument("--patch-sample-ratio", type=float, default=1.0,
                        help="Ratio of patch features to sample per slide during evaluation. "
                             "Use 1.0 to keep all patches.")
    args = parser.parse_args()

    eval_model(args)
