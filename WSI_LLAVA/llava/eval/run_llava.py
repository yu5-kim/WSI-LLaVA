import argparse
import torch
import os
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    IMAGE_PLACEHOLDER,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.eval.prompt_utils import (
    build_prompt_and_stop_words,
    postprocess_generated_text,
    resolve_generation_eos_and_pad,
)


def image_parser(args):
    return args.image_file.split(args.sep)


def load_image(image_files):
    image = torch.load(image_files)
    image = image.unsqueeze(0)
    return image


def eval_model(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    if IMAGE_PLACEHOLDER in qs:
        qs = qs.replace(IMAGE_PLACEHOLDER, "").strip()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    prompt, stop_words, qwen_mode = build_prompt_and_stop_words(
        qs, model, model_name, tokenizer, args.conv_mode
    )

    image_files = image_parser(args)
    for image_file in image_files:
        if not os.path.exists(image_file):
            print(f"文件 {image_file} 不存在")
        elif not os.access(image_file, os.R_OK):
            print(f"文件 {image_file} 不可读")
        else:
            print(f"文件 {image_file} 存在且可读")

        image = load_image(image_file)

    print(image.shape)

    image_sizes = image.shape
    images_tensor = image.to(model.device, dtype=torch.float16)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stopping_criteria = [KeywordsStoppingCriteria(list(dict.fromkeys([w for w in stop_words if w])), tokenizer, input_ids)]
    eos_token_id, pad_token_id = resolve_generation_eos_and_pad(model, tokenizer)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            stopping_criteria=stopping_criteria,
        )

    generated_ids = output_ids[:, input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    outputs = postprocess_generated_text(outputs, qwen_mode)
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
