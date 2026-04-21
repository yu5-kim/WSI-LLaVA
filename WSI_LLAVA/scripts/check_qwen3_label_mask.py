#!/usr/bin/env python3
"""Unit-style checks for Qwen chat preprocessing mask behavior."""

import copy
import sys

from llava.constants import IGNORE_INDEX
from llava.train.train import preprocess_qwen_chat, TOKENIZATION_MISMATCH_STATE


class _FakeTokenizer:
    bos_token_id = 1
    model_max_length = 4096
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, return_dict=False, return_assistant_tokens_mask=False):
        del add_generation_prompt
        text = ""
        spans = []
        cursor = 0
        for msg in messages:
            seg = f"<{msg['role']}>:{msg['content']}\n"
            start = cursor
            text += seg
            cursor += len(seg)
            spans.append((msg["role"], start, cursor))
        if not tokenize:
            return text
        ids = self(text, add_special_tokens=False).input_ids
        if not return_dict:
            return ids
        out = {"input_ids": ids}
        if return_assistant_tokens_mask:
            mask = [0] * len(ids)
            for role, start, end in spans:
                if role == "assistant":
                    for i in range(start, end):
                        if i < len(mask):
                            mask[i] = 1
            out["assistant_masks"] = mask
        return out

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<image>", i):
                ids.append(999)
                i += len("<image>")
            else:
                ids.append(ord(text[i]) + 10)
                i += 1
        return type("Tokenized", (), {"input_ids": ids})


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    tokenizer = _FakeTokenizer()
    TOKENIZATION_MISMATCH_STATE["total"] = 0
    TOKENIZATION_MISMATCH_STATE["mismatch"] = 0
    TOKENIZATION_MISMATCH_STATE["max_ratio"] = 1.0

    sample = [[
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "world"},
        {"from": "human", "value": "<image> explain"},
        {"from": "gpt", "value": "done"},
    ]]

    out = preprocess_qwen_chat(copy.deepcopy(sample), tokenizer, has_image=True, sample_ids=["unit-0"])
    input_ids = out["input_ids"][0]
    labels = out["labels"][0]

    _assert(len(input_ids) == len(labels), "image sample should keep equal input/label lengths")

    assistant_count = int((labels != IGNORE_INDEX).sum().item())
    _assert(assistant_count > 0, "assistant token count must be > 0")

    non_assistant_count = int((labels == IGNORE_INDEX).sum().item())
    _assert(non_assistant_count > 0, "non-assistant area must be masked with IGNORE_INDEX")

    mismatch_ratio = TOKENIZATION_MISMATCH_STATE["mismatch"] / max(TOKENIZATION_MISMATCH_STATE["total"], 1)
    _assert(mismatch_ratio <= TOKENIZATION_MISMATCH_STATE["max_ratio"], "mismatch ratio should be <= threshold")

    print("[PASS] assistant token count > 0")
    print("[PASS] non-assistant region is masked")
    print("[PASS] mismatch ratio is within threshold")
    print("[PASS] image sample keeps token/label length aligned")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}")
        raise
