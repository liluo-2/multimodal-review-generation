#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
History-Item Image Description Script

Given a split file (.npz) and a dataset (.jsonl), collect all unique
parent_asins from the history items of the selected samples, then
generate an image description for each unique parent_asin using Qwen2.5-VL.

Output: one JSONL file with {"parent_asin": ..., "description": ...} per line.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


PRODUCT_DESCRIPTION_PROMPT = """Look at this product image and describe it in 2-4 natural sentences. Include:
- What the product is (category, brand/name if visible)
- Appearance (color, material, shape/design)
- Any visible text or labels
- Notable features or details

Requirements:
1. Only describe what is actually visible in the image
2. Keep it concise and natural, like describing the product to a friend
3. Do not guess or assume information that isn't clearly shown"""


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_unique_parent_asins(data, indices):
    """Collect unique parent_asins from history items of selected samples."""
    asins = set()
    for idx in indices:
        if idx >= len(data):
            continue
        for item in data[idx].get("history", []):
            pa = item.get("parent_asin")
            if pa:
                asins.add(pa)
    return sorted(asins)


def load_model(args):
    print(f"Loading base model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2" if not args.disable_flash_attn else "sdpa",
    )
    if args.lora_path:
        print(f"Loading LoRA from: {args.lora_path}")
        model = PeftModel.from_pretrained(
            model, args.lora_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if args.merge_lora:
            model = model.merge_and_unload()
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    return model, processor


def describe_image(model, processor, image_path, args):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": PRODUCT_DESCRIPTION_PROMPT},
    ]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def describe_images_batch(model, processor, image_paths, args):
    messages_list = []
    texts = []
    all_image_inputs = []
    all_video_inputs = []

    for image_path in image_paths:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": PRODUCT_DESCRIPTION_PROMPT},
        ]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        messages_list.append(messages)
        texts.append(chat_text)
        all_image_inputs.append(image_inputs[0] if image_inputs else None)
        all_video_inputs.append(video_inputs[0] if video_inputs else None)

    inputs = processor(
        text=texts,
        images=all_image_inputs if any(x is not None for x in all_image_inputs) else None,
        videos=all_video_inputs if any(x is not None for x in all_video_inputs) else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
        )

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main():
    parser = argparse.ArgumentParser(description="Generate image descriptions for history items")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--dataset_jsonl", type=str, required=True,
                        help="Path to final-desc-img.jsonl")
    parser.add_argument("--split_npz", type=str, required=True,
                        help="Path to the .npz split file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing {parent_asin}.jpg images")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--disable_flash_attn", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for image generation")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for data parallel inference")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Current shard id in [0, num_shards)")
    args = parser.parse_args()

    # Load split indices (combine train + test)
    splits = np.load(args.split_npz)
    all_indices = np.concatenate([splits[k] for k in splits]).astype(int)
    all_indices = np.unique(all_indices)
    print(f"Split keys: {list(splits.keys())}")
    print(f"Combined unique indices: {len(all_indices)}")

    # Load dataset
    print(f"Loading dataset: {args.dataset_jsonl}")
    data = load_jsonl(args.dataset_jsonl)
    print(f"Total samples in dataset: {len(data)}")

    # Collect unique parent_asins from history
    asins = collect_unique_parent_asins(data, all_indices)
    print(f"Unique history parent_asins to process: {len(asins)}")

    # Skip already-processed asins if output file exists (for resume)
    done = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add(rec["parent_asin"])
        print(f"Resuming: {len(done)} already done, {len(asins) - len(done)} remaining")

    remaining = [a for a in asins if a not in done]
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    # Deterministic sharding: each process handles a disjoint slice.
    remaining = remaining[args.shard_id::args.num_shards]
    print(
        f"Shard info: shard_id={args.shard_id}, num_shards={args.num_shards}, "
        f"this_shard_items={len(remaining)}"
    )
    if not remaining:
        print("All parent_asins already processed. Nothing to do.")
        return

    # Load model
    model, processor = load_model(args)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    missing_count = 0
    error_count = 0
    with open(args.output_path, "a", encoding="utf-8") as fout:
        valid_pairs = []
        for pa in remaining:
            img_path = os.path.join(args.image_dir, f"{pa}.jpg")
            if not os.path.exists(img_path):
                missing_count += 1
                continue
            valid_pairs.append((pa, img_path))

        for i in tqdm(range(0, len(valid_pairs), args.batch_size), desc="Generating history-item descriptions"):
            batch_pairs = valid_pairs[i:i + args.batch_size]
            batch_asins = [x[0] for x in batch_pairs]
            batch_paths = [x[1] for x in batch_pairs]
            try:
                batch_descs = describe_images_batch(model, processor, batch_paths, args)
                for pa, desc in zip(batch_asins, batch_descs):
                    fout.write(json.dumps({"parent_asin": pa, "description": desc}, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                error_count += len(batch_pairs)
                print(f"Batch error for {len(batch_pairs)} items: {e}")
                for pa in batch_asins:
                    fout.write(json.dumps({"parent_asin": pa, "description": f"[ERROR: {e}]"}, ensure_ascii=False) + "\n")
                fout.flush()

    total = len(remaining)
    print(f"\nDone: {total} processed ({missing_count} missing images, {error_count} errors)")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
