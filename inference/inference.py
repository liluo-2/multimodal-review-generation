import argparse
import os
import json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
import numpy as np
from tqdm import tqdm
from SFTDataset import RevGenDataset, QwenRevGenDataset, load_hist_img_desc_map  # 使用 Review Dataset
from model import load_model_and_tokenizer
from transformers import DataCollatorWithPadding
import random
from Datacollator import DataCollatorForSupervisedDataset

def load_image_descriptions(paths_csv):
    image_desc_map = {}
    img_only_ids = set()
    if not paths_csv:
        return image_desc_map, img_only_ids

    paths = [p.strip() for p in paths_csv.split(",") if p.strip()]
    for path in paths:
        lower_path = path.lower()
        is_img_only_file = "img_only" in lower_path
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec_id = str(rec.get("id", "")).strip()
                if not rec_id:
                    continue
                image_descs = rec.get("image_descriptions") or []
                desc_text = ""
                if isinstance(image_descs, list) and image_descs:
                    first = image_descs[0] or {}
                    desc_text = str(first.get("description", "")).strip()
                if desc_text:
                    image_desc_map[rec_id] = desc_text
                if is_img_only_file:
                    img_only_ids.add(rec_id)
    return image_desc_map, img_only_ids

def main(args):
    # Load model and tokenizer first
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        is_train=False,
        use_lora=args.use_lora,
        use_expert_mlp=args.use_expert_mlp,
        expert_bottleneck_dim=args.expert_bottleneck_dim,
        use_multi_head=args.use_multi_head,
    )
    print(f"[DEBUG] model training mode is set to: ", model.training)
    device = model.device
    print(f"[INFO] Using device: {device}")

    # # Load split ids
    # split = np.load(args.split_npy, allow_pickle=True).item()
    if args.split_npy.endswith(".npz"):
        z = np.load(args.split_npy, allow_pickle=False)
        split = {k: z[k] for k in z.files}
    else:
        split = np.load(args.split_npy, allow_pickle=True).item()
    split_ids = split[args.test_type]  # e.g., 'test', 'val', 'train'
    image_desc_map, img_only_ids = load_image_descriptions(args.image_desc_paths)
    hist_img_desc_map = load_hist_img_desc_map(args.hist_img_desc_path)

    if not args.use_hist_img_desc:
        hist_img_desc_map = {}
    if not args.use_target_img_desc:
        image_desc_map = {}
        img_only_ids = set()

    # Load dataset (in no_eval mode for inference)
    if args.model_name == "llama-3":
        dataset = RevGenDataset(
            args.dataset_jsonl,
            split_ids,
            tokenizer,
            max_length=8192,
            is_train=False,
            has_desc=args.use_desc,
            model=model,
            num_hist=args.num_hist,
            image_desc_map=image_desc_map,
            img_only_ids=img_only_ids,
            hist_img_desc_map=hist_img_desc_map,
            debug_print_prompts_n=args.debug_print_prompts_n,
        )
    elif args.model_name == "qwen-vl":
        dataset = QwenRevGenDataset(args.dataset_jsonl, split_ids, tokenizer, max_length=4096, is_train=False, image_root=f"${MULTIMODAL_DATA_ROOT}/img/{args.dataset}")

    # collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Prepare output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.output_name}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            print("[DEBUG] use_expert_mlp:", args.use_expert_mlp)
            print("[DEBUG] input_type in batch:", "input_type" in batch)
            if "input_type" in batch:
                print("[DEBUG] sample input_type:", batch["input_type"][:5])

            generate_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                # temperature=0.8,
                # top_p=0.9,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
            )
            if (args.use_expert_mlp or args.use_multi_head) and "input_type" in batch:
                generate_kwargs["input_type"] = batch["input_type"]
            outputs = model.generate(**generate_kwargs)

            # generate() returns prompt + new tokens; slice off the (left-padded) prompt
            prompt_len = input_ids.size(1)
            n = args.num_return_sequences
            bsz = input_ids.size(0)

            for i in range(bsz):
                rec_id = int(batch['meta_id'][i])
                outs = []
                for j in range(n):
                    generated = outputs[i * n + j][prompt_len:]
                    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                    outs.append(text)

                if n == 1:
                    record = {"id": rec_id, "output": outs[0]}      # 保持旧格式，向后兼容
                else:
                    record = {"id": rec_id, "outputs": outs}        # n>1 时一行一个 id，输出是 list
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Inference results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'Movies')")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Path to JSONL dataset for review generation")
    parser.add_argument("--split_npy", type=str, required=True, help="Path to split npy")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., 'llama-3', 'qwen-vl')")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or LoRA adapter")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true", help="Whether to load LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="inference_output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--test_type", type=str, default="test")
    parser.set_defaults(use_desc=True, use_hist_img_desc=False, use_target_img_desc=True)
    parser.add_argument("--use_desc", dest="use_desc", action="store_true", help="Include reference product text description.")
    parser.add_argument("--no-use_desc", dest="use_desc", action="store_false", help="Do not include reference product text description.")
    parser.add_argument("--use_hist_img_desc", dest="use_hist_img_desc", action="store_true", help="Include history image descriptions from hist_img_desc_path.")
    parser.add_argument("--no-use_hist_img_desc", dest="use_hist_img_desc", action="store_false", help="Disable history image descriptions.")
    parser.add_argument("--use_target_img_desc", dest="use_target_img_desc", action="store_true", help="Use target image descriptions from image_desc_paths.")
    parser.add_argument("--no-use_target_img_desc", dest="use_target_img_desc", action="store_false", help="Disable target image descriptions.")
    parser.add_argument("--num_hist", type=str, default="all", help="Number of most recent history items to use (3, 5, 10, or 'all').")
    parser.add_argument("--debug_print_prompts_n", type=int, default=0, help="If > 0, print first N prompts per modality during inference.")
    parser.add_argument(
        "--image_desc_paths",
        type=str,
        default="",
        help="Comma-separated jsonl files containing id->image_descriptions.",
    )
    parser.add_argument(
        "--hist_img_desc_path",
        type=str,
        default="",
        help="Optional hist_img2txt.jsonl: parent_asin -> description for history items.",
    )
    parser.add_argument("--use_expert_mlp", action="store_true", default=False,
                        help="Load and apply input-type-conditioned expert MLPs.")
    parser.add_argument("--expert_bottleneck_dim", type=int, default=1024,
                        help="Bottleneck dim for expert MLPs (must match training).")
    parser.add_argument("--use_multi_head", action="store_true", default=False,
                        help="Load and apply input-type-specific LM heads.")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                    help="Number of samples to generate per input.")
    args = parser.parse_args()

    main(args)