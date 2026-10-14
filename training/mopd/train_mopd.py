"""
train_mopd.py — Multi-teacher On-Policy Distillation training script.

Supports two student initialization modes:

1. New LoRA:
   base LLaMA + newly initialized LoRA adapter -> OPD

2. Continue from existing LoRA:
   base LLaMA + existing trainable LoRA checkpoint -> OPD

Usage:
    torchrun --nproc_per_node=N train_mopd.py [args]
"""

import csv
import os
import argparse
import json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import wandb

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments

from Datacollator import DataCollatorForSupervisedDataset
from MOPDTrainer import MOPDTrainer
from SFTDataset import RevGenDataset, load_hist_img_desc_map


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0



# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_descriptions(paths_csv: str):
    image_desc_map, img_only_ids = {}, set()
    if not paths_csv:
        return image_desc_map, img_only_ids

    for path in [p.strip() for p in paths_csv.split(",") if p.strip()]:
        is_img_only = "img_only" in path.lower()

        if not os.path.isfile(path):
            print(f"[Warning] image description file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)
                rec_id = str(rec.get("id", "")).strip()
                if not rec_id:
                    continue

                descs = rec.get("image_descriptions") or []
                desc = ""
                if isinstance(descs, list) and descs:
                    desc = str((descs[0] or {}).get("description", "")).strip()

                if desc:
                    image_desc_map[rec_id] = desc

                if is_img_only:
                    img_only_ids.add(rec_id)

    return image_desc_map, img_only_ids


# ─────────────────────────────────────────────────────────────────────────────
# Student loader
# ─────────────────────────────────────────────────────────────────────────────

def load_student_model(args, tokenizer, is_distributed: bool, is_main: bool):
    """
    Load student model.

    If args.student_lora_path is provided:
        base LLaMA + existing LoRA checkpoint, trainable=True

    Else:
        base LLaMA + newly initialized LoRA adapter
    """
    device_map = None if is_distributed else "auto"

    student_base = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )

    if student_base.get_input_embeddings().num_embeddings != len(tokenizer):
        student_base.resize_token_embeddings(len(tokenizer))

    if args.student_lora_path:
        if not os.path.isdir(args.student_lora_path):
            raise FileNotFoundError(
                f"--student_lora_path does not exist: {args.student_lora_path}"
            )

        student = PeftModel.from_pretrained(
            student_base,
            args.student_lora_path,
            is_trainable=True,
        )

        if is_main:
            print(f"[MOPD] Loaded trainable student LoRA from: {args.student_lora_path}")

    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        student = get_peft_model(student_base, lora_cfg)

        if is_main:
            print("[MOPD] Initialized a new LoRA adapter")

    if is_main:
        student.print_trainable_parameters()

    return student


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = setup_distributed()

    is_main = rank == 0
    is_distributed = world_size > 1

    if is_main:
        print(f"[MOPD] distributed={is_distributed}  world_size={world_size}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.padding_side = "right"

    # Important: avoid pad_token == eos_token, because OPD response masking
    # needs to distinguish real EOS from padding.
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ── Student model ─────────────────────────────────────────────────────────
    student = load_student_model(
        args=args,
        tokenizer=tokenizer,
        is_distributed=is_distributed,
        is_main=is_main,
    )

    # ── Per-sample better_teacher map (from CSV) ──────────────────────────────
    better_teacher_map: dict[str, str] = {}
    with open(args.per_sample_detail_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"].strip() == args.dataset:
                better_teacher_map[row["id"].strip()] = row["better_teacher"].strip()
    if is_main:
        print(f"[MOPD] loaded {len(better_teacher_map)} per-sample teacher entries from CSV")

    # ── Teachers ──────────────────────────────────────────────────────────────
    # Teacher paths — loaded inside MOPDTrainer after student is placed on GPU.
    teacher_paths = {
        "mix":              args.teacher_mix_path,
        "both_single":      args.teacher_both_path,
        "img_only_single":  args.teacher_img_only_path,
        "text_only_single": args.teacher_text_only_path,
    }

    # ── Datasets ──────────────────────────────────────────────────────────────
    split_data = np.load(args.split_path, allow_pickle=False)
    split = dict(split_data)

    train_ids = split[args.train_type]
    val_ids = split[args.test_type]

    image_desc_map, img_only_ids = load_image_descriptions(args.image_desc_paths)
    hist_img_desc_map = load_hist_img_desc_map(args.hist_img_desc_path)

    if not args.use_hist_img_desc:
        hist_img_desc_map = {}

    if not args.use_target_img_desc:
        image_desc_map = {}
        img_only_ids = set()

    if is_main:
        print(
            f"[MOPD] use_desc={args.use_desc}  "
            f"use_hist_img_desc={args.use_hist_img_desc}  "
            f"use_target_img_desc={args.use_target_img_desc}  "
            f"target_imgs={len(image_desc_map)}  img_only={len(img_only_ids)}"
        )

    dataset_common = dict(
        jsonl_path=args.dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        has_desc=args.use_desc,
        num_hist=args.num_hist,
        image_desc_map=image_desc_map,
        img_only_ids=img_only_ids,
        hist_img_desc_map=hist_img_desc_map,
        better_teacher_map=better_teacher_map,
    )

    train_dataset = RevGenDataset(
        split_ids=train_ids,
        is_opd=True,
        is_train=True,
        debug_print_prompts_n=args.debug_print_prompts_n if is_main else 0,
        **dataset_common,
    )

    eval_dataset = RevGenDataset(
        split_ids=val_ids,
        is_opd=False,
        is_train=True,
        debug_print_prompts_n=0,
        **dataset_common,
    )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join("logs", args.run_name),
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_strategy=args.eval_strategy,
        fp16=not use_bf16,
        bf16=use_bf16,
        save_total_limit=5,
        load_best_model_at_end=True if args.eval_strategy != "no" else False,
        metric_for_best_model="eval_loss" if args.eval_strategy != "no" else None,
        greater_is_better=False if args.eval_strategy != "no" else None,
        report_to="wandb",
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        optim=args.optim,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = MOPDTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        teacher_paths=teacher_paths,
        mopd_tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.gen_temperature,
        kl_temperature=args.kl_temperature,
    )

    trainer.train()

    if args.eval_strategy != "no":
        trainer.evaluate()

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if is_main:
        print(f"[MOPD] training finished. Saved to: {args.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MOPD training for RevGen")

    # Required
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--split_path", type=str, required=True)

    p.add_argument(
        "--per_sample_detail_path",
        type=str,
        required=True,
        help="CSV with per-sample better_teacher column (id, modality, better_teacher, ...).",
    )
    p.add_argument(
        "--teacher_mix_path",
        type=str,
        required=True,
        help="Teacher checkpoint for samples where mix is better (or tie).",
    )
    p.add_argument(
        "--teacher_both_path",
        type=str,
        required=True,
        help="Teacher checkpoint for 'both' modality samples where single is better.",
    )
    p.add_argument(
        "--teacher_img_only_path",
        type=str,
        required=True,
        help="Teacher checkpoint for 'img-only' modality samples where single is better.",
    )
    p.add_argument(
        "--teacher_text_only_path",
        type=str,
        required=True,
        help="Teacher checkpoint for 'text-only' modality samples where single is better.",
    )

    # Student initialization
    p.add_argument(
        "--student_lora_path",
        type=str,
        default="",
        help=(
            "Optional existing LoRA checkpoint used to initialize the student. "
            "If empty, a new LoRA adapter is initialized."
        ),
    )

    # Training hyperparams
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )

    # LoRA config used only when --student_lora_path is empty
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # MOPD-specific
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate per on-policy rollout.",
    )
    p.add_argument(
        "--gen_temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for on-policy student generation.",
    )
    p.add_argument(
        "--kl_temperature",
        type=float,
        default=1.0,
        help="Temperature applied before KL. 1.0 means no scaling.",
    )

    # Split / data types
    p.add_argument("--train_type", type=str, default="train_random")
    p.add_argument("--test_type", type=str, default="test_random")
    p.add_argument("--num_hist", type=str, default="all")
    p.add_argument("--image_desc_paths", type=str, default="")
    p.add_argument("--hist_img_desc_path", type=str, default="")
    p.add_argument("--debug_print_prompts_n", type=int, default=0)

    # Prompt flags
    p.set_defaults(
        use_desc=True,
        use_hist_img_desc=False,
        use_target_img_desc=True,
    )

    p.add_argument("--use_desc", dest="use_desc", action="store_true")
    p.add_argument("--no-use_desc", dest="use_desc", action="store_false")

    p.add_argument("--use_hist_img_desc", dest="use_hist_img_desc", action="store_true")
    p.add_argument("--no-use_hist_img_desc", dest="use_hist_img_desc", action="store_false")

    p.add_argument("--use_target_img_desc", dest="use_target_img_desc", action="store_true")
    p.add_argument("--no-use_target_img_desc", dest="use_target_img_desc", action="store_false")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        wandb.init(project="RevGenMOPD", name=args.run_name)

    try:
        main(args)
    finally:
        if is_main:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()