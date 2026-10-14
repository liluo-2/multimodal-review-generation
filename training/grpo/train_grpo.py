"""
train_grpo.py — Stage 2: GRPO + conditional KD/CE training.

Loads student from Stage 1 OPD checkpoint, then trains with GRPO.

Usage:
    torchrun --nproc_per_node=N train_grpo.py [args]
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

from Datacollator_grpo import GRPODataCollator
from GRPOTrainer_cider import GRPOTrainer, load_teacher_rewards
from SFTDataset import RevGenDataset, load_hist_img_desc_map


# ─────────────────────────────────────────────────────────────────────────────
# Distributed
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
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


def load_gt_texts(jsonl_path: str, split_ids) -> dict:
    """Load ground truth texts for CE loss fallback. Returns {str(id): text}."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f]
    result = {}
    for sid in split_ids:
        sid_int = int(sid)
        entry = all_data[sid_int]
        ref_text = entry.get("reference", {}).get("text", "")
        result[str(sid_int)] = ref_text
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GRPORevGenDataset(torch.utils.data.Dataset):
    """
    Wraps RevGenDataset for GRPO training.
    Returns both:
    - prompt-only input_ids (for rollout generation) via is_opd=True
    - full input_ids + labels (for CE loss fallback) via is_opd=False
    - sample_id for teacher reward lookup
    """

    def __init__(self, sft_dataset: RevGenDataset, opd_dataset: RevGenDataset):
        """
        sft_dataset: is_opd=False, is_train=True → returns input_ids + labels
        opd_dataset: is_opd=True,  is_train=True → returns input_ids (prompt only)
        """
        self.sft_ds = sft_dataset
        self.opd_ds = opd_dataset
        assert len(self.sft_ds) == len(self.opd_ds)

    def __len__(self):
        return len(self.sft_ds)

    def __getitem__(self, idx):
        sft_item = self.sft_ds[idx]   # has input_ids, attention_mask, labels, input_type
        opd_item = self.opd_ds[idx]   # has input_ids (prompt only), attention_mask, input_type

        sample_id = str(self.sft_ds.split_ids[idx])

        return {
            # Prompt-only (for rollout generation)
            "input_ids":      opd_item["input_ids"],
            "attention_mask": opd_item["attention_mask"],
            # Full (for CE loss)
            "full_input_ids":      sft_item["input_ids"],
            "full_attention_mask": sft_item["attention_mask"],
            "labels":              sft_item["labels"],
            # Routing
            "input_type":  opd_item["input_type"],
            "modality":    opd_item["modality"],
            "sample_id":   sample_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Student loader
# ─────────────────────────────────────────────────────────────────────────────

def load_student_model(args, tokenizer, is_distributed: bool, is_main: bool):
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
        student = PeftModel.from_pretrained(
            student_base, args.student_lora_path, is_trainable=True
        )
        if is_main:
            print(f"[GRPO] Loaded student LoRA from Stage 1: {args.student_lora_path}")
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(","),
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        )
        student = get_peft_model(student_base, lora_cfg)
        if is_main:
            print("[GRPO] Initialized new LoRA (no Stage 1 init)")

    if is_main:
        student.print_trainable_parameters()

    # ── PPO value head (shared backbone) — only when loss_mode == "ppo" ────────
    # Attached as a submodule of `student` so it lands in model.parameters()
    # (HF Trainer's optimizer picks it up) and is moved to GPU with the model.
    # Created only for PPO so GRPO runs don't carry an unused, un-synced param
    # (which would trip DDP's find_unused_parameters=False).
    if getattr(args, "loss_mode", "grpo_kd_ce") in ("ppo", "ppo_kd_ce"):
        hidden_size = student_base.config.hidden_size
        value_head = torch.nn.Linear(hidden_size, 1, bias=True).to(dtype=torch.bfloat16)
        torch.nn.init.normal_(value_head.weight, std=1.0 / (hidden_size + 1) ** 0.5)
        torch.nn.init.zeros_(value_head.bias)
        student.value_head = value_head

        vh_path = os.path.join(args.resume_from_checkpoint, "value_head.pt") \
            if args.resume_from_checkpoint else ""
        if vh_path and os.path.isfile(vh_path):
            value_head.load_state_dict(torch.load(vh_path, map_location="cpu"))
            if is_main:
                print(f"[PPO] Loaded value head from {vh_path}")
        elif is_main:
            print("[PPO] Initialized new value head")

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
        print(f"[GRPO] distributed={is_distributed}  world_size={world_size}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ── Teacher rewards + per-sample teacher routing ──────────────────────────
    # Load teacher rewards first; better_teacher is derived from the selected
    # reward_metric (single_score vs mix_score) so routing is always consistent.
    teacher_rewards, cider_min, cider_max = load_teacher_rewards(
        args.per_sample_detail_path, args.dataset, args.reward_metric
    )
    better_teacher_map = {sid: info["better_teacher"] for sid, info in teacher_rewards.items()}
    if is_main:
        print(f"[GRPO] loaded {len(better_teacher_map)} per-sample teacher entries "
              f"(metric={args.reward_metric})")

    metric_stats = None
    if args.metric_stats_path:
        with open(args.metric_stats_path, "r", encoding="utf-8") as f:
            metric_stats = json.load(f)
        if is_main:
            print(f"[GRPO] loaded metric_stats from {args.metric_stats_path}: "
                  f"{list(metric_stats.keys())}")

    # ── Student ───────────────────────────────────────────────────────────────
    student = load_student_model(args, tokenizer, is_distributed, is_main)

    # ── Teacher paths ─────────────────────────────────────────────────────────
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

    # GT texts for CE loss
    gt_texts = load_gt_texts(args.dataset_path, train_ids)
    if is_main:
        print(f"[GRPO] loaded {len(gt_texts)} GT texts for CE fallback")

    # Trainset references for CIDEr IDF (all train GT texts as a list)
    trainset_references = list(gt_texts.values())

    # Eval GT texts (for the per-epoch eval-reward pass); merge into gt_texts so
    # the trainer can look up references by sample_id for both train and eval.
    eval_gt_texts = load_gt_texts(args.dataset_path, val_ids)
    gt_texts = {**gt_texts, **eval_gt_texts}

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

    # Train: two versions of the dataset (SFT mode for labels + OPD mode for prompt-only)
    train_sft = RevGenDataset(
        split_ids=train_ids, is_train=True, is_opd=False,
        debug_print_prompts_n=args.debug_print_prompts_n if is_main else 0,
        **dataset_common,
    )
    train_opd = RevGenDataset(
        split_ids=train_ids, is_train=True, is_opd=True,
        debug_print_prompts_n=0,
        **dataset_common,
    )
    train_dataset = GRPORevGenDataset(train_sft, train_opd)

    # Eval: standard SFT mode (for CE eval_loss)
    eval_dataset = RevGenDataset(
        split_ids=val_ids, is_train=True, is_opd=False,
        debug_print_prompts_n=0,
        **dataset_common,
    )

    # Eval (reward): prompt-only version so we can generate on the eval set and
    # score against GT. Wrapped like the train set so it carries sample_id.
    eval_opd = RevGenDataset(
        split_ids=val_ids, is_train=True, is_opd=True,
        debug_print_prompts_n=0,
        **dataset_common,
    )
    eval_reward_dataset = GRPORevGenDataset(eval_dataset, eval_opd)

    data_collator = GRPODataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    from Datacollator import DataCollatorForSupervisedDataset
    eval_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, pad_to_multiple_of=8)

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
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        load_best_model_at_end=args.eval_strategy != "no",
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
    trainer = GRPOTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # Stage 2 specific
        ref_model_path=args.ref_model_path,
        teacher_paths=teacher_paths,
        grpo_tokenizer=tokenizer,
        teacher_rewards=teacher_rewards,
        gt_texts=gt_texts,
        trainset_references=trainset_references,   # NEW: for CIDEr IDF
        cider_min=cider_min,                       # NEW: from load_teacher_rewards
        cider_max=cider_max,                       # NEW: from load_teacher_rewards
        metric_stats=metric_stats,
        reward_metric=args.reward_metric,
        # GRPO hyperparams
        num_rollouts=args.num_rollouts,
        num_grpo_epochs=args.num_grpo_epochs,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.gen_temperature,
        reward_threshold=args.reward_threshold,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        kd_lambda=args.kd_lambda,
        ce_gamma=args.ce_gamma,
        kl_temperature=args.kl_temperature,
        loss_mode=args.loss_mode,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        cliprange_value=args.cliprange_value,
        eval_reward_batch_size=args.eval_reward_batch_size,
    )

    # Override eval collator
    trainer.eval_data_collator = eval_collator

    # Per-epoch eval-reward pass (full eval set, single generation per prompt)
    trainer.reward_eval_dataset = eval_reward_dataset

    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume)
    if args.eval_strategy != "no":
        trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if is_main:
        print(f"[GRPO] Done. Saved to: {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset",        type=str, required=True)
    p.add_argument("--output_dir",     type=str, required=True)
    p.add_argument("--run_name",       type=str, required=True)
    p.add_argument("--model_path",     type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--dataset_path",   type=str, required=True)
    p.add_argument("--split_path",     type=str, required=True)

    # Per-sample CSV
    p.add_argument("--per_sample_detail_path", type=str, required=True)

    # Reference model (stage 1 OPD best ckpt)
    p.add_argument("--ref_model_path", type=str, required=True)

    # Student init (stage 1 OPD best ckpt, trainable)
    p.add_argument("--student_lora_path", type=str, default="")

    # Teacher checkpoints
    p.add_argument("--teacher_mix_path",       type=str, required=True)
    p.add_argument("--teacher_both_path",      type=str, required=True)
    p.add_argument("--teacher_img_only_path",  type=str, required=True)
    p.add_argument("--teacher_text_only_path", type=str, required=True)

    # Training
    p.add_argument("--batch_size",                  type=int,   default=1)
    p.add_argument("--eval_reward_batch_size",      type=int,   default=8,
                   help="Batch size for the per-epoch eval-reward generation pass "
                        "(no grad, so can be larger than train batch_size).")
    p.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    p.add_argument("--epochs",                      type=int,   default=10)
    p.add_argument("--learning_rate",               type=float, default=5e-7)
    p.add_argument("--warmup_steps",                type=int,   default=100)
    p.add_argument("--max_length",                  type=int,   default=8192)
    p.add_argument("--optim",                       type=str,   default="adamw_torch_fused")
    p.add_argument("--logging_steps",               type=int,   default=10)
    p.add_argument("--max_grad_norm",               type=float, default=1.0)
    p.add_argument("--eval_strategy",               type=str,   default="epoch")
    p.add_argument("--save_total_limit",            type=int,   default=5,
                   help="Max number of checkpoints to keep (oldest deleted first). "
                        "0 or negative = keep all. best model is kept on top when "
                        "load_best_model_at_end is on.")

    # LoRA
    p.add_argument("--target_modules", type=str,   default="q_proj,v_proj")
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--lora_dropout",   type=float, default=0.05)

    # GRPO
    p.add_argument("--reward_metric", type=str, default="composite",
                   choices=["composite", "rouge1", "rouge2", "rougel",
                            "bleu", "bertscore", "cider"],
                   help="Which metric to use as GRPO reward signal")
    p.add_argument("--metric_stats_path", type=str, default="",
                   help="Path to metric_stats JSON from eval_per_sample.py "
                        "(enables per-metric normalization for composite reward)")

    p.add_argument("--loss_mode", type=str, default="grpo_kd_ce",
                   choices=["ppo", "ppo_kd_ce", "grpo_only", "grpo_kd_ce"],
                   help="'ppo' = classic PPO (value head + per-token GAE, KL folded "
                        "into reward, teachers not loaded); "
                        "'ppo_kd_ce' = PPO + per-sample conditional KD/CE "
                        "(teachers loaded, same routing as grpo_kd_ce); "
                        "'grpo_only' = pure GRPO loss (L_GRPO with KL-to-ref only, "
                        "teachers not loaded); 'grpo_kd_ce' = current behavior "
                        "(GRPO loss + per-sample conditional KD/CE).")

    # PPO-only hyperparams (ignored unless loss_mode=ppo)
    p.add_argument("--gamma",           type=float, default=1.0,
                   help="PPO discount factor (1.0 is typical for seq-level reward).")
    p.add_argument("--gae_lambda",      type=float, default=0.95,
                   help="PPO GAE lambda.")
    p.add_argument("--vf_coef",         type=float, default=0.5,
                   help="PPO value-loss coefficient.")
    p.add_argument("--cliprange_value", type=float, default=0.2,
                   help="PPO value-clipping range.")

    p.add_argument("--num_rollouts",      type=int,   default=4)
    p.add_argument("--num_grpo_epochs",   type=int,   default=1,
                   help="Gradient updates per rollout batch. >1 makes PPO clip active. "
                        "Set gradient_accumulation_steps=1 when using >1.")
    p.add_argument("--max_new_tokens",    type=int,   default=256)
    p.add_argument("--gen_temperature",   type=float, default=1.0)
    p.add_argument("--reward_threshold",  type=float, default=0.3)
    p.add_argument("--clip_epsilon",      type=float, default=0.2)
    p.add_argument("--kl_beta",           type=float, default=0.01)
    p.add_argument("--kd_lambda",         type=float, default=0.5)
    p.add_argument("--ce_gamma",          type=float, default=0.5)
    p.add_argument("--kl_temperature",    type=float, default=1.0)

    # Data
    p.add_argument("--train_type",            type=str, default="train_random")
    p.add_argument("--test_type",             type=str, default="test_random")
    p.add_argument("--num_hist",              type=str, default="all")
    p.add_argument("--image_desc_paths",      type=str, default="")
    p.add_argument("--hist_img_desc_path",    type=str, default="")
    p.add_argument("--debug_print_prompts_n", type=int, default=0)
    p.add_argument("--resume_from_checkpoint", type=str, default="",
                   help="Path to a checkpoint dir to resume from (restores optimizer + LR scheduler). Empty = fresh start.")

    p.set_defaults(use_desc=True, use_hist_img_desc=False, use_target_img_desc=True)
    p.add_argument("--use_desc",               dest="use_desc",            action="store_true")
    p.add_argument("--no-use_desc",            dest="use_desc",            action="store_false")
    p.add_argument("--use_hist_img_desc",      dest="use_hist_img_desc",   action="store_true")
    p.add_argument("--no-use_hist_img_desc",   dest="use_hist_img_desc",   action="store_false")
    p.add_argument("--use_target_img_desc",    dest="use_target_img_desc", action="store_true")
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
        wandb.init(project="RevGenGRPO", name=args.run_name)

    try:
        main(args)
    finally:
        if is_main:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()