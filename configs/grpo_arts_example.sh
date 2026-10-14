#!/bin/bash
# ============================================================
# Example: GRPO training on Arts (mix-SFT → GRPO only)
# Adjust paths per configs/paths.yaml
# ============================================================

export DATASET="Arts"
export DATA_ROOT="/path/to/data"
export SPLIT_ROOT="/path/to/splits"
export MODEL_ROOT="/path/to/models"
export OUTPUT_ROOT="/path/to/outputs"

torchrun --nproc_per_node=2 training/grpo/train_grpo.py \
    --dataset          "${DATASET}" \
    --output_dir       "${OUTPUT_ROOT}/grpo-stg2/${DATASET}" \
    --run_name         "${DATASET}_grpo_only" \
    --model_path       "${MODEL_ROOT}/llama3.1-8b" \
    --tokenizer_path   "${MODEL_ROOT}/llama3.1-8b" \
    --dataset_path     "${DATA_ROOT}/${DATASET}/final.jsonl" \
    --split_path       "${SPLIT_ROOT}/${DATASET}_both_split.npz" \
    --per_sample_detail_path "${OUTPUT_ROOT}/per_sample_detail_${DATASET}.csv" \
    --ref_model_path   "${OUTPUT_ROOT}/sft-mix/${DATASET}/best" \
    --teacher_mix_path       "${OUTPUT_ROOT}/sft-mix/${DATASET}/best" \
    --teacher_both_path      "${OUTPUT_ROOT}/sft-single-both/${DATASET}/best" \
    --teacher_img_only_path  "${OUTPUT_ROOT}/sft-single-img/${DATASET}/best" \
    --teacher_text_only_path "${OUTPUT_ROOT}/sft-single-text/${DATASET}/best" \
    --image_desc_paths "${DATA_ROOT}/image_descriptions/${DATASET}/train_all_hist_both_img2txt.jsonl,${DATA_ROOT}/image_descriptions/${DATASET}/train_all_hist_img_only_img2txt.jsonl" \
    --loss_mode        "grpo_only" \
    --reward_metric    "composite" \
    --batch_size       1 \
    --gradient_accumulation_steps 4 \
    --epochs           10 \
    --learning_rate    5e-7 \
    --warmup_steps     100 \
    --max_length       8192 \
    --num_rollouts     4 \
    --max_new_tokens   256 \
    --gen_temperature  1.0 \
    --reward_threshold 0.3 \
    --clip_epsilon     0.2 \
    --kl_beta          0.01 \
    --num_hist         5 \
    --train_type       "train_random" \
    --test_type        "test_random" \
    --eval_strategy    "epoch" \
    --save_total_limit 5 \
    --logging_steps    10 \
    --lora_r           8 \
    --lora_alpha       16 \
    --target_modules   "q_proj,v_proj"
