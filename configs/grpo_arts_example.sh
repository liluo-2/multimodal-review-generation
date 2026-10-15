#!/bin/bash
# ============================================================
# Example: GRPO training on Arts, starting from the mix-SFT model (no Stage-1 distillation)
# Matches the reported run: union of three scenarios, batch 2, 40-epoch schedule;
# the reported checkpoint is step 3147 (epoch 12), highest reward on the evaluation split.
# NOTE: --test_type test_random makes the evaluation split the test split. To select
# checkpoints without touching the test set, hold out part of train_random as validation.
# Adjust paths per configs/paths.yaml
# ============================================================

export DATASET="Arts"
export DATA_ROOT="/path/to/data"
export SPLIT_ROOT="/path/to/splits"
export MODEL_ROOT="/path/to/models"
export OUTPUT_ROOT="/path/to/outputs"

# Build the mixed split (union of the three scenario splits) once:
#   python data/make_mix_split.py --split_dir ${SPLIT_ROOT} --dataset ${DATASET}
# Scenario per sample is inferred from --image_desc_paths (img-only file -> img_only;
# other ids with an image description -> both; the rest -> text_only).

torchrun --nproc_per_node=1 training/grpo/train_grpo.py \
    --dataset          "${DATASET}" \
    --output_dir       "${OUTPUT_ROOT}/grpo-stg2/${DATASET}" \
    --run_name         "${DATASET}_grpo_only" \
    --model_path       "${MODEL_ROOT}/llama3.1-8b" \
    --tokenizer_path   "${MODEL_ROOT}/llama3.1-8b" \
    --student_lora_path "${OUTPUT_ROOT}/sft-mix/${DATASET}/best" \
    --dataset_path     "${DATA_ROOT}/${DATASET}/final.jsonl" \
    --split_path       "${SPLIT_ROOT}/${DATASET}_mix_split.npz" \
    --per_sample_detail_path "${OUTPUT_ROOT}/per_sample_detail_${DATASET}.csv" \
    --ref_model_path   "${OUTPUT_ROOT}/sft-mix/${DATASET}/best" \
    --teacher_mix_path       "${OUTPUT_ROOT}/sft-mix/${DATASET}/best" \
    --teacher_both_path      "${OUTPUT_ROOT}/sft-single-both/${DATASET}/best" \
    --teacher_img_only_path  "${OUTPUT_ROOT}/sft-single-img/${DATASET}/best" \
    --teacher_text_only_path "${OUTPUT_ROOT}/sft-single-text/${DATASET}/best" \
    --image_desc_paths "${DATA_ROOT}/image_descriptions/${DATASET}/train_all_hist_both_img2txt.jsonl,${DATA_ROOT}/image_descriptions/${DATASET}/train_all_hist_img_only_img2txt.jsonl" \
    --loss_mode        "grpo_only" \
    --reward_metric    "composite" \
    --batch_size       2 \
    --gradient_accumulation_steps 4 \
    --epochs           40 \
    --learning_rate    5e-6 \
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
