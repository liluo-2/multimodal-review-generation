#!/bin/bash
# train_mopd.sh — Multi-teacher On-Policy Distillation for RevGen
#
# Usage:
#   bash train_mopd.sh
#   DATASETS=Sports bash train_mopd.sh
#   DATASETS="Arts Sports" bash train_mopd.sh
#
# Optional:
#   STUDENT_LORA=/path/to/student_lora bash train_mopd.sh
#   STUDENT_LORA="" bash train_mopd.sh   # initialize a new random LoRA

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${LLAMA_BASE_MODEL}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${LLAMA_BASE_MODEL}}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/split/revgenfinal}"
CKPT_ROOT="${CKPT_ROOT:-${DATA_ROOT}/checkpoints/revllm}"

DATASETS="${DATASETS:-Arts}"

# Do NOT hard-code WANDB_API_KEY here. Set it outside:
export WANDB_API_KEY=51a08d9271eb398e5202ee2ee1b4344997747225
export WANDB_PROJECT="${WANDB_PROJECT:-RevGenMOPD}"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs  DATASETS=[${DATASETS}]"

# ── Prompt / desc flags ────────────────────────────────────────────────────
USE_DESC="${USE_DESC:-1}"
USE_HIST_IMG_DESC="${USE_HIST_IMG_DESC:-0}"
USE_TARGET_IMG_DESC="${USE_TARGET_IMG_DESC:-1}"

# ── MOPD-specific knobs ────────────────────────────────────────────────────
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"

# ── Training knobs ─────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-25}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MAX_LENGTH="${MAX_LENGTH:-8192}" #8192
OPTIM="${OPTIM:-adamw_torch_fused}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# LoRA knobs only used if STUDENT_LORA is empty
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

NUM_HIST_LIST="${NUM_HIST_LIST:-all}"
DEBUG_PRINT_PROMPTS_N="${DEBUG_PRINT_PROMPTS_N:-15}"

# Slurm-safe master port
if [ -n "${SLURM_JOB_ID:-}" ]; then
  MASTER_PORT="$((20001 + SLURM_JOB_ID % 9999))"
else
  MASTER_PORT="${MASTER_PORT:-29501}"
fi

for DS in ${DATASETS}; do
  DATASET="${DS}"
  DATASET_PATH="${DATASET_PATH:-${MULTIMODAL_DATA_ROOT}/final-dataset/${DATASET}/final.jsonl}"

  # ── Per-dataset teacher + student checkpoints ───────────────────────────
  case "${DATASET}" in
    Arts)
      # Per-sample teacher selection CSV (better_teacher column)
      PER_SAMPLE_DETAIL="${PER_SAMPLE_DETAIL:-${DATA_ROOT}/output/revllm-img2txt/mopd/per_sample_detail_Arts_train.csv}"
      # Single-modality teachers (used when better_teacher == "single")
      TEACHER_BOTH="${TEACHER_BOTH:-${CKPT_ROOT}/Arts-all-hist-both_split-img2txt/checkpoint-2805}"
      TEACHER_IMG_ONLY="${TEACHER_IMG_ONLY:-${CKPT_ROOT}/Arts-all-hist-img-only_split-img2txt/checkpoint-2400}"
      TEACHER_TEXT_ONLY="${TEACHER_TEXT_ONLY:-${CKPT_ROOT}/Arts-all-hist-text-only_split-img2txt/checkpoint-2478}"
      # Mix teacher (used when better_teacher == "mix" or "tie")
      TEACHER_MIX="${TEACHER_MIX:-${CKPT_ROOT}/Arts-all-hist-revllm_mixed_split-mix-img2txt/checkpoint-6818}"

      # Start from base model with a new random LoRA (no pretrained student checkpoint).
      STUDENT_LORA="${STUDENT_LORA:-}"
      ;;
    Sports)
      PER_SAMPLE_DETAIL="${PER_SAMPLE_DETAIL:-${DATA_ROOT}/scripts/mopd/per_sample_detail.csv}"
      TEACHER_BOTH="${TEACHER_BOTH:-${CKPT_ROOT}/Sports-all-hist-both_split-img2txt/checkpoint-TODO}"
      TEACHER_IMG_ONLY="${TEACHER_IMG_ONLY:-${CKPT_ROOT}/Sports-all-hist-img-only_split-img2txt/checkpoint-TODO}"
      TEACHER_TEXT_ONLY="${TEACHER_TEXT_ONLY:-${CKPT_ROOT}/Sports-all-hist-text-only_split-img2txt/checkpoint-TODO}"
      TEACHER_MIX="${TEACHER_MIX:-${CKPT_ROOT}/Sports-all-hist-revllm_mixed_split-mix-img2txt/checkpoint-TODO}"
      STUDENT_LORA="${STUDENT_LORA:-${CKPT_ROOT}/Sports-all-hist-revllm_mixed_split-mix-img2txt/checkpoint-TODO}"
      ;;
    Toys)
      PER_SAMPLE_DETAIL="${PER_SAMPLE_DETAIL:-${DATA_ROOT}/scripts/mopd/per_sample_detail.csv}"
      TEACHER_BOTH="${TEACHER_BOTH:-${CKPT_ROOT}/Toys-all-hist-both_split-img2txt/checkpoint-TODO}"
      TEACHER_IMG_ONLY="${TEACHER_IMG_ONLY:-${CKPT_ROOT}/Toys-all-hist-img-only_split-img2txt/checkpoint-TODO}"
      TEACHER_TEXT_ONLY="${TEACHER_TEXT_ONLY:-${CKPT_ROOT}/Toys-all-hist-text-only_split-img2txt/checkpoint-TODO}"
      TEACHER_MIX="${TEACHER_MIX:-${CKPT_ROOT}/Toys-all-hist-revllm_mixed_split-mix-img2txt/checkpoint-TODO}"
      STUDENT_LORA="${STUDENT_LORA:-${CKPT_ROOT}/Toys-all-hist-revllm_mixed_split-mix-img2txt/checkpoint-TODO}"
      ;;
    *)
      echo "❌ Unknown dataset: ${DATASET} — no teacher checkpoint configured"
      exit 1
      ;;
  esac

  # Validate required paths
  if [ ! -f "${PER_SAMPLE_DETAIL}" ]; then
    echo "❌ Per-sample detail CSV not found: ${PER_SAMPLE_DETAIL}"
    exit 1
  fi
  for _tpath in "${TEACHER_BOTH}" "${TEACHER_IMG_ONLY}" "${TEACHER_TEXT_ONLY}" "${TEACHER_MIX}"; do
    if [ ! -d "${_tpath}" ]; then
      echo "❌ Teacher checkpoint not found: ${_tpath}"
      exit 1
    fi
  done

  if [ -n "${STUDENT_LORA}" ] && [ ! -d "${STUDENT_LORA}" ]; then
    echo "❌ Student LoRA checkpoint not found: ${STUDENT_LORA}"
    echo "   Set STUDENT_LORA='' to initialize a new random LoRA."
    exit 1
  fi

  SPLIT_DEFAULT="${SPLIT_ROOT}/${DATASET}_revllm_mixed_split.npz"
  SPLIT_PATH="${SPLIT_PATH:-${SPLIT_DEFAULT}}"

  IMG_DESC_ROOT="${DATA_ROOT}/output/image_descriptions/${DATASET}"
  IMAGE_DESC_PATHS="${IMAGE_DESC_PATHS:-${IMG_DESC_ROOT}/train_all_hist_img_only_img2txt.jsonl,${IMG_DESC_ROOT}/test_all_hist_img_only_img2txt.jsonl,${IMG_DESC_ROOT}/train_all_hist_both_img2txt.jsonl,${IMG_DESC_ROOT}/test_all_hist_both_img2txt.jsonl}"

  if [ ! -f "${SPLIT_PATH}" ]; then
    echo "❌ Split not found: ${SPLIT_PATH} — skip ${DATASET}"
    continue
  fi

  SPLIT_BASE="$(basename "${SPLIT_PATH}")"
  SPLIT_TAG="${SPLIT_BASE%.npz}"
  SPLIT_TAG="${SPLIT_TAG#${DATASET}_}"

  for NUM_HIST in ${NUM_HIST_LIST}; do
    if [ -n "${STUDENT_LORA}" ]; then
      INIT_TAG="init-existing-lora"
    else
      INIT_TAG="init-new-lora"
    fi

    OUTPUT_DIR="${DATA_ROOT}/checkpoints/revllm/mopd/${DATASET}-${NUM_HIST}-hist-${SPLIT_TAG}-mopd-${INIT_TAG}-base-1e-4-per-sample-inputid"
    RUN_NAME="${DATASET}-revllm-${NUM_HIST}-hist-${SPLIT_TAG}-mopd-${INIT_TAG}-base-1e-4-per-sample-inputid"

    echo ""
    echo "========== DATASET=${DATASET}  NUM_HIST=${NUM_HIST} =========="
    echo "Split:              ${SPLIT_PATH}"
    echo "Student LoRA init:  ${STUDENT_LORA:-<new random LoRA>}"
    echo "Per-sample CSV:     ${PER_SAMPLE_DETAIL}"
    echo "Teacher(both):      ${TEACHER_BOTH}"
    echo "Teacher(img-only):  ${TEACHER_IMG_ONLY}"
    echo "Teacher(text-only): ${TEACHER_TEXT_ONLY}"
    echo "Teacher(mix):       ${TEACHER_MIX}"
    echo "Output:             ${OUTPUT_DIR}"
    echo "LR:                 ${LR}"
    echo "GEN_TEMPERATURE:    ${GEN_TEMPERATURE}"
    echo "KL_TEMPERATURE:     ${KL_TEMPERATURE}"

    cmd=(torchrun
      --nproc_per_node="${NUM_GPUS}"
      --master_port="${MASTER_PORT}"
      train_mopd.py
      --dataset         "${DATASET}"
      --output_dir      "${OUTPUT_DIR}"
      --run_name        "${RUN_NAME}"
      --model_path      "${MODEL_PATH}"
      --tokenizer_path  "${TOKENIZER_PATH}"
      --dataset_path    "${DATASET_PATH}"
      --split_path      "${SPLIT_PATH}"
      --per_sample_detail_path "${PER_SAMPLE_DETAIL}"
      --teacher_both_path      "${TEACHER_BOTH}"
      --teacher_img_only_path  "${TEACHER_IMG_ONLY}"
      --teacher_text_only_path "${TEACHER_TEXT_ONLY}"
      --teacher_mix_path       "${TEACHER_MIX}"
      --batch_size      "${BATCH_SIZE}"
      --gradient_accumulation_steps "${GRAD_ACCUM}"
      --epochs          "${EPOCHS}"
      --learning_rate   "${LR}"
      --warmup_steps    "${WARMUP_STEPS}"
      --max_length      "${MAX_LENGTH}"
      --eval_strategy   "epoch"
      --train_type      "train_random"
      --test_type       "test_random"
      --target_modules  "${TARGET_MODULES}"
      --lora_r          "${LORA_R}"
      --lora_alpha      "${LORA_ALPHA}"
      --lora_dropout    "${LORA_DROPOUT}"
      --num_hist        "${NUM_HIST}"
      --max_new_tokens  "${MAX_NEW_TOKENS}"
      --gen_temperature "${GEN_TEMPERATURE}"
      --kl_temperature  "${KL_TEMPERATURE}"
      --debug_print_prompts_n "${DEBUG_PRINT_PROMPTS_N}"
      --optim           "${OPTIM}"
      --max_grad_norm   "${MAX_GRAD_NORM}"
    )

    if [ -n "${STUDENT_LORA}" ]; then
      cmd+=(--student_lora_path "${STUDENT_LORA}")
    fi

    if [ "${USE_DESC}" = "1" ]; then
      cmd+=(--use_desc)
    else
      cmd+=(--no-use_desc)
    fi

    if [ "${USE_HIST_IMG_DESC}" = "1" ]; then
      cmd+=(--use_hist_img_desc)
    else
      cmd+=(--no-use_hist_img_desc)
    fi

    if [ "${USE_TARGET_IMG_DESC}" = "1" ]; then
      cmd+=(--use_target_img_desc)
    else
      cmd+=(--no-use_target_img_desc)
    fi

    if [ -n "${IMAGE_DESC_PATHS}" ]; then
      cmd+=(--image_desc_paths "${IMAGE_DESC_PATHS}")
    fi

    echo "🚀 ${RUN_NAME}"
    "${cmd[@]}"
    echo "✅ ${RUN_NAME}"
  done
done

echo "🎉 Done $(date)"