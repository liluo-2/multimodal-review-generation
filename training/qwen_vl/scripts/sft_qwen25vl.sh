#!/bin/bash
# Usage: bash sft_30a3b_lora.sh <config_name>
# config_name 可选: arts_both arts_img_only arts_text_only toys_both toys_img_only toys_text_only

CONFIG_NAME=${1:?"用法: bash sft_30a3b_lora.sh <config_name>  (arts_both|arts_img_only|arts_text_only|toys_both|toys_img_only|toys_text_only)"}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

deepspeed=./scripts/zero2.json
llm=Qwen/Qwen3-VL-8B-Instruct

lr=5e-6
batch_size=1
grad_accum_steps=4

entry_file=qwenvl/train/train_qwen.py
CKPT_BASE=${SCRATCH_ROOT}/Qwen2.5-VL/ckpt

export WANDB_API_KEY=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
export WANDB_PROJECT="qwen3vl_revgen"

declare -A TRAIN_DS_MAP=(
  [arts_both]=train_arts
  [arts_img_only]=train_arts_img_only
  [arts_text_only]=train_arts_text_only
  [arts_mix]=train_arts_all_hist_mix        # ← 新增
  [toys_both]=train_toys
  [toys_img_only]=train_toys_img_only
  [toys_text_only]=train_toys_text_only
  [toys_mix]=train_toys_all_hist_mix        # ← 新增
)
declare -A EVAL_DS_MAP=(
  [arts_both]=eval_arts
  [arts_img_only]=eval_arts_img_only
  [arts_text_only]=eval_arts_text_only
  [arts_mix]=eval_arts_all_hist_mix         # ← 新增
  [toys_both]=eval_toys
  [toys_img_only]=eval_toys_img_only
  [toys_text_only]=eval_toys_text_only
  [toys_mix]=eval_toys_all_hist_mix         # ← 新增
)

if [[ -z "${TRAIN_DS_MAP[$CONFIG_NAME]}" ]]; then
  echo "❌ 未知 config_name: ${CONFIG_NAME}"
  echo "可选: ${!TRAIN_DS_MAP[@]}"
  exit 1
fi

datasets="${TRAIN_DS_MAP[$CONFIG_NAME]}"
eval_datasets="${EVAL_DS_MAP[$CONFIG_NAME]}"
run_suffix="qwen3-8b-${CONFIG_NAME//_/-}"
run_name="${run_suffix}"
output_dir="${CKPT_BASE}/${run_suffix}"

echo "=========================================="
echo "Config:   $CONFIG_NAME"
echo "Model:    $llm"
echo "Training: $datasets / $eval_datasets"
echo "Output:   $output_dir"
echo "=========================================="

args="
  --deepspeed ${deepspeed} \
  --model_name_or_path ${llm} \
  --dataset_use ${datasets} \
  --eval_dataset_use ${eval_datasets} \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 True \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --output_dir ${output_dir} \
  --num_train_epochs 25 \
  --per_device_train_batch_size ${batch_size} \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps ${grad_accum_steps} \
  --max_pixels $((1280 * 28 * 28)) \
  --min_pixels $((256 * 28 * 28)) \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 20 \
  --learning_rate ${lr} \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 32768 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --run_name ${run_name} \
  --report_to wandb"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

echo "✅ Done: ${CONFIG_NAME}"