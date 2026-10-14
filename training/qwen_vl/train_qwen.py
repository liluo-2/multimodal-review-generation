# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import re
import json
import torch
import transformers
import sys
from pathlib import Path
from copy import deepcopy

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoConfig,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _checkpoint_step(path_obj: pathlib.Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path_obj.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(output_dir: str) -> pathlib.Path | None:
    checkpoints = list(pathlib.Path(output_dir).glob("checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step)


def _is_torch_load_safety_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "serious vulnerability issue in `torch.load`" in msg
        and "upgrade torch to at least v2.6" in msg
    )


def _fallback_resume_hints(checkpoint_dir: pathlib.Path) -> tuple[float, float | None]:
    """Return (completed_epoch, last_lr) from trainer_state.json if present."""
    state_file = checkpoint_dir / "trainer_state.json"
    if not state_file.exists():
        return 0.0, None
    try:
        state = json.loads(state_file.read_text())
    except Exception:
        return 0.0, None

    completed_epoch = float(state.get("epoch") or 0.0)
    last_lr = None
    for item in reversed(state.get("log_history", [])):
        if isinstance(item, dict) and "learning_rate" in item:
            try:
                last_lr = float(item["learning_rate"])
                break
            except Exception:
                pass
    return completed_epoch, last_lr


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_path = model_args.model_name_or_path
    model_path_lower = model_path.lower()
    cfg = AutoConfig.from_pretrained(model_path, cache_dir=training_args.cache_dir)
    cfg_model_type = str(getattr(cfg, "model_type", "")).lower()
    cfg_arch = " ".join(getattr(cfg, "architectures", []) or []).lower()

    is_qwen3_moe = ("qwen3" in cfg_model_type and "moe" in cfg_model_type) or ("qwen3" in cfg_arch and "moe" in cfg_arch)
    is_qwen3 = ("qwen3" in cfg_model_type) or ("qwen3" in cfg_arch)
    is_qwen25 = (
        "qwen2_5" in cfg_model_type
        or "qwen2.5" in cfg_model_type
        or "qwen2_5" in cfg_arch
        or "qwen2.5" in cfg_arch
    )

    # Fallback to path-name hints only when config cannot identify model family.
    if not (is_qwen3_moe or is_qwen3 or is_qwen25):
        is_qwen3_moe = ("qwen3" in model_path_lower and "moe" in model_path_lower)
        is_qwen3 = "qwen3" in model_path_lower
        is_qwen25 = "qwen2.5" in model_path_lower or "qwen2_5" in model_path_lower

    if is_qwen3_moe:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif is_qwen3:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif is_qwen25:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    # data_module = make_supervised_data_module(processor, data_args=data_args)
    # ========== Build eval_data_args ==========
    eval_data_args = None
    if data_args.eval_dataset_use is not None and training_args.eval_strategy != "no":
        eval_names = [name.strip() for name in data_args.eval_dataset_use.split(",") if name.strip()]
        if len(eval_names) <= 1:
            eval_data_args = deepcopy(data_args)
            eval_data_args.dataset_use = data_args.eval_dataset_use
            rank0_print(f"Eval dataset enabled: {data_args.eval_dataset_use}")
        else:
            # Multiple eval datasets will be logged separately in Trainer/W&B:
            # eval_<dataset_key>_loss
            eval_data_args = {}
            for eval_name in eval_names:
                one_eval_args = deepcopy(data_args)
                one_eval_args.dataset_use = eval_name
                eval_data_args[eval_name] = one_eval_args
            rank0_print(f"Per-modality eval enabled: {eval_names}")
    # ============================================
    
    data_module = make_supervised_data_module(processor, data_args=data_args, eval_data_args=eval_data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    latest_checkpoint = _latest_checkpoint(training_args.output_dir)
    if latest_checkpoint is not None:
        logging.info(f"checkpoint found ({latest_checkpoint}), resume training")
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError as e:
            if not _is_torch_load_safety_error(e):
                raise
            # Fallback for torch<2.6: continue from latest LoRA weights
            # without loading optimizer/scheduler states.
            completed_epoch, last_lr = _fallback_resume_hints(latest_checkpoint)
            remaining_epochs = max(float(training_args.num_train_epochs) - completed_epoch, 0.0)
            if remaining_epochs <= 0.0:
                rank0_print(
                    f"Fallback resume detected but no remaining epochs "
                    f"(completed={completed_epoch:.4f}). Skip training."
                )
                trainer.save_state()
                model.config.use_cache = True
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
                processor.save_pretrained(training_args.output_dir)
                return

            rank0_print(
                "Resume blocked by torch.load safety check; "
                "fallback to loading latest LoRA weights only."
            )
            rank0_print(
                f"Fallback resume hints: completed_epoch={completed_epoch:.4f}, "
                f"remaining_epochs={remaining_epochs:.4f}, last_lr={last_lr}"
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                trainer.model.get_base_model(),
                str(latest_checkpoint),
                is_trainable=True,
            )
            training_args.num_train_epochs = remaining_epochs
            if last_lr is not None and last_lr > 0:
                training_args.learning_rate = last_lr
            training_args.warmup_ratio = 0.0
            training_args.warmup_steps = 0
            model.train()

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                **data_module,
            )
            trainer.train()
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
