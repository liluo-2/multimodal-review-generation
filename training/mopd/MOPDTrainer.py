"""
MOPDTrainer.py — Multi-teacher On Policy Distillation (MOPD)

Objective (Reverse KL, on-policy):
    L(θ) = E_{x~D, ŷ~π_θ(·|x)} [ Σ_t D_KL(π_θ(·|x,ŷ_{<t}) ‖ π_teacher_m(x)(·|x,ŷ_{<t})) ]

Reverse KL:  D_KL(Q‖P) = E_{x~Q}[log Q(x)/P(x)]
    Q = student  (π_θ)
    P = teacher  (π_expert)

Per-sample teacher routing (via input_type field set in RevGenDataset):
    "mix"              →  mix teacher  (better_teacher == "mix" or "tie")
    "both_single"      →  both single teacher  (both modality, better_teacher == "single")
    "img_only_single"  →  img-only single teacher
    "text_only_single" →  text-only single teacher
"""

import os
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM, Trainer, TrainerCallback


# ─────────────────────────────────────────────────────────────────────────────
# Teacher loader
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher(ckpt_path: str, tokenizer, local_rank: int = 0, bf16: bool = True) -> LlamaForCausalLM:
    peft_cfg = PeftConfig.from_pretrained(ckpt_path)
    base_path = peft_cfg.base_model_name_or_path
    device = f"cuda:{local_rank}"

    base = LlamaForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))

    teacher = PeftModel.from_pretrained(base, ckpt_path)
    teacher = teacher.merge_and_unload()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[Teacher] loaded (bf16 + FA2, merged) & frozen on {device}: {ckpt_path}")
    return teacher


# ─────────────────────────────────────────────────────────────────────────────
# Memory logging callback
# ─────────────────────────────────────────────────────────────────────────────

class _MemoryLogCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_reserved() / 1e9
            print(
                f"[Memory] epoch {state.epoch:.0f} — "
                f"allocated: {allocated:.1f} GB  peak reserved: {peak:.1f} GB"
            )
            torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# MOPD Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MOPDTrainer(Trainer):
    """
    HF Trainer that replaces the SFT cross-entropy loss with on-policy
    reverse-KL distillation.

    For every training batch:
      1. Student generates a response ŷ on-policy (no grad).
      2. Full sequence (prompt embeds + response embeds) is forwarded
         through student (grad) and the matching teacher (no grad).
      3. Reverse KL  D_KL(student ‖ teacher)  is averaged over
         non-padding response token positions.

    For evaluation, the standard CE loss on ground-truth labels is used
    so that eval_loss is comparable to SFT baselines.

    Teacher loading happens AFTER super().__init__() so HF Trainer places
    the student on GPU first, avoiding the OOM that occurs when teachers
    are loaded before the student's device placement.
    """

    def __init__(
        self,
        *args,
        teacher_paths: Dict[str, str],
        mopd_tokenizer,
        max_new_tokens: int = 128,
        gen_temperature: float = 0.8,
        kl_temperature: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            teacher_paths:   dict mapping teacher key → checkpoint path.
                             Expected keys: "mix", "both_single", "img_only_single",
                             "text_only_single".  Unknown keys fall back to "mix".
                             If two keys share the same path, the model is loaded
                             once and reused.
            mopd_tokenizer:  tokenizer (used for pad/eos ids during generation).
            max_new_tokens:  max tokens to generate per on-policy rollout.
            gen_temperature: sampling temperature for on-policy generation.
            kl_temperature:  temperature applied before KL (1.0 = no scaling).
        """
        # ── 1. Let HF Trainer initialize and place the student on GPU first ──
        super().__init__(*args, **kwargs)
        self.add_callback(_MemoryLogCallback())

        self._mopd_tok = mopd_tokenizer
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = gen_temperature
        self.kl_temperature = kl_temperature

        # ── 2. Load teachers after student is placed ──────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        loaded_by_path: Dict[str, LlamaForCausalLM] = {}
        self.teachers: Dict[str, LlamaForCausalLM] = {}
        for key, path in teacher_paths.items():
            if path in loaded_by_path:
                self.teachers[key] = loaded_by_path[path]
                print(f"[MOPD] teacher '{key}' reuses already-loaded checkpoint: {path}")
            else:
                self.teachers[key] = load_teacher(
                    path, mopd_tokenizer, local_rank=local_rank
                )
                loaded_by_path[path] = self.teachers[key]

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(model):
        """Strip DDP / FSDP wrapper to reach the base model."""
        return model.module if hasattr(model, "module") else model

    def _generate_on_policy(self, model, input_ids, attention_mask):
        base = self._unwrap(model)
        was_training = base.training
        base.eval()

        B, T_p = input_ids.shape
        actual_lens = attention_mask.sum(dim=1).long()  # (B,) per-sample real lengths
        max_len = int(actual_lens.max().item())
        pad_id = self._mopd_tok.pad_token_id

        # Rebuild as LEFT-padded so generate() starts from the true last token
        # for every sample in the batch.
        left_ids = torch.full(
            (B, max_len), pad_id,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        left_mask = torch.zeros(
            B, max_len,
            dtype=torch.long, device=attention_mask.device,
        )
        for b in range(B):
            al = int(actual_lens[b].item())
            left_ids[b, max_len - al :] = input_ids[b, :al]
            left_mask[b, max_len - al :] = 1

        try:
            with torch.inference_mode():
                generated = base.generate(
                    input_ids=left_ids,
                    attention_mask=left_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.gen_temperature,
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=pad_id,
                    eos_token_id=self._mopd_tok.eos_token_id,
                    synced_gpus=False,
                )
        finally:
            if was_training:
                base.train()

        # generate() returns prompt + new tokens; slice off the (left-padded) prompt
        return generated[:, max_len:]

    @torch.inference_mode()
    def _teacher_resp_logits(
        self,
        teacher: LlamaForCausalLM,
        full_ids: torch.Tensor,      # (sub_B, L)
        full_mask: torch.Tensor,     # (sub_B, L)
        actual_lens: torch.Tensor,   # (sub_B,)  per-sample real prompt lengths
        T_gen: int,
    ) -> torch.Tensor:
        """
        Run teacher on the full sequence and return logits at response positions.
        For sample b, resp_0 logit lives at position actual_lens[b]-1.
        Returns (sub_B, T_gen, V) on the original device, in float32.
        """
        t_dev = next(teacher.parameters()).device
        out = teacher(
            input_ids=full_ids.to(t_dev),
            attention_mask=full_mask.to(device=t_dev, dtype=torch.long),
        )
        logits = out.logits.to(device=full_ids.device, dtype=torch.float32)
        # (sub_B, L, V)

        assert logits.size(1) >= int(actual_lens.max().item()) - 1 + T_gen, (
            f"teacher logits too short: {logits.size(1)} < "
            f"{int(actual_lens.max().item()) - 1 + T_gen}"
        )

        return torch.stack([
            logits[b, actual_lens[b] - 1 : actual_lens[b] - 1 + T_gen, :]
            for b in range(full_ids.size(0))
        ])  # (sub_B, T_gen, V)

    def _make_resp_mask(self, resp_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a valid-token mask for the generated response.
        Keeps every token up to and including the first EOS; masks everything after.
        Works correctly whether pad_token_id == eos_token_id or not.
        """
        eos_id = self._mopd_tok.eos_token_id
        is_eos = (resp_ids == eos_id).int()       # 1 at EOS positions
        eos_seen = torch.cumsum(is_eos, dim=1)    # 0 before first EOS, ≥1 after
        keep = (eos_seen - is_eos) == 0           # True up to and including first EOS
        return keep.long()

    # ── training loss ──────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False):
        # Pop non-standard fields before any model forward call.
        input_types = inputs.pop("input_type", None)   # list[str], len B
        inputs.pop("labels", None)                     # not used in pure KL training

        input_ids = inputs["input_ids"]                    # (B, T_p)
        attention_mask = inputs["attention_mask"].long()   # (B, T_p)

        # One-time dtype sanity check
        if not hasattr(self, "_dtype_check_done"):
            print(
                f"[DTYPE CHECK] input_ids: {input_ids.dtype}  "
                f"attention_mask(raw): {inputs['attention_mask'].dtype}  "
                f"attention_mask(after .long()): {attention_mask.dtype}"
            )
            self._dtype_check_done = True

        B, T_p = input_ids.shape
        pad_id = self._mopd_tok.pad_token_id

        # ── 1. On-policy generation (no grad) ─────────────────────────────────
        resp_ids = self._generate_on_policy(
            model,
            input_ids.detach(),
            attention_mask.detach(),
        )                                            # (B, T_gen)
        T_gen = resp_ids.size(1)

        # ── 2. Build full-sequence ids (per-sample packed) ─────────────────────
        # Pack each sample as [real_prompt_tokens | response_tokens] with no
        # padding between prompt and response, then right-pad the batch with
        # pad_id. This ensures logit at actual_lens[b]-1 cleanly predicts
        # resp_0 for sample b, regardless of variable prompt lengths.
        actual_lens = attention_mask.sum(dim=1).long()   # (B,) real prompt lengths
        max_actual_len = int(actual_lens.max().item())
        max_full_len = max_actual_len + T_gen

        resp_mask = self._make_resp_mask(resp_ids).to(attention_mask.device, dtype=torch.long)  # (B, T_gen)

        full_ids = torch.full(
            (B, max_full_len), pad_id,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        full_mask = torch.zeros(
            B, max_full_len,
            dtype=torch.long, device=attention_mask.device,
        )
        for b in range(B):
            al = int(actual_lens[b].item())
            full_ids[b, :al]            = input_ids[b, :al]
            full_ids[b, al : al+T_gen]  = resp_ids[b]
            full_mask[b, :al]           = 1
            full_mask[b, al : al+T_gen] = resp_mask[b]

        # ── 3. Student forward (with grad) ────────────────────────────────────
        student_out = model(input_ids=full_ids, attention_mask=full_mask)
        s_logits_full = student_out.logits.float()  # (B, max_full_len, V)

        # Per-sample: resp_0 logit for sample b is at position actual_lens[b]-1.
        s_logits = torch.stack([
            s_logits_full[b, actual_lens[b] - 1 : actual_lens[b] - 1 + T_gen, :]
            for b in range(B)
        ])                                          # (B, T_gen, V)

        # ── 4. Teacher logits (no grad), per-sample routing ───────────────────
        types = input_types if input_types is not None else ["mix"] * B

        # Group sample indices by teacher key; unknown keys fall back to "mix".
        groups: Dict[str, list] = defaultdict(list)
        for i, t in enumerate(types):
            key = t if t in self.teachers else "mix"
            groups[key].append(i)

        V = s_logits.size(-1)
        t_logits = torch.empty(
            B, T_gen, V,
            device=input_ids.device, dtype=torch.float32,
        )

        for teacher_key, idx_list in groups.items():
            idx = torch.tensor(idx_list, device=input_ids.device)
            t_logits[idx] = self._teacher_resp_logits(
                self.teachers[teacher_key],
                full_ids[idx],
                full_mask[idx],
                actual_lens[idx],
                T_gen,
            )

        # ── 5. Reverse KL loss ─────────────────────────────────────────────────
        # D_KL(Q‖P)  Q=student, P=teacher
        # = Σ_v Q(v) · [log Q(v) − log P(v)]
        assert s_logits.shape == t_logits.shape, (
            f"logit shape mismatch: student {s_logits.shape} vs teacher {t_logits.shape}"
        )
        assert s_logits.shape[:2] == resp_mask.shape, (
            f"logit/mask shape mismatch: {s_logits.shape[:2]} vs {resp_mask.shape}"
        )

        tau = self.kl_temperature
        s_probs     = F.softmax(s_logits / tau, dim=-1)        # (B, T_gen, V)
        s_log_probs = F.log_softmax(s_logits / tau, dim=-1)    # (B, T_gen, V)
        t_log_probs = F.log_softmax(t_logits / tau, dim=-1)    # (B, T_gen, V)
        kl_tok = (s_probs * (s_log_probs - t_log_probs)).sum(-1)  # (B, T_gen)

        valid = resp_mask.float().to(kl_tok.device)
        loss = (kl_tok * valid).sum() / valid.sum().clamp(min=1.0)

        if return_outputs:
            return loss, student_out
        return loss

    # ── evaluation: standard CE on ground-truth ────────────────────────────────

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Use CE loss on ground-truth labels for evaluation."""
        inputs = self._prepare_inputs(inputs)
        inputs.pop("input_type", None)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.long()  # ← 保险

        with torch.no_grad():
            out = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=attention_mask,
                labels=inputs.get("labels"),
            )
        return (out.loss, None, None)

    # ── checkpoint save ────────────────────────────────────────────────────────

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # PeftModel.save_pretrained saves only the LoRA adapter weights.
        unwrapped = self._unwrap(self.model)
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(output_dir)
            print(f"[MOPD] adapter saved → {output_dir}")
        else:
            import warnings
            warnings.warn("[MOPD] model has no save_pretrained; skipping save.")

        self._mopd_tok.save_pretrained(output_dir)