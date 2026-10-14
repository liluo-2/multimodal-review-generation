"""
GRPOTrainer.py — Stage 2: GRPO + conditional KD / CE
                  with CPU offload for ref model and teachers

Training Objective:
    L = L_GRPO + I₁·λ·L_KD + I₂·γ·L_CE
    I₁, I₂ ∈ {0, 1}  (mutually exclusive per sample)

Memory strategy:
    - Student stays on GPU at all times
    - ref model + teachers are loaded on CPU
    - Moved to GPU only during forward pass, then moved back to CPU
    - This avoids OOM when running 6 × 8B models simultaneously
"""

import csv
import os
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert_score import score as bertscore_score
from rouge_score import rouge_scorer
from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM, Trainer, TrainerCallback


# ─────────────────────────────────────────────────────────────────────────────
# Frozen model loader (CPU offload)
# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_model_cpu(ckpt_path: str, tokenizer, local_rank: int = 0) -> LlamaForCausalLM:
    """
    Load a frozen model in 4-bit quantization on GPU.
    4bit: each model ~4GB instead of ~16GB (bf16).
    5 frozen models × 4GB = ~20GB, leaving plenty of room for student + activations.

    ckpt_path may be either:
      - a PEFT adapter dir (has adapter_config.json) → base + LoRA, or
      - a plain base-model dir (no adapter)          → base model only.
    The latter lets the KL reference (π_ref) be the raw base model when the
    student is trained from base rather than a Stage-1 checkpoint.

    Note: 4bit models cannot be merged with merge_and_unload(), so for adapters
    we keep the PeftModel wrapper and just freeze all parameters.
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    is_peft   = os.path.isfile(os.path.join(ckpt_path, "adapter_config.json"))
    base_path = PeftConfig.from_pretrained(ckpt_path).base_model_name_or_path if is_peft else ckpt_path
    device    = f"cuda:{local_rank}"

    base = LlamaForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map={"": device},    # ← back on GPU, but 4bit
    )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))

    if is_peft:
        # 4bit models cannot merge_and_unload(), keep PeftModel wrapper
        model = PeftModel.from_pretrained(base, ckpt_path)
    else:
        model = base
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[Frozen/4bit-GPU] loaded on {device} (peft={is_peft}): {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Reward function — 6-metric composite
#   BLEU + ROUGE-1 + ROUGE-2 + ROUGE-L + BERTScore + CIDEr_normalized
# Each metric naturally in [0, 1]; sum is in [0, 6].
# CIDEr is computed online via pre-built IDF scorer (trainset corpus).
# CIDEr is min-max normalized using global trainset stats so it sits in [0,1].
# ─────────────────────────────────────────────────────────────────────────────

import evaluate as hf_evaluate
from pycocoevalcap.cider.cider_scorer import CiderScorer as _CiderScorer

_rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_bleu_metric      = hf_evaluate.load("bleu")

# TEMP DEBUG: tracks which modalities have already been logged (DBG_MODALITY=1)
_DBG_SEEN_MODALITIES = set()

# Maps reward_metric name → (single_col, mix_col) in per_sample_detail CSV
METRIC_COL_MAP = {
    "composite": ("single_norm_total", "mix_norm_total"),
    "rouge1":    ("single_ROUGE-1",    "mix_ROUGE-1"),
    "rouge2":    ("single_ROUGE-2",    "mix_ROUGE-2"),
    "rougel":    ("single_ROUGE-L",    "mix_ROUGE-L"),
    "bleu":      ("single_BLEU",       "mix_BLEU"),
    "bertscore": ("single_BERTScore",  "mix_BERTScore"),
    "cider":     ("single_CIDEr",      "mix_CIDEr"),
}
VALID_REWARD_METRICS = set(METRIC_COL_MAP.keys())

# The six sub-metrics that make up the composite reward, and the short names used
# as wandb keys (wandb groups on "/", and "-" in a key is awkward to filter on).
COMPONENT_KEYS = ("ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore", "CIDEr")
COMPONENT_SHORT = {
    "ROUGE-1": "rouge1", "ROUGE-2": "rouge2", "ROUGE-L": "rougel",
    "BLEU": "bleu", "BERTScore": "bertscore", "CIDEr": "cider",
}


def build_cider_idf(trainset_references: list) -> "_CiderScorer":
    """Pre-build CIDEr IDF from trainset references (called once in __init__)."""
    scorer = _CiderScorer(n=4, sigma=6.0)
    for ref in trainset_references:
        scorer += (None, [ref.strip()])
    scorer.compute_doc_freq()
    # CiderScorer.compute_doc_freq() does NOT set ref_len; compute_score() would
    # set it to log(len(crefs)) of whatever scorer it runs on. For a corpus-level
    # IDF we must pin it to log(#corpus docs) once, here.
    scorer.ref_len = np.log(float(max(len(trainset_references), 1)))
    print(f"[GRPO] CIDEr IDF built from {len(trainset_references)} trainset references "
          f"(ref_len={scorer.ref_len:.4f}).")
    return scorer


def compute_cider_online(pred: str, ref: str, idf_scorer: "_CiderScorer") -> float:
    """
    CIDEr for a single (pred, ref) pair against the pre-built trainset IDF.

    NOTE: we deliberately call compute_cider() and NOT compute_score(). The latter
    (a) re-runs compute_doc_freq(), which would *mutate the shared IDF table* since
    document_frequency is assigned by reference, (b) asserts
    len(ctest) >= max(document_frequency.values()) — always false for a single pair
    against a corpus IDF, and (c) resets ref_len to log(1)=0. Previously that
    assertion fired on every call and the except-branch returned 0.0, so the CIDEr
    term was a constant and contributed nothing to the GRPO advantage.
    """
    try:
        scorer = _CiderScorer(n=4, sigma=6.0)
        scorer.document_frequency = idf_scorer.document_frequency   # shared, not mutated
        scorer.ref_len = idf_scorer.ref_len
        scorer += (pred.strip(), [ref.strip()])
        return float(scorer.compute_cider()[0])
    except Exception:
        return 0.0


def compute_reward(
    pred: str,
    ref: str,
    cider_idf_scorer: "_CiderScorer",
    cider_min: float,
    cider_max: float,
    reward_metric: str = "composite",
    precomputed_bertscore: Optional[float] = None,
    metric_stats: Optional[dict] = None,
    modality: Optional[str] = None,
    components: Optional[dict] = None,
) -> float:
    """
    Compute reward for a single (pred, ref) pair.
    reward_metric selects which signal to use:
      "composite"  → ROUGE-1 + ROUGE-2 + ROUGE-L + BLEU + BERTScore-F1 + CIDEr_norm  (range [0, 6])
      "rouge1"     → ROUGE-1 F1                                                        (range [0, 1])
      "rouge2"     → ROUGE-2 F1                                                        (range [0, 1])
      "rougel"     → ROUGE-L F1                                                        (range [0, 1])
      "bleu"       → BLEU                                                               (range [0, 1])
      "bertscore"  → official BERTScore F1 (roberta-large, rescaled)                  (range [0, 1])
      "cider"      → CIDEr normalized to [0, 1] via trainset min/max                  (range [0, 1])
    BERTScore must be passed via precomputed_bertscore (batch-computed in _compute_rewards).
    Only the metrics needed for the selected reward_metric are computed.
    """
    if not pred or not ref:
        return 0.0

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    if reward_metric in ("rouge1", "rouge2", "rougel", "composite"):
        r      = _rouge_scorer_obj.score(ref, pred)
        rouge1 = r["rouge1"].fmeasure
        rouge2 = r["rouge2"].fmeasure
        rougel = r["rougeL"].fmeasure
        if reward_metric == "rouge1": return rouge1
        if reward_metric == "rouge2": return rouge2
        if reward_metric == "rougel": return rougel
    else:
        rouge1 = rouge2 = rougel = 0.0

    # ── BLEU ──────────────────────────────────────────────────────────────────
    if reward_metric in ("bleu", "composite"):
        try:
            # smooth=True is REQUIRED here: this is a per-sentence BLEU (one pred
            # at a time), and unsmoothed BLEU-4 is exactly 0 for ~94% of samples
            # because its geometric mean collapses when no 4-gram matches. A
            # near-constant term contributes nothing to the GRPO advantage
            # (R - mu)/sigma, so BLEU was effectively absent from the reward.
            b    = _bleu_metric.compute(predictions=[pred], references=[[ref]], smooth=True)
            bleu = b["bleu"]
        except Exception:
            bleu = 0.0
        if reward_metric == "bleu":
            if components is not None:
                components["BLEU"] = bleu
            return bleu
    else:
        bleu = 0.0

    # ── BERTScore (official roberta-large F1, batch-computed upstream) ────────
    if reward_metric in ("bertscore", "composite"):
        bertscore = precomputed_bertscore if precomputed_bertscore is not None else 0.0
        if reward_metric == "bertscore": return bertscore
    else:
        bertscore = 0.0

    # ── CIDEr — online, normalized to [0, 1] using trainset global min/max ───
    if reward_metric in ("cider", "composite"):
        raw_cider = compute_cider_online(pred, ref, cider_idf_scorer)
        if cider_max > cider_min:
            cider_norm = float(np.clip((raw_cider - cider_min) / (cider_max - cider_min), 0.0, 1.0))
        else:
            cider_norm = 0.0
        if reward_metric == "cider":
            if components is not None:
                components["CIDEr"] = cider_norm
            return cider_norm
    else:
        raw_cider = 0.0
        cider_norm = 0.0

    # ── Composite: per-metric normalization when stats available ─────────────
    if metric_stats is not None:
        # metric_stats is keyed by modality ("both", "img_only", "text_only")
        mod_stats = metric_stats.get(modality) if modality else None
        if mod_stats is None:
            # fallback: use first available modality (e.g. stats is flat dict from old format)
            mod_stats = next(iter(metric_stats.values()))
            if not isinstance(mod_stats, dict) or "min" in mod_stats:
                mod_stats = metric_stats  # old flat format
        # ── TEMP DEBUG: verify per-modality stats are actually selected ──────
        # Run with DBG_MODALITY=1. Prints each modality only the FIRST time it is
        # seen (≤3 lines total). Expect both/img_only/text_only with matching
        # BERTScore.min (both=-0.430, img_only=-0.690, text_only=-0.748).
        # Only 'both' ever printed ⇒ modality propagation fix did NOT take.
        if os.environ.get("DBG_MODALITY") == "1" and modality not in _DBG_SEEN_MODALITIES:
            _DBG_SEEN_MODALITIES.add(modality)
            _bs_min = (mod_stats or {}).get("BERTScore", {}).get("min")
            print(f"[DBG] modality={modality!r}  BERTScore.min={_bs_min}", flush=True)
        # ── END TEMP DEBUG ──────────────────────────────────────────────────
        def _norm(val: float, key: str) -> float:
            s = mod_stats[key]
            r = s["max"] - s["min"]
            return float(np.clip((val - s["min"]) / r, 0.0, 1.0)) if r > 1e-9 else 0.0
        vals = {
            "ROUGE-1":   _norm(rouge1,    "ROUGE-1"),
            "ROUGE-2":   _norm(rouge2,    "ROUGE-2"),
            "ROUGE-L":   _norm(rougel,    "ROUGE-L"),
            "BLEU":      _norm(bleu,      "BLEU"),
            "BERTScore": _norm(bertscore, "BERTScore"),
            "CIDEr":     _norm(raw_cider, "CIDEr"),
        }
        if components is not None:
            components.update(vals)
        return sum(vals.values())

    if components is not None:
        components.update({
            "ROUGE-1": rouge1, "ROUGE-2": rouge2, "ROUGE-L": rougel,
            "BLEU": bleu, "BERTScore": bertscore, "CIDEr": cider_norm,
        })
    return rouge1 + rouge2 + rougel + bleu + bertscore + cider_norm


# Keep old name as alias for backward compatibility
def compute_composite_reward(pred, ref, cider_idf_scorer, cider_min, cider_max,
                             precomputed_bertscore=None):
    return compute_reward(pred, ref, cider_idf_scorer, cider_min, cider_max,
                          "composite", precomputed_bertscore)


# ─────────────────────────────────────────────────────────────────────────────
# Teacher reward lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher_rewards(csv_path: str, dataset: str,
                         reward_metric: str = "composite") -> Dict[str, dict]:
    """
    Load precomputed teacher scores from per_sample_detail CSV.
    reward_metric selects which column to use as the teacher score for routing:
      "composite" → single_norm_total / mix_norm_total  (sum of 6 group-normalized metrics)
      "rouge1/2/l/bleu/bertscore" → raw per-metric columns, already in [0, 1]
      "cider"     → raw CIDEr columns, normalized to [0, 1] using trainset min/max
    CIDEr min/max is always extracted for online student reward normalization.
    """
    if reward_metric not in METRIC_COL_MAP:
        raise ValueError(f"Unknown reward_metric '{reward_metric}'. "
                         f"Choose from: {sorted(METRIC_COL_MAP)}")
    single_col, mix_col = METRIC_COL_MAP[reward_metric]

    result = {}
    cider_vals = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"].strip().lower() != dataset.lower():
                continue
            sid = row["id"].strip()

            def _safe_float(key):
                v = row.get(key, "") or ""
                return float(v) if v not in ("", "nan") else 0.0

            single_score = _safe_float(single_col)
            mix_score    = _safe_float(mix_col)
            result[sid] = {
                "single_score":   single_score,
                "mix_score":      mix_score,
                "best_score":     max(single_score, mix_score),
                "better_teacher": "single" if single_score > mix_score else "mix",
            }
            # Always collect raw CIDEr for student normalization
            for k in ("single_CIDEr", "mix_CIDEr"):
                v = row.get(k, "") or ""
                if v not in ("", "nan"):
                    try:
                        cider_vals.append(float(v))
                    except ValueError:
                        pass

    cider_min = float(min(cider_vals)) if cider_vals else 0.0
    cider_max = float(max(cider_vals)) if cider_vals else 1.0

    # CIDEr columns are raw (unbounded); normalize teacher scores to [0, 1]
    # so they are on the same scale as the student's online CIDEr reward.
    if reward_metric == "cider" and cider_max > cider_min:
        for sid in result:
            s = float(np.clip(
                (result[sid]["single_score"] - cider_min) / (cider_max - cider_min), 0.0, 1.0))
            m = float(np.clip(
                (result[sid]["mix_score"]    - cider_min) / (cider_max - cider_min), 0.0, 1.0))
            result[sid]["single_score"] = s
            result[sid]["mix_score"]    = m
            result[sid]["best_score"]   = max(s, m)

    print(f"[GRPO] Loaded teacher rewards ({reward_metric}, col={single_col}/{mix_col}) "
          f"for {len(result)} samples")
    print(f"[GRPO] CIDEr range for student normalization: [{cider_min:.4f}, {cider_max:.4f}]")
    return result, cider_min, cider_max


# ─────────────────────────────────────────────────────────────────────────────
# Memory callback
# ─────────────────────────────────────────────────────────────────────────────

class _MemoryLogCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            peak      = torch.cuda.max_memory_reserved() / 1e9
            print(f"[Memory] epoch {state.epoch:.0f} — alloc: {allocated:.1f}GB  peak: {peak:.1f}GB")
            torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

class GRPOTrainer(Trainer):

    def __init__(
        self,
        *args,
        ref_model_path:       str,
        teacher_paths:        Dict[str, str],
        grpo_tokenizer,
        teacher_rewards:      Dict[str, dict],
        gt_texts:             Dict[str, str],
        trainset_references:  list,           # NEW: List[str] for CIDEr IDF
        cider_min:            float = 0.0,    # NEW: from load_teacher_rewards
        cider_max:            float = 1.0,    # NEW: from load_teacher_rewards
        metric_stats:         Optional[dict] = None,  # per-metric min/max for composite norm
        reward_metric:        str   = "composite",
        num_rollouts:         int   = 4,
        max_new_tokens:       int   = 256,
        gen_temperature:      float = 1.0,
        reward_threshold:     float = 0.3,
        clip_epsilon:         float = 0.2,
        kl_beta:              float = 0.01,
        kd_lambda:            float = 0.5,
        ce_gamma:             float = 0.5,
        kl_temperature:       float = 1.0,
        num_grpo_epochs:      int   = 1,
        loss_mode:            str   = "grpo_kd_ce",
        # ── PPO-only hyperparams (ignored by grpo modes) ──
        gamma:                float = 1.0,
        gae_lambda:           float = 0.95,
        vf_coef:              float = 0.5,
        cliprange_value:      float = 0.2,
        eval_reward_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_callback(_MemoryLogCallback())

        if reward_metric not in VALID_REWARD_METRICS:
            raise ValueError(f"Unknown reward_metric '{reward_metric}'. "
                             f"Choose from: {sorted(VALID_REWARD_METRICS)}")
        if loss_mode not in ("ppo", "ppo_kd_ce", "grpo_only", "grpo_kd_ce"):
            raise ValueError(f"Unknown loss_mode '{loss_mode}'. "
                             f"Choose from: ['ppo', 'ppo_kd_ce', 'grpo_only', 'grpo_kd_ce']")
        self.loss_mode        = loss_mode
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.vf_coef          = vf_coef
        self.cliprange_value  = cliprange_value
        self.eval_reward_batch_size = eval_reward_batch_size
        self.reward_metric    = reward_metric
        self._tok             = grpo_tokenizer
        self.teacher_rewards  = teacher_rewards
        self.gt_texts         = gt_texts
        self.num_rollouts     = num_rollouts
        self.max_new_tokens   = max_new_tokens
        self.gen_temperature  = gen_temperature
        self.reward_threshold = reward_threshold
        self.clip_epsilon     = clip_epsilon
        self.kl_beta          = kl_beta
        self.kd_lambda        = kd_lambda
        self.ce_gamma         = ce_gamma
        self.kl_temperature   = kl_temperature
        self.num_grpo_epochs  = num_grpo_epochs
        self.cider_min        = cider_min
        self.cider_max        = cider_max
        self.metric_stats     = metric_stats

        # Reward stats accumulated per step, flushed to wandb inside log()
        self._rew_mean_acc = []
        self._rew_std_acc  = []
        # Per-modality: modality → [sum, count]. Lets us see whether the composite
        # reward is rising on a modality while one of its components falls
        # (the reward-hacking signature we saw on img_only / CIDEr).
        self._rew_mod_acc  = defaultdict(lambda: [0.0, 0])
        # modality → metric → [sum, count], plus an all-modality bucket per metric
        self._comp_mod_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        self._comp_all_acc = defaultdict(lambda: [0.0, 0])

        # ── Build CIDEr IDF from trainset references ──────────────────────────
        self.cider_idf_scorer = build_cider_idf(trainset_references)

        # ── BERTScore uses official bert_score library (roberta-large) ──────────
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._bertscore_device = f"cuda:{local_rank}"
        print(f"[GRPO] BERTScore will use roberta-large on {self._bertscore_device}")
        # Pre-warm roberta-large so both ranks load it before DDP training starts.
        # Without this, the first _compute_rewards call triggers a slow model load
        # on one rank while the other rank waits at the NCCL barrier → timeout.
        print("[GRPO] Pre-warming BERTScore (roberta-large) ...")
        bertscore_score(["warmup"], ["warmup"], model_type="roberta-large", lang="en",
                        rescale_with_baseline=True, device=self._bertscore_device, verbose=False)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print("[GRPO] BERTScore pre-warm done.")

        # ── Load ref model and teachers in 4bit on GPU ────────────────────────

        print("[GRPO] Loading reference model (π_ref) in 4bit ...")
        self.ref_model = load_frozen_model_cpu(ref_model_path, grpo_tokenizer, local_rank)

        self.teachers:  Dict[str, LlamaForCausalLM] = {}
        if self.loss_mode in ("grpo_only", "ppo"):
            # Pure GRPO / PPO: no KD/CE routing, so teacher models are never used.
            # Skip loading them to save ~4GB × #teachers of GPU memory.
            print(f"[GRPO] loss_mode={self.loss_mode} → skipping teacher model loading.")
        else:
            print("[GRPO] Loading teacher models in 4bit ...")
            loaded_by_path: Dict[str, LlamaForCausalLM] = {}
            for key, path in teacher_paths.items():
                if path in loaded_by_path:
                    self.teachers[key] = loaded_by_path[path]
                    print(f"[GRPO] teacher '{key}' reuses: {path}")
                else:
                    self.teachers[key] = load_frozen_model_cpu(path, grpo_tokenizer, local_rank)
                    loaded_by_path[path] = self.teachers[key]

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _gpu_device(self, input_ids: torch.Tensor) -> str:
        """Return the GPU device string matching the input tensor."""
        return str(input_ids.device)

    def _left_pad_for_generate(self, input_ids, attention_mask):
        B             = input_ids.size(0)
        actual_lens   = attention_mask.sum(dim=1).long()
        max_len       = int(actual_lens.max().item())
        pad_id        = self._tok.pad_token_id

        left_ids  = torch.full((B, max_len), pad_id, dtype=input_ids.dtype,  device=input_ids.device)
        left_mask = torch.zeros(B, max_len,          dtype=torch.long,        device=input_ids.device)
        for b in range(B):
            al = int(actual_lens[b].item())
            left_ids[b,  max_len - al:] = input_ids[b, :al]
            left_mask[b, max_len - al:] = 1
        return left_ids, left_mask, max_len

    @torch.inference_mode()
    def _generate_rollouts(self, model, input_ids, attention_mask):
        base        = self._unwrap(model)
        was_training = base.training
        base.eval()

        B, G = input_ids.size(0), self.num_rollouts
        ids_rep  = input_ids.repeat_interleave(G, dim=0)
        mask_rep = attention_mask.repeat_interleave(G, dim=0)
        left_ids, left_mask, max_len = self._left_pad_for_generate(ids_rep, mask_rep)

        try:
            generated = base.generate(
                input_ids=left_ids,
                attention_mask=left_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.gen_temperature,
                top_p=1.0, top_k=0,
                pad_token_id=self._tok.pad_token_id,
                eos_token_id=self._tok.eos_token_id,
                synced_gpus=False,
            )
        finally:
            if was_training:
                base.train()

        return generated[:, max_len:].clone()   # (B*G, T_gen)  ← .clone() avoids inference-tensor error

    def _make_resp_mask(self, resp_ids):
        eos_id   = self._tok.eos_token_id
        is_eos   = (resp_ids == eos_id).int()
        eos_seen = torch.cumsum(is_eos, dim=1)
        return ((eos_seen - is_eos) == 0).long()

    def _build_full_sequence(self, input_ids, attention_mask, resp_ids):
        B_total    = input_ids.size(0)
        T_gen      = resp_ids.size(1)
        actual_lens = attention_mask.sum(dim=1).long()
        max_actual  = int(actual_lens.max().item())
        max_full    = max_actual + T_gen
        pad_id      = self._tok.pad_token_id

        resp_mask = self._make_resp_mask(resp_ids).to(input_ids.device)
        full_ids  = torch.full((B_total, max_full), pad_id, dtype=input_ids.dtype,  device=input_ids.device)
        full_mask = torch.zeros(B_total, max_full,          dtype=torch.long,        device=input_ids.device)

        for b in range(B_total):
            al = int(actual_lens[b].item())
            full_ids[b,  :al]       = input_ids[b, :al]
            full_ids[b,  al:al+T_gen] = resp_ids[b]
            full_mask[b, :al]       = 1
            full_mask[b, al:al+T_gen] = resp_mask[b]

        return full_ids, full_mask, actual_lens, T_gen, resp_mask

    def _get_resp_logprobs_gpu(self, model, full_ids, full_mask, actual_lens, T_gen, resp_ids):
        """
        Run frozen model (4bit, on GPU) forward, extract per-token log-probs.
        Returns (B_total, T_gen) log-prob tensor.
        """
        with torch.inference_mode():
            out    = model(input_ids=full_ids, attention_mask=full_mask)
            logits = out.logits.to(dtype=torch.float32)

        log_probs_all = F.log_softmax(logits, dim=-1)
        resp_logprobs = []
        for b in range(full_ids.size(0)):
            al     = int(actual_lens[b].item())
            lp     = log_probs_all[b, al - 1 : al - 1 + T_gen, :]
            tok_lp = lp.gather(1, resp_ids[b].unsqueeze(1)).squeeze(1)
            resp_logprobs.append(tok_lp)
        return torch.stack(resp_logprobs)   # (B_total, T_gen)

    def _compute_rewards(self, resp_ids, gt_text_list, B, G, modalities=None):
        # Decode all predictions upfront
        pred_texts = [
            self._tok.decode(resp_ids[b * G + g], skip_special_tokens=True).strip()
            for b in range(B) for g in range(G)
        ]
        ref_texts = [gt_text_list[b] for b in range(B) for g in range(G)]

        # Batch BERTScore once for all B*G pairs (only when needed)
        bertscore_f1s = None
        if self.reward_metric in ("composite", "bertscore"):
            _, _, F1 = bertscore_score(
                pred_texts, ref_texts,
                model_type="roberta-large",
                lang="en",
                rescale_with_baseline=True,
                device=self._bertscore_device,
                verbose=False,
            )
            bertscore_f1s = F1.tolist()

        rewards = torch.zeros(B, G)
        # Every normalized sub-metric, so we can see which ones actually move the
        # composite and which are effectively constant (e.g. BLEU is ~0 for 90%+
        # of samples, so it contributes nothing to the group-normalized advantage).
        comps = {k: torch.zeros(B, G) for k in COMPONENT_KEYS}
        for b in range(B):
            for g in range(G):
                idx = b * G + g
                comp = {}
                rewards[b, g] = compute_reward(
                    pred_texts[idx], ref_texts[idx],
                    self.cider_idf_scorer, self.cider_min, self.cider_max,
                    self.reward_metric,
                    precomputed_bertscore=bertscore_f1s[idx] if bertscore_f1s else None,
                    metric_stats=self.metric_stats,
                    modality=modalities[b] if modalities else None,
                    components=comp,
                )
                for k in COMPONENT_KEYS:
                    comps[k][b, g] = comp.get(k, 0.0)
        # rewards: (B, G) — range depends on reward_metric
        # comps:   {metric: (B, G)} — each normalized to [0, 1]
        return rewards, comps

    # ── standard GRPO clip: rollouts frozen, num_grpo_epochs updates ───────────

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Override HF Trainer's training_step to implement standard GRPO clipping.

        old_logprobs are computed ONCE from the model before any gradient update.
        num_grpo_epochs gradient steps then run on the same rollout batch, so the
        policy gradually drifts and the PPO clip starts to constrain updates
        from epoch 1 onward.

        When num_grpo_epochs == 1 this is identical to the original compute_loss flow.
        For num_grpo_epochs > 1, set gradient_accumulation_steps=1 in TrainingArguments
        to avoid interleaving inner manual steps with HF Trainer's outer accumulation.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        input_types         = inputs.pop("input_type",          None)
        labels              = inputs.pop("labels",              None)
        sample_ids          = inputs.pop("sample_id",           None)
        modalities          = inputs.pop("modality",            None)
        full_input_ids      = inputs.pop("full_input_ids",      None)
        full_attention_mask = inputs.pop("full_attention_mask", None)

        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].long()
        B = input_ids.size(0)
        G = self.num_rollouts

        if input_types is None: input_types = ["mix"] * B
        if sample_ids  is None: sample_ids  = [str(i) for i in range(B)]

        # ── Generate rollouts ONCE ────────────────────────────────────────────
        resp_ids = self._generate_rollouts(
            model, input_ids.detach(), attention_mask.detach()
        ).clone()
        T_gen = resp_ids.size(1)

        # ── Rewards & advantages ONCE ─────────────────────────────────────────
        gt_text_list    = [self.gt_texts.get(sid, "") for sid in sample_ids]
        rewards, comps = self._compute_rewards(resp_ids, gt_text_list, B, G, modalities=modalities)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        mean_rewards    = rewards.mean(dim=1)
        # accumulate reward stats for wandb (flushed in log()); group_std is the
        # within-prompt spread across rollouts (≈0 when num_rollouts==1, i.e. PPO)
        self._rew_mean_acc.append(float(mean_rewards.mean().item()))
        self._rew_std_acc.append(float(rewards.std(dim=1, unbiased=False).mean().item()))
        # Average each component over the G rollouts → (B,) per metric
        comp_mean = {k: t.mean(dim=1) for k, t in comps.items()}
        for k, t in comp_mean.items():
            self._comp_all_acc[k][0] += float(t.mean().item())
            self._comp_all_acc[k][1] += 1
        if modalities is not None:
            for b, m in enumerate(modalities):
                self._rew_mod_acc[m][0] += float(mean_rewards[b].item())
                self._rew_mod_acc[m][1] += 1
                for k, t in comp_mean.items():
                    acc = self._comp_mod_acc[m][k]
                    acc[0] += float(t[b].item())
                    acc[1] += 1
        mu_R            = rewards.mean(dim=1, keepdim=True)
        sigma_R         = rewards.std(dim=1,  keepdim=True)
        advantages      = (rewards - mu_R) / (sigma_R + 1e-8)
        advantages_flat = advantages.reshape(B * G).to(input_ids.device)

        # ── Full sequences ONCE ───────────────────────────────────────────────
        ids_rep  = input_ids.repeat_interleave(G, dim=0)
        mask_rep = attention_mask.repeat_interleave(G, dim=0)
        full_ids, full_mask, actual_lens, _, resp_mask = self._build_full_sequence(
            ids_rep, mask_rep, resp_ids
        )
        valid = resp_mask.float()

        # ── old_logprobs ONCE — frozen before any weight update ───────────────
        with torch.no_grad():
            old_logprobs = self._get_resp_logprobs_student(
                self._unwrap(model), full_ids, full_mask, actual_lens, T_gen, resp_ids
            )   # (B*G, T_gen) — fixed reference throughout all epochs

        # ── ref log-probs ONCE ────────────────────────────────────────────────
        ref_logprobs = self._get_resp_logprobs_gpu(
            self.ref_model, full_ids, full_mask, actual_lens, T_gen, resp_ids
        )

        # ── PPO branch: value function + per-token GAE, no group baseline ──────
        # (ppo = pure PPO; ppo_kd_ce = PPO + per-sample conditional KD/CE)
        if self.loss_mode in ("ppo", "ppo_kd_ce"):
            return self._ppo_training_step(
                model, full_ids, full_mask, actual_lens, T_gen, resp_ids,
                valid, old_logprobs, ref_logprobs, rewards, mean_rewards, B, G,
                input_types=input_types, sample_ids=sample_ids, modalities=modalities,
                labels=labels, full_input_ids=full_input_ids,
                full_attention_mask=full_attention_mask,
            )

        # ── Multi-epoch loop ──────────────────────────────────────────────────
        # Epochs 0 .. N-2 : manual backward + optimizer.step (model weights update).
        # Epoch  N-1 (last): call accelerator.backward, return detached loss so
        #                    HF Trainer's outer optimizer.step() runs as normal.
        loss = None
        n_grpo_only = n_grpo_kd = n_grpo_ce = 0
        for grpo_epoch in range(self.num_grpo_epochs):
            # new_logprobs come from updated weights after epoch > 0 → ratio != 1
            student_out      = model(input_ids=full_ids, attention_mask=full_mask)
            new_logits       = student_out.logits.float()
            new_logprobs_all = F.log_softmax(new_logits, dim=-1)

            new_logprobs = []
            for b in range(B * G):
                al     = int(actual_lens[b].item())
                lp     = new_logprobs_all[b, al - 1 : al - 1 + T_gen, :]
                tok_lp = lp.gather(1, resp_ids[b].unsqueeze(1)).squeeze(1)
                new_logprobs.append(tok_lp)
            new_logprobs = torch.stack(new_logprobs)

            # L_GRPO
            eps          = self.clip_epsilon
            ratio        = torch.exp(new_logprobs - old_logprobs)
            adv_expanded = advantages_flat.unsqueeze(1).expand_as(ratio)
            surr1        = ratio * adv_expanded
            surr2        = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_expanded
            policy_loss  = -torch.min(surr1, surr2)

            # k3 KL estimator (DeepSeek/TRL): unbiased, non-negative, and its
            # gradient correctly pulls the policy back toward π_ref. The naive
            # (new - ref) term has zero expected gradient and only adds variance.
            kl_diff          = ref_logprobs - new_logprobs
            kl_penalty       = torch.exp(kl_diff) - kl_diff - 1.0
            grpo_per_token   = policy_loss + self.kl_beta * kl_penalty
            grpo_per_rollout = (grpo_per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            grpo_per_sample  = grpo_per_rollout.reshape(B, G).mean(dim=1)

            # Per-sample routing (routing decisions fixed by mean_rewards, same each epoch)
            total_loss  = torch.zeros(B, device=input_ids.device)
            n_grpo_only = n_grpo_kd = n_grpo_ce = 0

            # ── Pure GRPO: no KD/CE, just the GRPO loss for every sample ──────────
            if self.loss_mode == "grpo_only":
                total_loss  = grpo_per_sample
                n_grpo_only = B
                loss = total_loss.mean()

                is_last = (grpo_epoch == self.num_grpo_epochs - 1)
                if not is_last:
                    loss.backward()
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.accelerator.unwrap_model(model).parameters(),
                            self.args.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()
                continue

            for b in range(B):
                sid    = sample_ids[b]
                mean_r = float(mean_rewards[b].item())
                total_loss[b] = grpo_per_sample[b]

                if mean_r >= self.reward_threshold:
                    n_grpo_only += 1
                    continue

                teacher_info   = self.teacher_rewards.get(sid)
                teacher_better = False
                teacher_key    = "mix"
                if teacher_info is not None and teacher_info["best_score"] > mean_r:
                    teacher_better = True
                    teacher_key    = input_types[b] if input_types[b] in self.teachers else "mix"

                if teacher_better:
                    teacher   = self.teachers.get(teacher_key, self.teachers["mix"])
                    b_indices = list(range(b * G, (b + 1) * G))
                    b_idx     = torch.tensor(b_indices, device=input_ids.device)
                    t_logits  = self._get_teacher_logits_gpu(teacher, full_ids[b_idx], full_mask[b_idx])
                    tau       = self.kl_temperature
                    # Reverse KL D_KL(student‖teacher), per-rollout token-mean then
                    # averaged over the G rollouts (same formula as Stage 1 MOPDTrainer;
                    # reduction is per-rollout to weight each rollout equally).
                    kd_loss_b = 0.0
                    for g_idx, flat_idx in enumerate(b_indices):
                        al         = int(actual_lens[flat_idx].item())
                        s_logits_g = new_logits[flat_idx, al - 1 : al - 1 + T_gen, :] / tau
                        t_logits_g = t_logits[g_idx,      al - 1 : al - 1 + T_gen, :] / tau
                        s_probs    = F.softmax(s_logits_g, dim=-1)
                        s_lp       = F.log_softmax(s_logits_g, dim=-1)
                        t_lp       = F.log_softmax(t_logits_g, dim=-1)
                        kl_tok     = (s_probs * (s_lp - t_lp)).sum(-1)   # D_KL(student‖teacher)
                        v          = valid[flat_idx]
                        kd_loss_b += (kl_tok * v).sum() / v.sum().clamp(min=1)
                    total_loss[b] = total_loss[b] + self.kd_lambda * (kd_loss_b / G)
                    n_grpo_kd += 1
                else:
                    if labels is not None and full_input_ids is not None:
                        ce_out = model(
                            input_ids=full_input_ids[b:b+1],
                            attention_mask=full_attention_mask[b:b+1].long(),
                            labels=labels[b:b+1],
                        )
                        total_loss[b] = total_loss[b] + self.ce_gamma * ce_out.loss
                    n_grpo_ce += 1

            loss = total_loss.mean()

            is_last = (grpo_epoch == self.num_grpo_epochs - 1)
            if not is_last:
                # Manual step: backward, clip, optimizer.step, zero grad
                loss.backward()
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.accelerator.unwrap_model(model).parameters(),
                        self.args.max_grad_norm,
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()

        if not hasattr(self, "_log_counter"):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 1:
            print(
                f"[GRPO] routing: only={n_grpo_only}  +KD={n_grpo_kd}  +CE={n_grpo_ce}"
                f"  mean_r={mean_rewards.mean():.4f}"
                f"  grpo_epochs={self.num_grpo_epochs}"
            )

        # Last epoch: hand off to HF Trainer's outer optimizer.step() path
        self.accelerator.backward(loss)
        return loss.detach()

    # ── wandb logging: attach reward stats to the trainer's log stream ─────────

    def log(self, logs, *args, **kwargs):
        """
        Inject reward stats into HF Trainer's periodic log dict so they land on
        the same wandb step axis as train loss / lr. Values are averaged over all
        training_steps since the previous log event (cadence = logging_steps).
        """
        if getattr(self, "_rew_mean_acc", None):
            n = len(self._rew_mean_acc)
            logs["reward/mean"]      = sum(self._rew_mean_acc) / n
            logs["reward/group_std"] = sum(self._rew_std_acc)  / n
            self._rew_mean_acc = []
            self._rew_std_acc  = []

            # Per-modality composite reward
            for m, (s, c) in list(self._rew_mod_acc.items()):
                if c:
                    logs[f"reward/mean_{m}"] = s / c
            # Each sub-metric: overall, and split by modality
            for k, (s, c) in list(self._comp_all_acc.items()):
                if c:
                    logs[f"reward/{COMPONENT_SHORT.get(k, k)}"] = s / c
            for m, per_metric in list(self._comp_mod_acc.items()):
                for k, (s, c) in per_metric.items():
                    if c:
                        logs[f"reward/{COMPONENT_SHORT.get(k, k)}_{m}"] = s / c
            self._rew_mod_acc.clear()
            self._comp_mod_acc.clear()
            self._comp_all_acc.clear()
        return super().log(logs, *args, **kwargs)

    # ── eval reward: generate on the full eval set once per epoch ─────────────

    def evaluate(self, *args, **kwargs):
        """Run the standard CE eval, then a reward pass over the eval set."""
        metrics = super().evaluate(*args, **kwargs)
        ds = getattr(self, "reward_eval_dataset", None)
        if ds is not None:
            reward_mean = self._evaluate_reward(ds)
            metrics["eval_reward_mean"] = reward_mean
            self.log({"reward/eval_mean": reward_mean})
            print(f"[GRPO] eval_reward_mean = {reward_mean:.4f}  (n={len(ds)})")
        return metrics

    @torch.no_grad()
    def _evaluate_reward(self, dataset) -> float:
        """
        Generate ONE sample per eval prompt over the full eval set and average
        the reward vs GT. Uses the same reward function as training.
        """
        was_training = self.model.training
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.eval_reward_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,      # GRPODataCollator (carries sample_id/modality)
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        saved_G, self.num_rollouts = self.num_rollouts, 1   # single generation per prompt
        total, count = 0.0, 0
        try:
            for batch in loader:
                batch          = self._prepare_inputs(batch)
                input_ids      = batch["input_ids"]
                attention_mask = batch["attention_mask"].long()
                sample_ids     = batch.get("sample_id")  or [str(i) for i in range(input_ids.size(0))]
                modalities     = batch.get("modality")
                B              = input_ids.size(0)

                resp_ids = self._generate_rollouts(self.model, input_ids, attention_mask)  # (B, T)
                gt_list  = [self.gt_texts.get(sid, "") for sid in sample_ids]
                rewards, _ = self._compute_rewards(resp_ids, gt_list, B, 1, modalities=modalities)  # (B, 1)
                total   += float(rewards.sum().item())
                count   += B
        finally:
            self.num_rollouts = saved_G
            if was_training:
                self.model.train()

        return total / max(count, 1)

    # ── PPO: value head + GAE ────────────────────────────────────────────────

    def _values_from_hidden(self, unwrapped_model, hidden, actual_lens, T_gen):
        """
        Apply the value head to the last-layer hidden states and slice out the
        per-token value estimates aligned with the response tokens.
        Same alignment convention as log-probs: position (al-1 + i) is the state
        that predicts response token i, so V(s_i) is read from there.
        Returns (N, T_gen) float tensor.
        """
        vh   = unwrapped_model.value_head
        vals = vh(hidden.to(vh.weight.dtype)).squeeze(-1).float()   # (N, L)
        out  = []
        for b in range(vals.size(0)):
            al = int(actual_lens[b].item())
            out.append(vals[b, al - 1 : al - 1 + T_gen])
        return torch.stack(out)   # (N, T_gen)

    def _get_resp_values_student(self, unwrapped_model, full_ids, full_mask, actual_lens, T_gen):
        """Forward the student once to get per-response-token value estimates."""
        out    = unwrapped_model(input_ids=full_ids, attention_mask=full_mask,
                                 output_hidden_states=True)
        hidden = out.hidden_states[-1]
        return self._values_from_hidden(unwrapped_model, hidden, actual_lens, T_gen)

    def _ppo_training_step(
        self, model, full_ids, full_mask, actual_lens, T_gen, resp_ids,
        valid, old_logprobs, ref_logprobs, rewards, mean_rewards, B, G,
        input_types=None, sample_ids=None, modalities=None,
        labels=None, full_input_ids=None, full_attention_mask=None,
    ):
        """
        PPO with a learned value baseline (value head on the shared backbone) +
        per-token GAE advantages. The KL-to-ref penalty is folded into the
        per-token reward (RLHF-style), NOT added as a separate loss term.

        loss_mode:
          "ppo"       → pure PPO (L = L_policy + vf_coef · L_value)
          "ppo_kd_ce" → PPO + per-sample conditional KD/CE, same routing as
                        grpo_kd_ce: if mean reward ≥ threshold keep PPO only;
                        else if a teacher is better add reverse-KL KD; else add CE.
        """
        device    = full_ids.device
        N         = B * G
        unwrapped = self._unwrap(model)
        do_routing = (self.loss_mode == "ppo_kd_ce")

        if input_types is None: input_types = ["mix"] * B
        if sample_ids  is None: sample_ids  = [str(i) for i in range(B)]

        # ── old value estimates (frozen, before any update) ───────────────────
        with torch.no_grad():
            old_values = self._get_resp_values_student(
                unwrapped, full_ids, full_mask, actual_lens, T_gen
            )   # (N, T_gen)

        # ── per-token reward: terminal metric reward + per-token KL penalty ───
        # KL_t uses the sampling (old) policy vs ref: KL_t = logπ_old - logπ_ref.
        kl_per_tok     = old_logprobs - ref_logprobs                  # (N, T_gen)
        per_tok_reward = (-self.kl_beta * kl_per_tok) * valid
        rewards_flat   = rewards.reshape(N).to(device)
        last_idx       = valid.long().sum(dim=1) - 1                  # (N,) last valid token
        row            = torch.arange(N, device=device)
        li             = last_idx.clamp(min=0)
        per_tok_reward[row, li] += rewards_flat * (last_idx >= 0).float()

        # ── GAE (backward recursion over response tokens) ─────────────────────
        advantages = torch.zeros(N, T_gen, device=device)
        lastgae    = torch.zeros(N, device=device)
        for t in reversed(range(T_gen)):
            if t < T_gen - 1:
                next_val     = old_values[:, t + 1]
                next_nonterm = valid[:, t + 1]
            else:
                next_val     = torch.zeros(N, device=device)
                next_nonterm = torch.zeros(N, device=device)
            delta   = per_tok_reward[:, t] + self.gamma * next_val * next_nonterm - old_values[:, t]
            lastgae = delta + self.gamma * self.gae_lambda * next_nonterm * lastgae
            advantages[:, t] = lastgae
        returns = advantages + old_values

        # ── whiten advantages over valid tokens ───────────────────────────────
        vmask = valid.bool()
        if vmask.any():
            adv_valid = advantages[vmask]
            advantages = (advantages - adv_valid.mean()) / (adv_valid.std().clamp(min=1e-8))
        advantages = advantages * valid   # zero out padding

        # ── PPO multi-epoch update ────────────────────────────────────────────
        eps  = self.clip_epsilon
        loss = None
        n_ppo_only = n_ppo_kd = n_ppo_ce = 0
        for grpo_epoch in range(self.num_grpo_epochs):
            student_out      = model(input_ids=full_ids, attention_mask=full_mask,
                                     output_hidden_states=True)
            new_logits       = student_out.logits.float()
            new_logprobs_all = F.log_softmax(new_logits, dim=-1)

            new_logprobs = []
            for b in range(N):
                al     = int(actual_lens[b].item())
                lp     = new_logprobs_all[b, al - 1 : al - 1 + T_gen, :]
                tok_lp = lp.gather(1, resp_ids[b].unsqueeze(1)).squeeze(1)
                new_logprobs.append(tok_lp)
            new_logprobs = torch.stack(new_logprobs)

            new_values = self._values_from_hidden(
                unwrapped, student_out.hidden_states[-1], actual_lens, T_gen
            )

            # policy loss (clipped surrogate), per-token advantage from GAE
            ratio       = torch.exp(new_logprobs - old_logprobs)
            surr1       = ratio * advantages
            surr2       = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            policy_tok  = -torch.min(surr1, surr2)                       # (N, T_gen)

            # clipped value loss
            v_clipped  = old_values + torch.clamp(
                new_values - old_values, -self.cliprange_value, self.cliprange_value
            )
            vf1        = (new_values - returns) ** 2
            vf2        = (v_clipped - returns) ** 2
            value_tok  = 0.5 * torch.max(vf1, vf2)                       # (N, T_gen)

            # per-rollout token-mean → per-sample (mean over G rollouts)
            denom              = valid.sum(dim=1).clamp(min=1)
            policy_per_rollout = (policy_tok * valid).sum(dim=1) / denom
            value_per_rollout  = (value_tok  * valid).sum(dim=1) / denom
            ppo_per_rollout    = policy_per_rollout + self.vf_coef * value_per_rollout
            ppo_per_sample     = ppo_per_rollout.reshape(B, G).mean(dim=1)   # (B,)

            # aggregate scalars for logging
            policy_loss = (policy_tok * valid).sum() / valid.sum().clamp(min=1)
            value_loss  = (value_tok  * valid).sum() / valid.sum().clamp(min=1)

            if not do_routing:
                # ── Pure PPO ──────────────────────────────────────────────────
                loss       = ppo_per_sample.mean()
                n_ppo_only = B
            else:
                # ── PPO + conditional KD/CE (same routing as grpo_kd_ce) ──────
                total_loss = torch.zeros(B, device=device)
                n_ppo_only = n_ppo_kd = n_ppo_ce = 0
                for b in range(B):
                    total_loss[b] = ppo_per_sample[b]
                    sid    = sample_ids[b]
                    mean_r = float(mean_rewards[b].item())

                    if mean_r >= self.reward_threshold:
                        n_ppo_only += 1
                        continue

                    teacher_info   = self.teacher_rewards.get(sid)
                    teacher_better = False
                    teacher_key    = "mix"
                    if teacher_info is not None and teacher_info["best_score"] > mean_r:
                        teacher_better = True
                        teacher_key    = input_types[b] if input_types[b] in self.teachers else "mix"

                    if teacher_better:
                        teacher   = self.teachers.get(teacher_key, self.teachers["mix"])
                        b_indices = list(range(b * G, (b + 1) * G))
                        b_idx     = torch.tensor(b_indices, device=device)
                        t_logits  = self._get_teacher_logits_gpu(teacher, full_ids[b_idx], full_mask[b_idx])
                        tau       = self.kl_temperature
                        kd_loss_b = 0.0
                        for g_idx, flat_idx in enumerate(b_indices):
                            al         = int(actual_lens[flat_idx].item())
                            s_logits_g = new_logits[flat_idx, al - 1 : al - 1 + T_gen, :] / tau
                            t_logits_g = t_logits[g_idx,      al - 1 : al - 1 + T_gen, :] / tau
                            s_probs    = F.softmax(s_logits_g, dim=-1)
                            s_lp       = F.log_softmax(s_logits_g, dim=-1)
                            t_lp       = F.log_softmax(t_logits_g, dim=-1)
                            kl_tok     = (s_probs * (s_lp - t_lp)).sum(-1)   # D_KL(student‖teacher)
                            v          = valid[flat_idx]
                            kd_loss_b += (kl_tok * v).sum() / v.sum().clamp(min=1)
                        total_loss[b] = total_loss[b] + self.kd_lambda * (kd_loss_b / G)
                        n_ppo_kd += 1
                    else:
                        if labels is not None and full_input_ids is not None:
                            ce_out = model(
                                input_ids=full_input_ids[b:b+1],
                                attention_mask=full_attention_mask[b:b+1].long(),
                                labels=labels[b:b+1],
                            )
                            total_loss[b] = total_loss[b] + self.ce_gamma * ce_out.loss
                        n_ppo_ce += 1

                loss = total_loss.mean()

            is_last = (grpo_epoch == self.num_grpo_epochs - 1)
            if not is_last:
                loss.backward()
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.accelerator.unwrap_model(model).parameters(),
                        self.args.max_grad_norm,
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()

        if not hasattr(self, "_log_counter"):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 1:
            tag = "PPO+KD/CE" if do_routing else "PPO"
            print(
                f"[{tag}] policy={policy_loss.item():.4f}  value={value_loss.item():.4f}"
                f"  routing: only={n_ppo_only} +KD={n_ppo_kd} +CE={n_ppo_ce}"
                f"  mean_r={mean_rewards.mean():.4f}  ppo_epochs={self.num_grpo_epochs}"
            )

        self.accelerator.backward(loss)
        return loss.detach()

    # ── student-only logprob helper (stays on GPU, no offload needed) ──────────

    def _get_resp_logprobs_student(self, model, full_ids, full_mask, actual_lens, T_gen, resp_ids):
        """Student forward (grad disabled), no device transfer needed."""
        out           = model(input_ids=full_ids, attention_mask=full_mask)
        logits        = out.logits.float()
        log_probs_all = F.log_softmax(logits, dim=-1)

        resp_logprobs = []
        for b in range(full_ids.size(0)):
            al     = int(actual_lens[b].item())
            lp     = log_probs_all[b, al - 1 : al - 1 + T_gen, :]
            tok_lp = lp.gather(1, resp_ids[b].unsqueeze(1)).squeeze(1)
            resp_logprobs.append(tok_lp)
        return torch.stack(resp_logprobs)   # (B*G, T_gen)

    def _get_teacher_logits_gpu(self, teacher, full_ids, full_mask):
        """
        Run teacher (4bit, on GPU) forward, return full logits (B, L, V).
        """
        with torch.inference_mode():
            out    = teacher(input_ids=full_ids, attention_mask=full_mask)
            logits = out.logits.to(dtype=torch.float32)
        return logits

    # ── evaluation ────────────────────────────────────────────────────────────

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        for k in ("input_type", "sample_id", "modality", "full_input_ids", "full_attention_mask"):
            inputs.pop(k, None)

        attn = inputs.get("attention_mask")
        if attn is not None:
            inputs["attention_mask"] = attn.long()

        with torch.no_grad():
            out = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
            )
        return (out.loss, None, None)

    # ── checkpoint save ────────────────────────────────────────────────────────

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self._unwrap(self.model)
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(output_dir)
            print(f"[GRPO] adapter saved → {output_dir}")
        else:
            import warnings
            warnings.warn("[GRPO] model has no save_pretrained; skipping save.")
        # PPO value head is not part of the LoRA adapter → save it separately so
        # resume can restore the critic.
        value_head = getattr(unwrapped, "value_head", None)
        if value_head is not None:
            torch.save(value_head.state_dict(), os.path.join(output_dir, "value_head.pt"))
            print(f"[PPO] value head saved → {output_dir}/value_head.pt")
        self._tok.save_pretrained(output_dir)