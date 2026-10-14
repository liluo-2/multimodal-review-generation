"""
Datacollator_grpo.py — Collator for Stage 2 GRPO training.

Each sample has:
  - input_ids / attention_mask          → prompt-only (for rollout generation)
  - full_input_ids / full_attention_mask → prompt + GT (for CE loss fallback)
  - labels                              → with prompt masked to -100
  - input_type                          → teacher routing key (str)
  - sample_id                           → for teacher reward lookup (str)

During evaluation, the eval dataset (RevGenDataset SFT mode) does NOT have
full_input_ids — the collator handles both cases gracefully.
"""

from typing import Any, List, Dict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class GRPODataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ── String fields (pass through as lists) ─────────────────────────────
        input_type_list = [f.get("input_type", "mix")  for f in features]
        sample_id_list  = [f.get("sample_id",  "")     for f in features]
        modality_list   = [f.get("modality",   "both") for f in features]

        # ── Prompt-only (for rollout generation) ──────────────────────────────
        input_ids      = [f["input_ids"]      for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        enc = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = dict(enc)

        # ── Full sequence + labels (for CE loss) ──────────────────────────────
        # Training samples (GRPORevGenDataset) have full_input_ids.
        # Eval samples (RevGenDataset SFT mode) do NOT — fall back to input_ids.
        is_grpo_train = "full_input_ids" in features[0]

        if is_grpo_train:
            full_ids  = [f["full_input_ids"]      for f in features]
            full_mask = [f["full_attention_mask"]  for f in features]
            labels    = [f["labels"]               for f in features]
        else:
            # Eval mode: no full_input_ids, use input_ids as fallback
            full_ids  = [f["input_ids"]      for f in features]
            full_mask = [f["attention_mask"] for f in features]
            labels    = [f.get("labels", torch.full((f["input_ids"].size(0),), -100, dtype=torch.long))
                         for f in features]

        full_enc = self.tokenizer.pad(
            {"input_ids": full_ids, "attention_mask": full_mask},
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["full_input_ids"]      = full_enc["input_ids"]
        batch["full_attention_mask"] = full_enc["attention_mask"]

        # Pad labels to match full sequence length
        T_full = full_enc["input_ids"].size(1)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        if padded_labels.size(1) < T_full:
            pad = torch.full(
                (padded_labels.size(0), T_full - padded_labels.size(1)),
                -100, dtype=padded_labels.dtype,
            )
            padded_labels = torch.cat([padded_labels, pad], dim=1)
        else:
            padded_labels = padded_labels[:, :T_full]
        batch["labels"] = padded_labels

        # ── Routing metadata ──────────────────────────────────────────────────
        batch["input_type"] = input_type_list
        batch["sample_id"]  = sample_id_list
        batch["modality"]   = modality_list

        return batch