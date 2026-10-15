# Planning Notes

Moved from the main README — benchmark design, GitHub/HuggingFace setup, and CV material.


# Part II · Turning this into a benchmark

You have most of a benchmark already. What's missing is mostly **protocol**, not new experiments.

## What already exists

- A cleaning pipeline over 10 Amazon categories + Goodreads (dedup, video-template removal, short-review filtering, description/image presence filters, 10–30 reviews per user, context-length filtering)
- A **principled three-scenario protocol** (text-only / img-only / both) derived from MLLM data filtering rather than assumed
- An extended eval harness with variance reporting and a **visual-grounding metric** nobody else has
- 6+ baselines already run: ReviewLLM, ReviewLLM+desc, VLM single, VLM mix, img2txt single, img2txt mix, MOPD, GRPO variants

## What must be fixed first

**1. Shared test set across modalities — this is the blocking issue.**
Right now the three modality test sets have **non-overlapping ids**. That makes every cross-modality number incomparable, and it's a confound you already flagged yourself. The fix: one test set of user–item pairs, three **input views** of the same pairs (mask the image → text-only; mask title+desc → img-only; full → both). Same references, same difficulty, differences attributable to modality alone.

**2. Reference-answerability filter.**
Ship two tracks: `full` and `answerable`. Rule- or LLM-filter out shipping complaints, seller questions, and non-evaluative text. Report both. Your zero-overlap analysis shows this shifts the numbers materially.

**3. Scale.**
600–1,300 per cell is below what BLEU/CIDEr variance can support. Target ≥5,000 per cell, or ship bootstrap confidence intervals as mandatory (your `_std` and `n_draws` columns are already halfway there).

**4. Two official decoding tracks.**
`greedy` and `sampled (mean over 5 draws, temp=0.8, top_p=0.9)`. Your own results show methods reverse rank between them. A benchmark that reports only one is hiding a real effect.

**5. Leakage audit.**
Users disjoint across train/test; every history item strictly earlier than its target; no target review appearing in any user's history.

## What to add

**6. A personalization metric.** The gap in the whole field. Two cheap options:
   - *User-discrimination*: train a classifier on held-out real reviews to identify the author; report accuracy on generated reviews. Higher = better style capture.
   - *LLM-as-judge pairwise*: given the user's history, which of two candidate reviews is more likely theirs?

**7. Fixed CIDEr IDF corpus** — you already track `CIDEr_idf_source`; just freeze it per split and document it.

**8. Reproducibility package** — download + preprocess script with pinned seeds, a dataset card (license, Amazon Reviews'23 / Goodreads attribution), split files as release artifacts, and a `results.csv` schema.

**Suggested name/framing**: *MMRevBench — a modality-heterogeneous benchmark for personalized review generation*. The pitch writes itself: existing work assumes one input modality; real data has three; here's the split, the protocol, the baselines, and eight things that don't work.

---

# Part III · Getting it onto GitHub

Yes — send me code and I'll organize it. Ranked by how much value each file adds:

**Essential**
1. `eval.py` / `eval_cc.py` — the eval harness is the single most reusable asset here. I'd extract it into a standalone `mmrevbench-eval` module with a clean CLI.
2. Data preprocessing + split generation (whatever produces `final-desc-img.jsonl` and `split/revgenfinal/*.npz`)
3. `train_grpo.py` / `GRPOTrainer_cider.py` — the composite reward + threshold + teacher-fallback routing is the novel implementation
4. `train_mopd.sh` + the MOPD trainer

**Valuable**
5. `train_img.py` (SFT, single/mix), `inference_0529.py`
6. The img2txt generation script + prompt
7. The BLIP-2 + GMU fusion model code (Phase 2) — even though it failed, the gating-weight logging is what makes negative result 1 credible

**Nice to have**
8. Per-sample metric computation → `per_sample_detail_*.csv`
9. Plotting / results aggregation

**What I'd deliver**: a repo with `data/` `models/` `training/` `eval/` `scripts/` `docs/`, this README as the front page, a `RESULTS.md` with the full tables, a `NEGATIVE_RESULTS.md`, per-directory READMEs, and cluster-specific paths abstracted into a config file. If you'd rather not publish training code, an **eval-harness-only** repo plus this README is still a strong artifact on its own.

**One caution**: several scripts contain absolute paths under `/scratch/xs2334/` and `/scratch/cc8463/`, and the notes file contains **live OpenAI and OpenRouter API keys**. Rotate those keys now regardless of whether you publish — they've been sitting in a shared document.

---

# Part IV · CV bullets

Written in the register of your reference bullet. Pick 3–5 that match what you personally owned — don't use all of them.

**Architecture / framework**
- Built an end-to-end multimodal review generation framework integrating BLIP-2 encoders, a multi-head GMU gating network, an MLP projector, and LLaMA-3, with learnable modality vectors enabling inference under missing text or image inputs; diagnosed gating-weight collapse across 3 datasets to establish that embedding-level fusion underperforms prompt-level injection.

**The headline result — use this one**
- Built a unified review generator serving text-only, image-only, and multimodal product inputs from one checkpoint via mixed-scenario SFT on LLaMA-3.1-8B followed by GRPO with a scenario-normalized composite reward; **improved BLEU by 32% and ROUGE-2 by 28% on multimodal inputs and ROUGE-2 by 19% on text-only inputs** over the strongest baseline, with GRPO improving every SFT initialization tested.

**Pipeline / efficiency**
- Replaced end-to-end VLM backbones with a Qwen2.5-VL image-captioning pipeline feeding LLaMA-3.1, outperforming the VLM on multimodal inputs across Arts, Sports, and Toys (BLEU +21% to +69%) while keeping a single text-only model for all input scenarios.

**Multi-task learning**
- Investigated gradient interference in 3-way multimodal multi-task training through ratio training, curriculum learning, modality dropout, hard-routing MoE, and modality-specific LM heads; established that task-specific capacity must live in hidden-representation space rather than the vocabulary projection layer, supported by a 500× parameter-scale mismatch analysis between LM heads and LoRA adapters.

**RL / reward design**
- Implemented GRPO with a modality-normalized composite reward (BLEU/ROUGE/BERTScore) and per-sample routing between policy-gradient, knowledge-distillation, and cross-entropy objectives; identified and corrected reward-design failures including corpus-level CIDEr misuse causing full model collapse.

**Analysis / data**
- Ran modality-attribution analysis with a fine-tuned Qwen2.5-VL over 13.9k reviews; among the 3.3k with a strictly best input configuration, text-, image-, and jointly-explained reviews were nearly equally represented, motivating a three-scenario evaluation protocol.
- Diagnosed persistent image-only underperformance as an input information bottleneck rather than a training failure via zero-overlap content-word analysis (41% vs 30% for text) and metric-divergence tracking, ruling out the hallucination hypothesis through reward ablation.
- Conducted case-level analysis of user-history contribution across 27 annotated samples, establishing that history transfers **writing templates rather than product knowledge**, and that recency outperforms semantic retrieval by preserving ongoing usage context, current vocabulary, and recent sentence structure.

**Benchmark / tooling**
- Built an extended evaluation harness for personalized generation covering BLEU, ROUGE, BERTScore, chrF++, CIDEr with controlled IDF corpora, Distinct-n, and a custom **Visual-Entity Coverage/Precision/F1** metric for measuring visual grounding, with multi-draw sampling and per-draw variance reporting.
- Curated a modality-heterogeneous benchmark spanning 10 Amazon categories and 3 Goodreads genres with a cleaning and filtering pipeline, and benchmarked 8 model configurations across three input scenarios.

---

## Still to fill in

| Gap | Source |
|---|---|
| **Qwen3-VL-8B results** — six modality-specific SFT models were trained (Arts/Toys × 3); the numbers and the comparison against `img2txt + LLaMA-3` aren't in the notes. **If Qwen3-VL beats img2txt, Positive Result 1 must be scoped to the Qwen2.5-VL generation.** | your runs |
| Goodreads baseline numbers | 4.23–4.30 experiments |
| Whether LLMLingua / cross-attn compression was ever run | if not, mark "designed, not executed" |
| Whether PEPLER was actually run as a baseline or only cited | confirm before publishing |
| Final conclusions from the original long/negative-review hard-case experiments (June–July 2025) | early results |
| Sports/Toys GRPO numbers to complete the main table | partially run |
