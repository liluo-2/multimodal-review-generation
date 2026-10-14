# Negative Results

Each finding is stated as a claim, followed by the evidence and the mechanism.

---

### 1. Embedding-level fusion loses to prompt-level text injection

**Claim**: A learned fusion module (BLIP-2 encoders → cross-attention → 4-head GMU gating → MLP projector → soft prompt tokens) consistently underperforms simply writing the same information as natural language into the LLM prompt.

**Evidence**: Controlled comparison across Arts/Sports/Toys, ruling out history count and title-input format as confounds. GMU gating weight traces show all 4 heads oscillating at ≈0.33 (uniform averaging) across all three datasets. Under text-only input the image weight does not drop; under img-only the text weight stays ~0.3.

**Mechanism**: A softmax gate over three similarly-scaled embeddings has no gradient pressure to specialize. It converges to uniform averaging, making the entire fusion stack an expensive mean-pool.

---

### 2. Semantic retrieval does not beat recency for history selection

**Claim**: Title-based BERT cosine retrieval does not outperform simply taking the most recent N reviews. A retrieval + 120-day time-window combination initially appeared better but the gain vanished once history count was controlled.

**Evidence**: 3 datasets × 5 history counts (0/3/5/10/all) × 2 strategies. 17-sample case study showing recent history reconstructs the user's ongoing project context (14/17), current vocabulary (17/17), and latest sentence patterns (9/17).

**Mechanism**: History transfers *writing templates*, not product knowledge. Recency preserves the user's current project timeline and active phrasing; retrieval matches topical keywords but loses temporal coherence.

---

### 3. LLM-extracted preference profiles do not help

**Claim**: Explicitly extracting a user's style patterns and content preferences via GPT-4.1-mini and injecting them into the prompt produces no improvement over raw history.

**Evidence**: 5-way ablation (history / +preference / +content / +style / preference-only) on 982 samples.

**Mechanism**: The profile forces the model to use listed phrasings even when they don't fit naturally. The extraction is noise-sensitive. Current metrics cannot detect style transfer — the signal may exist but be unmeasurable with BLEU/ROUGE.

---

### 4. Modality-specific bottleneck MLP (hard-routing MoE) hurts

**Claim**: Inserting a per-modality bottleneck MLP between the last transformer layer and `lm_head` (`h_out = h + MLP_type(h)`) degrades performance on most dataset × modality cells.

**Evidence**: Multi-dataset, multi-modality comparison. No router, no balancing loss — pure sample-level hard routing by known input type.

---

### 5. Modality-specific LM heads are worse still

**Claim**: Giving each input type its own `lm_head` (initialized from pretrained, not random) produces worse results than a shared head.

**Evidence**: Three mechanistic reasons:
1. All three input types produce natural-language reviews. The vocabulary distributions are near-identical — there is no modality-specific signal in vocab space to capture.
2. Three heads share one LoRA. Gradients from three modalities superimpose with no conflict resolution.
3. Scale mismatch: 3 × `lm_head` = 1.58B params vs LoRA = 3.4M params (~500×). They cannot co-converge under one optimizer.

**Design principle**: In LLM multi-task learning, task-specific capacity belongs in the hidden representation space (middle-layer adapters, LoRA experts, MoE FFN), with a shared output head.

---

### 6. Per-sample best-teacher selection via GT metrics is unsound

**Claim**: Choosing the best teacher per training sample by evaluating each teacher's output against the ground truth, then distilling that teacher's distribution, does not produce a useful student.

**Evidence**: Five-point diagnosis:
1. Labels encode which teacher *overfit this training sample best*, not which is best in general.
2. Cross-modality pairing runs the teacher OOD (a text-only input routed to the `both` teacher).
3. Per-sample BLEU CV ≈ 50%; ~30% of samples have near-identical teacher scores → labels are noise.
4. Teachers have systematically different output distributions; per-sample switching gives an incoherent mixture target.
5. Reverse KL is mode-seeking; a jumping target distribution has no stable convergence point.

---

### 7. Stage 1 (MOPD) is not necessary

**Claim**: The best balanced configuration skips multi-teacher distillation entirely and runs GRPO directly on the mix-SFT checkpoint.

**Evidence**: Full comparison table. `mix-SFT → GRPO` outperforms `MOPD → GRPO` on text-only and img-only while remaining competitive on both.

---

### 8. img-only underperformance is an input information bottleneck

**Claim**: Image-only performance stays weak across all methods, and the cause is insufficient information in the input, not a training or reward failure.

**Evidence**:
- CIDEr diverges across stages (both: +0.008 → −0.056 → −0.173) while BERTScore stays flat.
- Zero content-word overlap with reference: img-only 41.4% vs text 30.0%.
- Removing CIDEr from reward changes img-only CIDEr by only 0.0008 (hallucination hypothesis falsified).
- Case-level: errors are *wrong-purpose guesses* (pottery tools classified as jewelry tools), not fabricated brands.

---

### 9. Adding raw product images to a text generator hurts on average

**Claim**: Across 8–10 Amazon categories, adding the product's first image to the input reduces performance overall. Adding text description helps on only about half of categories.

**Evidence**: 2×2 comparison (LLaMA-3 vs Qwen-VL × base vs +description), extended across categories.
