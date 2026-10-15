# Multimodal Personalized Review Generation

**A systematic study of whether — and how — product images help LLMs write personalized user reviews.**

2025.06 – 2026.08 · NYU · LLaMA-3.1-8B / Qwen2.5-VL / Qwen3-VL / BLIP-2

---

## TL;DR

We asked one question and spent 14 months answering it properly:

> Real product data comes in three flavors — **text-only**, **image-only**, and **both**. Existing review generators handle exactly one of them. Can a single model handle all three, and does adding vision actually help?

**Answer: partially.** Mix-SFT + GRPO gives a single model that beats the strongest baseline on every metric for text-only and both (both: BLEU +32.0%, ROUGE-2 +27.7%; text-only: ROUGE-2 +18.5%), with smaller gains on image-only. We traced the smaller image-only gains to an **information bottleneck in the input**, not a modeling failure. Along the way we ruled out — with mechanistic evidence, not just worse numbers — embedding-level fusion, retrieval-based history selection, LLM-extracted preference profiles, modality-specific MoE, modality-specific LM heads, and multi-teacher on-policy distillation.

The final method works. It just isn't novel enough to be a paper. So this repo is written as **a benchmark and a set of negative results** instead.

---

## Task

| | |
|---|---|
| **Input** | **History**: past items (title / description / image) + the user's reviews + ratings.<br>**Target**: item title / description / image + rating. |
| **Output** | A review in that user's voice for the target item. |
| **Datasets** | 10 Amazon categories (Movies, Grocery, Arts, Video_Games, Health, Beauty, Pet_Supplies, Electronics, Sports, Toys) + Goodreads (Comics/Fantasy/History-Biography). Main experiments on **Arts / Sports / Toys**. |
| **Metrics** | BLEU · ROUGE-1/2/L · BERTScore · CIDEr · chrF++ · Distinct-1/2 · Visual-Entity Coverage/Precision/F1 · output-length stats |
| **Baselines** | PEPLER, ReviewLLM (LLaMA-3), LLaVA-Tour (MLLM), VLM single/mix |

Final experimental splits:

| Dataset | both | text-only | img-only |
|---|---|---|---|
| Arts | 900 (750 / 150) | 848 (707 / 141) | 769 (641 / 128) |
| Sports | 1078 (898 / 180) | 1069 (891 / 178) | 1008 (840 / 168) |
| Toys | 1459 (1216 / 243) | 1428 (1190 / 238) | 1476 (1230 / 246) |

---

# The story, in six questions

Each section: **the question → what we tried → what happened → what it means.**

---

## Q1 · Does multimodal information help at all?

### The question
ReviewLLM and PEPLER take only text. Everyone assumes adding the product image helps. Nobody had checked.

### What we tried
A clean 2×2 on Movies, then extended to 8–10 categories:

| | LLaMA-3 | Qwen-VL |
|---|---|---|
| **base** | ReviewLLM | ReviewLLM + image |
| **+ description** | ReviewLLM + desc | ReviewLLM + image + desc |

### What happened
- **+ description**: improved on only about half the categories.
- **+ image**: **hurt overall.**
- **+ image, then + description**: better than image alone on >half the categories.

### The analysis
"Adding images hurts" has three possible causes, and they're confounded. We separated them:

1. **Backbone confound.** Qwen-VL is weaker than LLaMA-3, so the drop might be the backbone, not the modality. → Ran Qwen-VL on *text-only* and LLaVA (LLaMA-3 backbone) on *image+text* to level the comparison.
2. **The data doesn't need it.** Some categories have low-quality descriptions; some users' reviews simply never reference product specs.
3. **Image noise.** We used the product's *first* image, unfiltered. Much of what's visible is irrelevant to a review ("they all come in a plastic case").

### The pivot that mattered
To test cause 2 properly we fine-tuned Qwen2.5-VL to generate the review from item information alone under three input configurations (text only / image only / both). Of 13,900 reviews, 3,305 have one configuration that is strictly best on **every** metric; among these the split is nearly even:

| Best-explained by | count |
|---|---|
| text only | 1,122 |
| image only | 1,054 |
| text + image | 1,129 |

**Real review data is genuinely heterogeneous.** That killed the framing "does vision help on average" and replaced it with the framing that drove the rest of the project: *three input scenarios, one model.*

---

## Q2 · How should multimodal information enter the LLM — as embeddings or as text?

### The question
The natural architecture is a learned fusion module producing soft prompt tokens. That's what everyone in multimodal recsys does. Is it actually better than just writing the information into the prompt?

### What we tried — the fusion architecture

```
Target item                                                     ┌─────────────┐
  title+desc → BLIP-2 text enc  ─┬─→ txtemb (CLS)                │             │
                                 ├─→ crossemb (cross-attn)  ──→  │ Multi-head  │
  image      → BLIP-2 vis enc   ─┴─→ imgemb (mean-pool)          │ (4 heads)   │──→ Projector ──→ <item_tokens>
                                                                 │ GMU gating  │      (MLP)              │
User history (top-5, CLIP retrieval,                             │             │                        ▼
  priority: same rating → same sentiment → other)                └─────────────┘                     LLaMA-3
  title+desc → BERT   ─┬─→ txtemb                                      ▲                                │
                       ├─→ crossemb  ────────────────────────────────  ┘                                ▼
  image      → CLIP   ─┴─→ imgemb                                                                    Review
```

**Cross-attention**: `Q = learnable vector`, `K,V = W[Seq_txt ; Seq_img]`, `crossemb = W_o(softmax(QKᵀ/√d)·V)`
**GMU gating**: `w = softmax(W·[txtemb, crossemb, imgemb])`, `fused = w₁·txtemb + w₂·crossemb + w₃·imgemb`
**Missing modality**: learnable text/image sequences of the same shape stand in for absent inputs — this is what lets the framework handle all three scenarios where baselines can't.

### What happened
Persistent instability, then persistent underperformance.

**Debugging trail:**
- CLIP text encoder caps at 77 tokens → descriptions truncated → switched to BLIP-2 text encoder (later raised to its 512-token max).
- Loss wouldn't converge → replaced raw features with **BLIP-2 Q-Former-aligned** features; changed cross-attention so `Q` is a learnable query and `K/V` is `[text_seq ; image_seq]`. Loss stabilized.
- Removed the gating network entirely in favor of **4 learnable queries** doing cross-attention.

**Ruling out the boring explanations:**

| Suspicion | Test | Verdict |
|---|---|---|
| We only give 5 history items, ReviewLLM gets all | Give ReviewLLM only top-5 too | No real change — **not the cause** |
| ReviewLLM wins because product title is explicit in the prompt | Put title explicitly in prompt, description via embedding | Small gain, **still below ReviewLLM** |

### The analysis — why it actually failed
We logged the GMU gating weights across training. This is the real evidence:

- **Arts** — *head collapse*: all weights oscillate in 0.3–0.4. None of the 4 heads develops a preference for any modality.
- **Sports** — under **text-only** input, the image weight **does not drop**. Under **image-only** input, the text weight stays around 0.3. The gate sits in a uniform-averaging state; it has essentially **no modality awareness**.
- **Toys** — no convergence trend at all across steps; a random walk. The model is stuck in a local optimum with no gradient pressure to break the symmetry.

> **A softmax gate over three ~equally-scaled embeddings has no incentive to specialize.** It converges to ≈⅓/⅓/⅓, which makes the whole fusion stack an expensive mean-pool. That is why it loses to simply writing the same information into the prompt as natural language.

**❌ Negative result 1 — Embedding-level fusion (GMU gating + projector + soft prompts) is consistently worse than prompt-level text injection, and the gating weights show why: the gate never becomes modality-aware.**

### The pivot: img2txt
If the LLM wants text, give it text. We used Qwen2.5-VL to convert each product image into 2–4 natural-language sentences and wrote that into the ReviewLLM prompt.

```
Look at this product image and describe it in 2-4 natural sentences. Include:
- What the product is (category, brand/name if visible)
- Appearance (color, material, shape/design)
- Any visible text or labels
- Notable features or details
Requirements: only describe what is actually visible; be concise; do not guess.
```

**✅ Positive result 1 — `img2txt + LLaMA-3` beats the end-to-end VLM backbone across the board** (largest gain on *both*), while cutting input length and training cost dramatically.

VLM input lengths this replaced:

| Dataset | avg tokens | max | 95th pct | train time (H200) |
|---|---|---|---|---|
| Toys | 11,091 | 22,744 | 16,014 | 11h51m |
| Sports | 10,572 | 21,047 | 15,814 | 7h58m |
| Arts | 10,570 | 20,465 | 15,309 | 6h43m |

*Minor finding*: keeping `%Product description:` and `%Image description:` as **separate** fields outperformed a single merged "description and image" field (Arts both BLEU 0.0457 vs 0.0371), most clearly in the img-only scenario. The final prompt uses separate fields.

---

## Q3 · What does user history actually contribute, and how should we select it?

### The question
Everyone retrieves "relevant" history. Is relevance the right criterion?

### What we tried
Qwen2.5-VL backbone, `0 / 3 / 5 / 10 / all` history items × `{recent, title-retrieval (BERT cosine, re-sorted by time)}`.

### What happened

1. **History is necessary** — 0-history is clearly worst.
2. **More is not better** — Arts BERTScore peaks at 5 items (0.869) and *declines* at 10 (0.862) and all (0.858).
3. **❌ Negative result 2 — semantic retrieval does not beat recency.**

### The analysis — two case studies

**(a) What does history give the model? (3 retrieved vs 0 history, 10 improved + 10 degraded cases)**

The dominant mechanism is **template transfer**, not product knowledge:

> The model lifts the user's recurring phrasings — *"along with…"*, *"as advertised"*, *"quick delivery"*, *"shipped quickly"*, *"from this Seller"*, *"will order again"*.
> When those templates happen to match the ground truth → metrics go up (**8/10** of the improved cases).
> When they don't → the templates drag the model off-target.

Secondary contributions: product aspects the user cares about, and personal context (teacher / parent / hobbyist).

> **History supplies a writing template, not knowledge about the product.** This reframes the whole personalization problem — and explains why retrieval-by-topical-similarity is the wrong objective.

**(b) Why does recency beat retrieval? (17 cases where recent wins on every metric)**

| Mechanism | Frequency |
|---|---|
| Recent history reconstructs the user's **ongoing project / usage context** | 14/17 |
| Recent history covers the user's **current vocabulary and attended aspects** | 17/17 |
| Recent history supplies the user's **latest sentence patterns and rhythm** | 9/17 (4/17 near-verbatim) |

Overlap between the two strategies:

| Dataset | hist=3 | hist=5 | hist=10 |
|---|---|---|---|
| Arts | 26.2% | 39.3% | 72.9% |
| Sports | 25.5% | 37.5% | 71.5% |

*Canonical example* — target: floral stem wire. Recent history reveals a continuous **wedding paper-flower project** (origami → hot glue → beads → thin wire), and the model produces *"I use these to hold the beads on the paper flowers."* Title-retrieval matches the keyword "wire" but loses the project timeline, so the intended use is never recovered.

*Style example* — ground truth: *"As always … cut wonderful. I love …"*. That exact construction appears in the user's **immediately preceding** review. Retrieval also surfaces it, but buried mid-list, and the model flattens it into generic prose.

### The follow-up that we then had to retract
- **A**: retrieve → filter to last 120 days → re-sort by time (variable count, avg 3.37)
- **B**: filter to last 120 days → retrieve top-3 → re-sort by time (fixed count)

Both initially beat `recent=5`, with A ahead. **But once we equalized the number of history items at inference, `recent_5` was still the most robust and best overall.** The earlier gain was a history-count artifact.

> Worth stating plainly: this was a result we reported, then overturned by controlling a confound.

---

## Q4 · Can we make the personalization signal explicit?

### The question
If history works via style templates + attended aspects, extract those explicitly and hand them to the model.

### What we tried
GPT-4.1-mini extracts a **target-aware** profile from recent history, split into:
- **Style** — frequent expressions, tone, structure
- **Content** — preferred aspects, personal context, usage scenarios

Constraints in the extraction prompt: only patterns recurring across *multiple* reviews; use the target item to *filter* patterns, never to invent them; keep "who the user is" separate from "how the product is used"; return empty lists when no pattern exists; output JSON.

Ablation: `history` / `history+preference` / `history+content` / `history+style` / `preference only`. Cost: **$0.775 for 982 samples**.

### What happened
**❌ Negative result 3 — the preference profile contributes nothing.**

### The analysis
- The profile makes the model **force** the listed phrasings even when they don't fit — the prompt explicitly said *"do not force expressions if they do not fit naturally"* and that didn't hold.
- Very sensitive to noise in the extracted profile.
- Structured JSON is probably the wrong container; natural-language guidance is the untested alternative.
- Current metrics can't detect style transfer anyway (see Q6) — an LLM-as-judge protocol would be needed to know whether style *did* transfer while n-gram metrics stayed flat.

---

## Q5 · One model, three input scenarios — how do you train it?

This is the core technical problem of the project.

### The setup

| | text-only | img-only | both |
|---|---|---|---|
| **Single training** (one model per input type) | **best** | — | — |
| **Mix training** (all three jointly) | ↓ | ↓ | **best** |

Framed as **3-way multi-task learning with gradient interference**.

### Round 1 — training-recipe fixes

| Method | Idea | Outcome |
|---|---|---|
| Ratio training (both:text:img = 2:2:1) | rebalance the data mixture | modest gains; on VLM, ratio > mix > ReviewLLM for both / img-only |
| Curriculum (text-first, freeze→unfreeze projector) | build the text backbone, then add vision | no clear gain |
| Modality dropout (drop image more often than text on `both` samples) | protect the strong modality | designed, limited effect |
| Dynamic loss weighting (Uncertainty / GradNorm / DWA) | rebalance gradients by task | evaluated; uncertainty weighting is *anti*-intuitive here — it down-weights tasks that are genuinely under-learned, not just noisy |

### Round 2 — architectural separation

**(a) Modality-specific bottleneck MLPs (hard-routing MoE).** Inserted between the last transformer layer and `lm_head`:

```
h_out = h + MLP_{input_type}(h)     MLP: Linear(hidden, bottleneck) → ReLU → Linear(bottleneck, hidden)
```

Deliberately **no router, no softmax gate, no load-balancing loss** — the input type is known, so routing is sample-level and hard. Batch supports per-sample expert selection.

**❌ Negative result 4 — mixed at best.** On Arts it helps text-only (ROUGE-2 +17.8%) but sharply hurts img-only (BLEU −43.5%); on Sports and Toys it lowers most metrics.

**(b) Modality-specific LM heads.** Shared backbone, three `lm_head`s (text / img / both), each initialized from the pretrained `lm_head` rather than randomly.

**❌ Negative result 5 — worse still.** Three reasons, and this is the most transferable finding in the project:

1. **The outputs are all natural-language reviews.** The vocabulary distributions across the three input types are nearly identical. Branching at the *vocab projection* layer cannot capture a modality-specific signal that doesn't exist in vocab space.
2. **All three heads share one LoRA.** Gradients from the three modalities superimpose directly on it, with no conflict-resolution mechanism whatsoever.
3. **Scale mismatch: 1.58B (three heads) vs 3.4M (LoRA) — ~500×.** The two cannot converge stably under one optimizer.

> **Design principle extracted:** in LLM multi-task learning, task-specific capacity belongs in the **hidden representation space** (adapters, LoRA experts, MoE FFN in the middle layers), with a **shared output head**. The vocab projection is the least expressive place to branch.

**Directions this pointed to** (surveyed, not fully executed): Mixture-of-LoRA-Experts (HydraLoRA NeurIPS'24, MTL-LoRA AAAI'25, MALoRA NAACL'25, R-LoRA EMNLP'25), shared+private LoRA (IJCAI'25), gradient surgery (PCGrad, CAGrad, MoCo), FAMO loss weighting.

---

## Q6 · Can distillation and RL recover what multi-task training loses?

### Stage 1 — Multi-teacher On-Policy Distillation (MOPD)

**Idea**: the single-trained modality experts are each better than the mixed model on their own modality. Distill all of them into one student.

**Objective** — student samples its own rollouts, teachers score them:

$$\mathcal{L}_{\text{OPD}}(\theta)=\mathbb{E}_{x\sim\mathcal D,\ \hat y\sim\pi_\theta(\cdot|x)}\left[\sum_{t=1}^{T}D_{\text{KL}}\!\left(\pi_{\text{expert}_{m(x)}}(\cdot\,|\,x,\hat y_{<t})\ \middle\|\ \pi_\theta(\cdot\,|\,x,\hat y_{<t})\right)\right]$$

- Teachers: `T=3` (π_both, π_txt, π_img), later `T=4` (+ π_mix).
- Student: LLaMA-3.1-8B + LoRA.
- `m(x)` selected either **by input type** (hard modality routing) or **per-sample** (best of the 4 by normalized metric sum on train).
- Divergence: **reverse KL**, `D_KL(student ‖ teacher)` — mode-seeking. *(Note for anyone reading the slides: the operand order in the rendered formula reads teacher-first; the implementation is student-first. Worth fixing before reuse.)*

**Teacher-agreement statistics** (why per-sample selection is fragile):

| Score gap between the two candidate teachers | share of samples |
|---|---|
| ≈ identical | ~30% |
| 0.01 – 0.1 | ~57% |
| > 0.1 | ~13% |

**❌ Negative result 6 — MOPD underperformed every baseline.** Five diagnosed causes:

1. **(fatal) Best-teacher labels are built from ground-truth metrics.** Selecting a teacher by its BLEU/ROUGE against the GT means the label encodes *which teacher overfit this training sample best*. The student learns to reproduce overfitting patterns, which don't transfer.
2. **Cross-modality pairing is pure noise.** If a text-only sample is routed to the `both` teacher, that teacher is being run **out of distribution** — it was never trained on inputs missing the image description. Distilling OOD behavior is distilling noise. Only same-modality routing is defensible.
3. **Per-sample metric sums are dominated by variance.** BLEU's coefficient of variation on single samples is ~50%. With ~30% of samples having near-identical teacher scores, the "best teacher" label is largely a coin flip that the student is then forced to imitate.
4. **The teachers have systematically different output distributions** (style, length, phrasing). Per-sample switching gives the student an incoherent mixture target. This is precisely why mainstream multi-teacher work (FuseLLM, FuseChat+WRPO, MoDE-CoTD) fuses at the **logits level** rather than *selecting* one teacher per sample.
5. **Reverse KL is mode-seeking.** With the target distribution jumping between teachers, there is no stable mode to converge to.

**Implementation trap**: when the best teacher *is* the mix-training checkpoint and the student is initialized from that same checkpoint, `KL ≈ 0` and those samples contribute no gradient at all. Fix: initialize the student from the **base model** instead.

### ⭐ The finding that reframed the evaluation

MOPD **loses under greedy decoding but wins under sampled decoding.** Averaged over 5 sampled outputs (`temp=0.8, top_p=0.9`), per-sample MOPD is the best model on every img-only metric and on text-only BLEU/ROUGE-2.

> On-policy distillation improves the **quality of the output distribution**, not the single highest-probability trajectory. Merging several teachers' generation modes widens the distribution to cover more good candidates — which sampling exposes and greedy decoding hides.
>
> Every review-generation paper reports greedy only. Under the field's own protocol, this method looks like a failure.

### Stage 2 — GRPO with teacher fallback

$$\mathcal{L}=\mathcal{L}_{\text{GRPO}}+\mathbb I_1\lambda\mathcal{L}_{\text{KD}}+\mathbb I_2\gamma\mathcal{L}_{\text{CE}},\qquad \mathbb I_1,\mathbb I_2\in\{0,1\}$$

Per-sample routing:

```
student generates G rollouts → mean reward
   ├─ reward ≥ threshold ────────────────→  L = GRPO                (student is already good)
   └─ reward < threshold
        ├─ some teacher scores higher ───→  L = GRPO + KD           (dense supervision from best teacher)
        └─ no teacher is better ─────────→  L = GRPO + CE           (fall back to ground truth)
```

- `L_GRPO`: group-normalized advantage `Â_i = (r_i − μ_R)/(σ_R + δ)`, clipped ratio `ρ_i,t`, plus `β·D_KL[π_θ ‖ π_ref]` with `π_ref` = the Stage-1 checkpoint.
- `L_KD`: on-policy, over the **same G rollouts**, `D_KL[π_θ ‖ π_{E*}]` against the selected best teacher.
- `L_CE`: standard token-level CE on the ground-truth review.
- **Reward**: per-metric min-max normalization (computed at p3/p97 on the train set, per dataset × modality), then summed. **Threshold**: 70th percentile of the train-set reward distribution.

### ✅ Positive result 2 — the headline numbers

**Best configuration: mix-SFT → GRPO** (Arts, greedy decoding; Δ vs. the strongest baseline for each metric, including Qwen3-VL):

| | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | CIDEr |
|---|---|---|---|---|---|---|
| **Both** | **+32.0%** | **+16.4%** | **+27.7%** | **+13.4%** | **+8.3%** | **+20.1%** |
| **Text-only** | **+10.2%** | **+9.7%** | **+18.5%** | **+9.1%** | **+9.3%** | **+2.0%** |
| **Img-only** | −3.3% | +4.5% | +1.7% | +1.0% | +3.5% | +11.8% |

Pushing `both` harder — **Stage-1 (input-type routing) → GRPO**: BLEU **+39.3%**, ROUGE-2 **+22.3%**, CIDEr **+26.2%**.

**❌ Negative result 7 — Stage 1 is not necessary.** The best-balanced configuration skips MOPD entirely and runs GRPO directly on the mix-SFT checkpoint. A two-stage pipeline that we spent two months on is outperformed by its own second stage.

### Reward-design lessons (all learned the hard way)

| Issue | What happened | Fix |
|---|---|---|
| **CIDEr is corpus-level** | Used per-sample it contributes almost nothing; CIDEr-only reward caused **model collapse** — the model emitted prompt repetitions instead of reviews, at every learning rate tried | Drop CIDEr from the student reward; 5-metric composite (BLEU + ROUGE-1/2/L + BERTScore) |
| **BLEU needs smoothing** | Unsmoothed sentence BLEU is 0 for most short reviews, so the normalization percentiles were wrong too | `smooth=True` in both the reward and the p3/p97 stats |
| **Single-metric rewards trade off hard** | BLEU-only: `both` BLEU +17.0% but img-only CIDEr **−42.5%**. CIDEr-only: everything collapses (−93% to −100%) | Composite reward, min-max normalized per metric |
| **Narrow LR window** | Too high → eval SFT loss overfits; too low → metrics don't move at all. True for GRPO, PPO, and the conditional variant | — |
| **PPO instability** | Value model hard to train, very batch-size sensitive | GRPO preferred |
| **Wrong model-selection criterion** | For a long time we selected checkpoints by eval **loss** rather than eval **reward** | — |
| **1-sample vs 5-sample** | GRPO-trained models consistently *underperform* at 1 sample and *outperform* at 5 — characteristic of RL favoring Best-of-N. RFT / rejection-sampling FT would target 1-sample quality directly | open |

### Why img-only stays weak — diagnosed, not hand-waved

The gap is **metric-specific**, which is the clue:

| Model | Both CIDEr | Img-only CIDEr | gap |
|---|---|---|---|
| baseline (revllm mix) | 0.3291 | 0.3370 | **+0.008** (img higher!) |
| after Stage 1 | 0.3431 | 0.2875 | −0.056 |
| after Stage 2 (GRPO) | 0.4153 | 0.2425 | **−0.173** |

The two trajectories **diverge** during training. BERTScore, meanwhile, stays flat across all three modalities (0.26–0.30) — so img-only output is *semantically fine*, just not hitting the right specific words.

**Zero content-word overlap with the reference:**

| | Stage 1 | Stage 2 |
|---|---|---|
| both | 30.0% | 22.7% |
| text-only | 34.0% | 21.3% |
| **img-only** | **41.4%** | **29.7%** |

GRPO compresses "completely off-target" from ~32% to ~22% for text and both, but img-only only reaches 29.7%. CIDEr's penalty for a total miss is a cliff, so the mean gets dragged.

**Case-level inspection** separated three distinct causes:

1. **Real modality bottleneck (fixable).** `EuTengHao 18Pcs Dotting Tools Kit Clay Pottery Carving…` — the title says *pottery*; the img-only model guesses *jewelry-making*. A small tool kit is genuinely ambiguous from a photo. Brand, category and intended use live in the title, not the image.
2. **Reference noise (not the model's fault).** Some references aren't reviews at all — *"Description says two which I had ordered. I only received one"*, *"What is the difference between fine tip and extra fine tip?"* Shipping complaints and pre-purchase questions are unpredictable from any modality and depress the CIDEr floor for *every* condition.
3. **False negatives in the overlap metric.** *"I absolutely love this Cricut and great buy"* vs *"…it has been a game changer"* — semantically aligned, zero lexical overlap. This is why BERTScore disagrees with CIDEr.

**The hallucination hypothesis did not survive.** We had assumed GRPO pushes the model to confidently invent specifics that img-only can't ground. The evidence doesn't support it — the errors are mostly *wrong-purpose guesses* and *noisy references*, not fabricated brands. Removing CIDEr from the reward changed img-only CIDEr by 0.0008 (0.2433 vs 0.2425), which independently confirms this isn't a reward-design artifact.

> **❌ Negative result 8 — img-only underperformance is an input information bottleneck, not a training-strategy failure.** GRPO doesn't cause it; GRPO raises the ceiling for text and both, and img-only simply can't follow.
>
> Implied fix, untested: give img-only a cheap textual prior — even a category tag or an OCR-aware caption prompt that extracts brand text from packaging — rather than a purely visual caption.

---

## Evaluation: the meta-finding

Half of this project's difficulty was the metrics.

1. **Per-sample BLEU/CIDEr variance is enormous** (BLEU CV ≈ 50%). Any decision built on per-sample metrics — teacher selection, reward shaping, "which config won" — is partly reading noise.
2. **CIDEr is a corpus-level metric.** Using it per-sample is a category error that cost us a reward-design cycle and one model collapse.
3. **Greedy vs. sampled decoding give opposite rankings** for distillation-trained models (sampled = mean over 5 draws at T=0.8, top-p=0.9). The field reports greedy only.
4. **Split hygiene.** Early mix-training runs had overlap between input types and between train/test. Re-splitting changed conclusions. We also had to define "better" explicitly: **all 5 metrics higher, or 4 higher and 1 tied.**
5. **Reference noise is systematic**, not incidental — roughly a fifth to a third of "total misses" are unanswerable references.
6. **Scale.** 600–1,300 training samples per dataset × modality cell amplifies every problem above.
7. **No metric measures personalization.** BLEU/ROUGE/BERTScore measure similarity to one reference. Whether the output sounds like *this user* is untested. We added **Visual-Entity Coverage / Precision / F1** to at least measure visual grounding directly; a style-discrimination metric is still missing.

---

## Negative results index

| # | Finding | Evidence |
|---|---|---|
| 1 | Embedding-level fusion (GMU gating + projector + soft prompts) loses to prompt-level text injection — the gate never becomes modality-aware and collapses to uniform averaging | gating-weight traces on 3 datasets × 3 modalities; two confounds ruled out |
| 2 | Title-based retrieval does not beat recency for history selection; the apparent win from retrieval+time-range vanishes once history count is controlled | 3 datasets × 5 history counts + 17-case study |
| 3 | LLM-extracted content/style preference profiles contribute nothing and induce forced, unnatural phrasing | 5-way ablation, 982 samples |
| 4 | Modality-specific bottleneck MLPs (hard-routing MoE before `lm_head`) degrade performance broadly | multi-dataset × multi-modality |
| 5 | Modality-specific LM heads are worse still — vocab distributions are near-identical across input types, and a 1.58B head vs 3.4M LoRA cannot co-converge | quantitative + mechanism |
| 6 | Per-sample best-teacher selection via GT metrics is unsound: it encodes overfitting, creates OOD cross-modality distillation, and is dominated by metric noise | 5-part diagnosis + teacher-agreement stats |
| 7 | Stage 1 (MOPD) is unnecessary — GRPO directly on the mix-SFT checkpoint is the best-balanced configuration | full comparison table |
| 8 | img-only weakness is an input information bottleneck, not a training or reward problem | divergence analysis + zero-overlap rates + case study + reward ablation |
| 9 | Adding raw product images to a text review generator hurts on average; adding descriptions helps on only ~half of categories | 8–10 categories |

## Positive results index

| # | Finding |
|---|---|
| 1 | **img2txt > end-to-end VLM.** Converting product images to natural-language descriptions and feeding an LLM beats a native VLM backbone on every modality, at a fraction of the input length and training cost. |
| 2 | **mix-SFT + GRPO is the best single model.** Both: BLEU +32.0%, ROUGE-2 +27.7%, BERTScore +8.3%; text-only: ROUGE-2 +18.5%, BERTScore +9.3% over the strongest baseline; from one checkpoint serving all three input scenarios. |
| 3 | **History supplies writing templates, not product knowledge** — a reframing of what "personalization" means in this task, supported by case-level analysis. |
| 4 | **Recency beats semantic relevance** for history selection, because recent reviews carry the user's ongoing project context, current vocabulary, and latest sentence patterns. |
| 5 | **On-policy distillation shapes the output distribution more than the greedy trajectory** — per-sample MOPD is weakest under greedy decoding but strongest under sampled decoding on img-only. |
| 6 | **Real review data is modality-heterogeneous**: among the 3,305 of 13,900 reviews with a strictly best input configuration, ≈1/3 favor text, ≈1/3 image, ≈1/3 both. |

---

## Stack

**Models** LLaMA-3 / LLaMA-3.1-8B · Qwen2.5-VL · Qwen3-VL-8B-Instruct · LLaVA · BLIP-2 (Q-Former) · CLIP · BERT · GPT-4.1-mini
**Training** LoRA · full-parameter tuning · single / mix / ratio / curriculum training · on-policy distillation (reverse KL) · GRPO · PPO · LLaMA-Factory
**Architecture** GMU multi-head gating · cross-attention resampler with learnable queries · MLP projector / soft prompts · sample-level hard-routing MoE · modality-specific LM heads · learnable missing-modality vectors
**Infra** NYU HPC (Slurm, Singularity) · H200 / H100 / A100 · 8-bit teacher loading · teacher-key batch grouping
**Eval** BLEU · ROUGE-1/2/L · BERTScore · chrF++ · CIDEr (pycocoevalcap, with IDF-corpus tracking) · Distinct-1/2 · Visual-Entity Coverage/Precision/F1 · multi-draw sampling with per-draw std

---

---

See [docs/PLANNING.md](docs/PLANNING.md) for benchmark design notes, GitHub setup guide, and CV bullets.
