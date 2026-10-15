---
base_model: meta-llama/Llama-3.1-8B
library_name: peft
license: llama3.1
language:
- en
pipeline_tag: text-generation
tags:
- lora
- peft
- grpo
- personalization
- review-generation
- multimodal
datasets:
- {HF_USER}/mmrev-data
---

# MMRev: LLaMA-3.1-8B + LoRA for multimodal personalized review generation (mix-SFT → GRPO)

A LoRA adapter for [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) that writes a product review in the voice of a specific user, given that user's recent reviews and a target product. **One adapter serves three input scenarios**:

| Scenario | What the model sees about the target item |
|---|---|
| `both` | title, text description, image description |
| `text_only` | title, text description |
| `img_only` | image description only (no title) |

Product images are provided as short natural-language captions generated offline by Qwen2.5-VL, so the model is text-only.

Paper: *What Helps and What Hurts: A Systematic Study of Multimodal Personalized Review Generation* (technical report, link TBA). Code: https://github.com/{HF_USER}/mmrev

## Training

| Step | Details |
|---|---|
| mix-SFT | LoRA (r=8, α=16, q_proj/v_proj) on the union of the three scenarios, Amazon Reviews 2023 *Arts, Crafts & Sewing* (2,098 training samples); LLaMA-Factory, lr 5e-6, cosine. This is the "ReviewLLM mix" baseline in the paper. |
| GRPO | Continues training the **same** LoRA from the mix-SFT model, on the same union of three scenarios. G=4 rollouts (T=1.0, ≤256 new tokens), clip ε=0.2, KL β=0.01 (k3 estimator) to the mix-SFT model, batch 2 (effective 8), peak lr 5e-6 with linear warmup/decay over a 40-epoch schedule. Reward = sum of per-metric, per-scenario min–max-normalized BLEU (smoothed), ROUGE-1/2/L, BERTScore-F1, CIDEr. |

This repository contains the **GRPO checkpoint at step 3,147 (epoch 12)**, the checkpoint with the highest reward on the evaluation split. No separate validation split was held out, so the evaluation split is the test split; the reported numbers therefore involve test-set checkpoint selection and may be somewhat optimistic. Optimizer/scheduler states are omitted.

The original project pipeline was SFT → Stage 1 (multi-teacher on-policy distillation) → Stage 2 (GRPO). The released model skips Stage 1: GRPO directly on mix-SFT gave the best results across all three scenarios.

## Results (Arts test set, greedy decoding)

Δ is relative to the strongest baseline for each metric (ReviewLLM single/mix, Qwen2.5-VL mix, Qwen3-VL mix).

| Scenario | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | CIDEr |
|---|---|---|---|---|---|---|
| both | .0557 (+32.0%) | .2776 (+16.4%) | .0848 (+27.7%) | .2190 (+13.4%) | .3082 (+8.3%) | .7603 (+20.1%) |
| text_only | .0646 (+10.2%) | .2528 (+9.7%) | .0558 (+18.5%) | .1961 (+9.1%) | .2922 (+9.3%) | .6905 (+2.0%) |
| img_only | .0436 (−3.3%) | .2647 (+4.5%) | .0703 (+1.7%) | .2105 (+1.0%) | .3082 (+3.5%) | .7719 (+11.8%) |

BERTScore is RoBERTa-large F1 with baseline rescaling.

## Usage

The tokenizer adds one token (`<pad>`, id 128256), so **resize the base model's embeddings before loading the adapter**.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER = "{HF_USER}/mmrev-llama3.1-8b-grpo"
tok  = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto")
base.resize_token_embeddings(len(tok))            # 128,257
model = PeftModel.from_pretrained(base, ADAPTER).eval()

def build_prompt(history, target, scenario):
    """history: list of dicts with product_title, review, rating (and optional image_description), oldest → newest.
       target: dict with product_title, rating, product_desc, image_description.
       scenario: 'both' | 'text_only' | 'img_only'."""
    provided = {"both": "the provided product and image descriptions",
                "text_only": "the provided product descriptions",
                "img_only": "the provided image descriptions"}[scenario]
    instruction = ("Given input, including the items purchased by the user historically, and corresponding reviews and ratings. "
                   f"Please generate a review for the new item purchased by the user, drawing from their historical reviews, ratings, and {provided}. "
                   "Keep in mind that lower ratings typically result in poorer reviews.")
    lines = []
    for i, h in enumerate(history[-5:], 1):                      # 5 most recent
        s = f"({i}) {h['product_title']}."
        if scenario != "text_only" and h.get("image_description"):
            s += f" %Image description: {h['image_description']}."
        s += f" %Review: {h['review']} %Rating: {h['rating']:.1f}."
        lines.append(s)
    t = target
    if scenario == "both":
        new = f"{t['product_title']}. %Rating: {t['rating']:.1f}. %Product description: {t['product_desc']}. %Image description: {t['image_description']}."
    elif scenario == "text_only":
        new = f"{t['product_title']}. %Rating: {t['rating']:.1f}. %Product description: {t['product_desc']}."
    else:
        new = f"%Rating: {t['rating']:.1f}. %Image description: {t['image_description']}."
    inp = f"Item purchased by users historically: {' '.join(lines)}\n\n### The user purchased a new item:\n{new}"
    return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"

prompt = build_prompt(history, target, "both")
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

Data in the expected format (including image descriptions) is available at [`{HF_USER}/mmrev-data`](https://huggingface.co/datasets/{HF_USER}/mmrev-data).

## Limitations

- Trained and evaluated on one category (Arts); other categories may need re-training.
- Small training set (about 2.1k Arts samples); evaluation uses single-reference automatic metrics without human judgment.
- Gains in the `img_only` scenario are small on categories where product photos say little about the product's purpose.
- Outputs imitate user writing style and may reproduce opinions or phrasing from the user history; do not use to fabricate reviews on real platforms.

## License

Use of this adapter is subject to the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/). Training data derives from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
