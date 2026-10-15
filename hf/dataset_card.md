---
license: other
license_name: amazon-reviews-2023
license_link: https://amazon-reviews-2023.github.io/
language:
- en
task_categories:
- text-generation
tags:
- personalization
- review-generation
- multimodal
- recommendation
pretty_name: MMRev — Multimodal Personalized Review Generation
size_categories:
- 1K<n<10K
configs:
- config_name: arts_both
  data_files:
  - split: train
    path: data/Arts/both/train.jsonl
  - split: test
    path: data/Arts/both/test.jsonl
- config_name: arts_text_only
  data_files:
  - split: train
    path: data/Arts/text_only/train.jsonl
  - split: test
    path: data/Arts/text_only/test.jsonl
- config_name: arts_img_only
  data_files:
  - split: train
    path: data/Arts/img_only/train.jsonl
  - split: test
    path: data/Arts/img_only/test.jsonl
- config_name: sports_both
  data_files:
  - split: train
    path: data/Sports/both/train.jsonl
  - split: test
    path: data/Sports/both/test.jsonl
- config_name: sports_text_only
  data_files:
  - split: train
    path: data/Sports/text_only/train.jsonl
  - split: test
    path: data/Sports/text_only/test.jsonl
- config_name: sports_img_only
  data_files:
  - split: train
    path: data/Sports/img_only/train.jsonl
  - split: test
    path: data/Sports/img_only/test.jsonl
- config_name: toys_both
  data_files:
  - split: train
    path: data/Toys/both/train.jsonl
  - split: test
    path: data/Toys/both/test.jsonl
- config_name: toys_text_only
  data_files:
  - split: train
    path: data/Toys/text_only/train.jsonl
  - split: test
    path: data/Toys/text_only/test.jsonl
- config_name: toys_img_only
  data_files:
  - split: train
    path: data/Toys/img_only/train.jsonl
  - split: test
    path: data/Toys/img_only/test.jsonl
---

# MMRev: Multimodal Personalized Review Generation

Data accompanying *What Helps and What Hurts: A Systematic Study of Multimodal Personalized Review Generation* (technical report, link TBA). Code: https://github.com/{HF_USER}/mmrev · Model: [`{HF_USER}/mmrev-llama3.1-8b-grpo`](https://huggingface.co/{HF_USER}/mmrev-llama3.1-8b-grpo)

**Task.** Given a user's purchase history (past items with the user's reviews and ratings) and a target item with its rating, generate the review this user would write.

**Three input scenarios.** Each sample belongs to exactly one scenario, which determines what the model may see about the target item:

| Scenario | Target-item fields used |
|---|---|
| `both` | title, text description, image (as a caption) |
| `text_only` | title, text description |
| `img_only` | image (as a caption) only — the title is hidden |

Scenario membership comes from a data-filtering experiment: a fine-tuned Qwen2.5-VL generated each review from text only, image only, or both, and each sample was placed in the scenario whose input configuration best explained its review. The three subsets are therefore **disjoint and differ in content**; compare models within a scenario, not across scenarios.

## Subsets

| Category | both (train/test) | text_only | img_only |
|---|---|---|---|
| Arts (Arts, Crafts & Sewing) | 750 / 150 | 707 / 141 | 641 / 128 |
| Sports (Sports & Outdoors) | 898 / 180 | 891 / 178 | 840 / 168 |
| Toys (Toys & Games) | 1,216 / 243 | 1,190 / 238 | 1,230 / 246 |

```python
from datasets import load_dataset
ds = load_dataset("{HF_USER}/mmrev-data", "arts_both")      # configs: {arts,sports,toys}_{both,text_only,img_only}
print(ds["train"][0]["reference_review"])
```

## Fields (`data/<Category>/<scenario>/<split>.jsonl`)

| Field | Description |
|---|---|
| `id` | Row index into `raw/<Category>/final.jsonl` (also the key used by `splits/*.npz` and `image_descriptions/`) |
| `dataset`, `scenario`, `split` | Category, input scenario, train/test |
| `user_id` | Anonymized user ID from Amazon Reviews 2023 |
| `history` | List of past purchases, oldest → newest: `product_title`, `parent_asin`, `rating`, `timestamp`, `review`. The paper uses the **5 most recent**. |
| `target_product_title`, `target_product_desc`, `target_parent_asin`, `target_image_url`, `target_rating`, `target_timestamp` | Target item. Fields not visible in the sample's scenario must be dropped when building the prompt. |
| `reference_review` | Ground-truth review |

Product descriptions are provided for target items only; history items carry titles, reviews, and ratings.

## Other files

| Path | Content |
|---|---|
| `raw/<Category>/final.jsonl` | Full preprocessed pool per category (all users after filtering), in the original nested format used by the training code |
| `splits/<Category>_<scenario>_split.npz` | Arrays `train_random` / `test_random` of row indices into `final.jsonl`. `<Category>_mix_split.npz` is the union of the three scenarios, used for mix-SFT and GRPO training |
| `image_descriptions/<Category>/*.jsonl` | Qwen2.5-VL captions of product images: `{"id", "image_descriptions": [{"image", "description"}]}`, keyed by `id` |
| `teacher_scores/per_sample_detail_<Category>_train.csv` | Per-sample metrics (BLEU, ROUGE-1/2/L, BERTScore, CIDEr) of the single-scenario and mix-scenario SFT models on the training set; used for reward normalization statistics and teacher look-up in Stage II |

Image captions were generated with this prompt:

```
Look at this product image and describe it in 2-4 natural sentences. Include:
- What the product is (category, brand/name if visible)
- Appearance (color, material, shape/design)
- Any visible text or labels
- Notable features or details
Requirements: only describe what is actually visible; be concise; do not guess.
```

## Construction

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab). Users with 10–30 reviews were kept; duplicate, very short, and templated reviews were removed, as were items lacking the fields their scenario requires and samples exceeding the LLaMA-3.1 context window. Scenario membership was assigned with the data-filtering experiment described above; the three scenario subsets do not overlap.

## License and intended use

Derived from Amazon Reviews 2023; use is subject to the terms of the original dataset. Intended for research on personalized and multimodal text generation. Do not use to produce or post synthetic reviews on real platforms.

## Citation

```bibtex
@techreport{chen2026mmrev,
  title  = {What Helps and What Hurts: A Systematic Study of Multimodal Personalized Review Generation},
  author = {Chen, Chuqiao and others},
  year   = {2026},
  institution = {New York University Shanghai}
}
@article{hou2024bridging,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}
```
