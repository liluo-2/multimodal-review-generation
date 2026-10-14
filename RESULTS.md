# Results

All numbers are on **Arts** (greedy decoding, 1 sample) unless noted. Source: experimental tracking spreadsheet, 7.7 final table.

## Main comparison (greedy, 1 sample)

### Arts

| Model | Both BLEU | Both R-1 | Both R-2 | Both R-L | Both BS | Both CIDEr | Text BLEU | Text R-1 | Text R-2 | Text R-L | Text BS | Text CIDEr | Img BLEU | Img R-1 | Img R-2 | Img R-L | Img BS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mix-SFT + GRPO only** | **0.0511** | **0.2695** | **0.0737** | **0.2128** | **0.3012** | 0.3316 | **0.0639** | **0.2573** | **0.0579** | **0.1968** | **0.2891** | 0.2103 | 0.0442 | 0.2613 | 0.0678 | 0.2073 | 0.3060 |
| stg1 (input-type) + GRPO only | 0.0588 | 0.2574 | 0.0795 | 0.2045 | 0.2813 | **0.4153** | 0.0572 | 0.2480 | 0.0541 | 0.1910 | 0.2757 | 0.2181 | 0.0402 | 0.2529 | 0.0669 | 0.2009 | 0.2895 |
| mix-SFT + GRPO (CIDEr+BLEU smooth) | 0.0526 | 0.2811 | 0.0793 | 0.2197 | 0.3102 | — | 0.0673 | 0.2668 | 0.0622 | 0.2025 | 0.2976 | — | 0.0414 | 0.2654 | 0.0683 | 0.2102 | 0.3088 |
| ReviewLLM mix (baseline) | 0.0422 | 0.2384 | 0.0650 | 0.1914 | 0.2847 | 0.3291 | 0.0586 | 0.2260 | 0.0416 | 0.1789 | 0.2590 | 0.2046 | 0.0451 | 0.2491 | 0.0691 | 0.2085 | 0.2978 |
| ReviewLLM single (baseline) | 0.0354 | 0.2275 | 0.0535 | 0.1780 | 0.2634 | 0.2468 | 0.0412 | 0.2305 | 0.0471 | 0.1782 | 0.2496 | 0.2220 | 0.0346 | 0.2362 | 0.0579 | 0.1934 | 0.2784 |
| VLM mix (baseline) | 0.0348 | 0.2270 | 0.0557 | 0.1826 | 0.2658 | 0.2443 | 0.0571 | 0.2280 | 0.0440 | 0.1798 | 0.2673 | 0.2151 | 0.0383 | 0.2534 | 0.0627 | 0.2006 | 0.2879 |
| Qwen3-VL mix | 0.0417 | 0.2335 | 0.0664 | 0.1931 | 0.2665 | 0.5753 | 0.0404 | 0.2237 | 0.0462 | 0.1741 | 0.2474 | 0.6481 | 0.0408 | 0.2359 | 0.0583 | 0.1950 | 0.2804 |

**Δ (mix-SFT+GRPO vs ReviewLLM mix)**: Both BLEU **+21.1%**, R-1 **+13.0%**, R-2 **+13.4%**, R-L **+11.1%**, BS **+5.8%** · Text BLEU +9.0%, R-1 +11.6%, R-2 **+22.9%**, R-L +9.4%, BS **+8.1%** · Img BLEU −2.0%, R-1 +3.1%.

### Toys (partial — GRPO)

| Model | Both BLEU | Both R-1 | Both R-2 | Both R-L | Both BS |
|---|---|---|---|---|---|
| mix-SFT + GRPO | 0.0465 | 0.2741 | 0.0774 | 0.2097 | 0.2964 |
| ReviewLLM mix (baseline) | 0.0450 | 0.2418 | 0.0775 | 0.1936 | 0.2717 |

## Reward ablation (Stage 2 only, Arts, greedy)

| Reward | Both BLEU | Both CIDEr | Text R-2 | Img CIDEr | Note |
|---|---|---|---|---|---|
| composite (5 metrics) | 0.0407 | 0.2690 | 0.0348 | 0.3326 | default |
| BLEU only | 0.0476 (+17.0%) | 0.3066 | 0.0400 | 0.1911 (−42.5%) | img CIDEr collapses |
| ROUGE-2 only | 0.0403 | 0.2391 | 0.0425 (+22.1%) | 0.1691 (−49.2%) | |
| BERTScore only | 0.0386 | 0.2472 | 0.0433 (+24.4%) | 0.2108 (−36.6%) | |
| CIDEr only | 0.0026 (−93.6%) | 0.0000 (−100%) | 0.0106 | 0.0000 (−100%) | **model collapse** |

## Stage 1 (MOPD) variants

| Model | Both BLEU | Both CIDEr | Text BLEU | Text CIDEr | Img BLEU | Img CIDEr |
|---|---|---|---|---|---|---|
| MOPD per-sample (base init), greedy | 0.0448 | 0.3173 | 0.0531 | 0.2195 | 0.0286 | — |
| MOPD by-input-type (base init), greedy | 0.0474 | 0.3431 | 0.0489 | 0.2194 | 0.0428 | — |
| MOPD by-input-type, 3 heads | 0.0476 | 0.3511 | 0.0520 | 0.2011 | 0.0445 | — |
| MOPD by-input-type, img data×2 | 0.0498 | 0.3824 | 0.0484 | 0.2090 | 0.0431 | — |
| ReviewLLM mix (baseline) | 0.0422 | 0.3291 | 0.0586 | 0.2046 | 0.0451 | — |

## GRPO with CIDEr fix — latest (Arts)

From the final rows of the tracking spreadsheet (with BLEU smoothing + CIDEr normalization fix):

| Model | Both BLEU | Both R-1 | Both R-2 | Both R-L | Both BS | Text BLEU | Text R-1 | Text R-2 | Text R-L | Text BS | Img BLEU | Img R-1 | Img R-2 | Img R-L | Img BS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mix + GRPO (cider+bleu smooth) ckpt 16259 | 0.0526 | 0.2811 | 0.0793 | 0.2197 | 0.3102 | 0.0673 | 0.2668 | 0.0622 | 0.2025 | 0.2976 | 0.0414 | 0.2654 | 0.0683 | 0.2102 | 0.3088 |

**Δ vs baseline**: Both R-1 **+17.9%**, R-2 **+22.0%**, R-L **+14.8%**, BS **+8.9%** · Text R-1 **+15.7%**, R-2 **+32.1%** · Img R-1 +4.8%.

Note: this is likely the best overall checkpoint. The "mix + GRPO only" row (R129) uses a different reward configuration.
