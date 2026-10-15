# Results

All numbers are taken from the project tracking spreadsheet. Greedy decoding unless noted.
Δ is relative to the **strongest baseline for that metric and scenario**, so the reference baseline can differ across columns.
BERTScore is baseline-rescaled (RoBERTa-large) unless noted.
CIDEr values depend on the IDF corpus; values computed under different corpora are not comparable (Qwen3-VL CIDEr is therefore omitted).

## Main results — Arts (mix-SFT → GRPO; reported GRPO checkpoint: step 3147)

GRPO starts from the mix-SFT model (ReviewLLM mix), is trained on the union of the three scenarios, and is evaluated on each scenario separately. The reported checkpoint is step 3147 (epoch 12 of a 40-epoch schedule), which has the highest reward on the evaluation split. No separate validation split was held out, so the evaluation split is the test split: this is test-set checkpoint selection, and the GRPO numbers may be somewhat optimistic.

All CIDEr values in this table use the same IDF corpus; VLM mix was not evaluated under this protocol.

| Scenario | Model | BLEU | R-1 | R-2 | R-L | BERTScore | CIDEr |
|---|---|---|---|---|---|---|---|
| both | ReviewLLM single | .0354 | .2275 | .0535 | .1780 | .2634 | .5385 |
| both | ReviewLLM mix (= mix-SFT, GRPO starting point) | .0422 | .2384 | .0650 | .1914 | .2847 | .6330 |
| both | VLM mix (Qwen2.5-VL) | .0348 | .2270 | .0557 | .1826 | .2658 | – |
| both | Qwen3-VL mix | .0417 | .2335 | .0664 | .1931 | .2665 | .5753 |
| both | **mix-SFT + GRPO (ours)** | **.0557** | **.2776** | **.0848** | **.2190** | **.3082** | **.7603** |
| both | Δ | +32.0% | +16.4% | +27.7% | +13.4% | +8.3% | +20.1% |
| text | ReviewLLM single | .0412 | .2305 | .0471 | .1782 | .2496 | .6471 |
| text | ReviewLLM mix | .0586 | .2260 | .0416 | .1789 | .2590 | .6767 |
| text | VLM mix | .0571 | .2280 | .0440 | .1798 | .2673 | – |
| text | Qwen3-VL mix | .0404 | .2237 | .0462 | .1741 | .2474 | .6481 |
| text | **ours** | **.0646** | **.2528** | **.0558** | **.1961** | **.2922** | **.6905** |
| text | Δ | +10.2% | +9.7% | +18.5% | +9.1% | +9.3% | +2.0% |
| img | ReviewLLM single | .0346 | .2362 | .0579 | .1934 | .2784 | .6650 |
| img | ReviewLLM mix | .0451 | .2491 | .0691 | .2085 | .2978 | .6907 |
| img | VLM mix | .0383 | .2534 | .0627 | .2006 | .2879 | – |
| img | Qwen3-VL mix | .0408 | .2359 | .0583 | .1950 | .2804 | .6433 |
| img | **ours** | .0436 | **.2647** | **.0703** | **.2105** | **.3082** | **.7719** |
| img | Δ | −3.3% | +4.5% | +1.7% | +1.0% | +3.5% | +11.8% |

## GRPO applied to four SFT initializations — Arts

Value after GRPO (relative change vs. the same model before GRPO). ROUGE-1 improves in all 12 cells.

| SFT init | both BLEU | both R-1 | both R-2 | text BLEU | text R-1 | text R-2 | img BLEU | img R-1 | img R-2 |
|---|---|---|---|---|---|---|---|---|---|
| ReviewLLM single | .0481 (+36%) | .2655 (+17%) | .0686 (+28%) | .0595 (+44%) | .2520 (+9%) | .0519 (+10%) | .0466 (+35%) | .2517 (+7%) | .0690 (+19%) |
| ReviewLLM mix | .0557 (+32%) | .2776 (+16%) | .0848 (+30%) | .0646 (+10%) | .2528 (+12%) | .0558 (+34%) | .0436 (−3%) | .2647 (+6%) | .0703 (+2%) |
| VLM single | .0350 (+0%) | .2582 (+15%) | .0541 (−1%) | .0579 (+20%) | .2470 (+13%) | .0503 (+14%) | .0457 (+58%) | .2669 (+19%) | .0697 (+31%) |
| VLM mix | .0440 (+26%) | .2696 (+19%) | .0702 (+26%) | .0633 (+11%) | .2588 (+14%) | .0528 (+20%) | .0364 (−5%) | .2690 (+6%) | .0666 (+6%) |

## Toys

| Scenario | Model | BLEU | R-1 | R-2 | R-L | BERTScore |
|---|---|---|---|---|---|---|
| both | ReviewLLM single | .0398 | .2381 | .0693 | .1869 | .2621 |
| both | ReviewLLM mix | .0450 | .2418 | .0775 | .1936 | .2717 |
| both | **ours** | .0465 (+3.3%) | .2741 (+13.4%) | .0774 (−0.1%) | .2097 (+8.3%) | .2964 (+9.1%) |
| text | ReviewLLM single | .0403 | .2358 | .0696 | .1892 | .2714 |
| text | ReviewLLM mix | .0365 | .2305 | .0635 | .1838 | .2756 |
| text | **ours** | .0411 (+2.0%) | .2565 (+8.8%) | .0671 (−3.6%) | .1960 (+3.6%) | .2950 (+7.0%) |
| img | ReviewLLM single | .0433 | .2606 | .0867 | .2060 | .2879 |
| img | ReviewLLM mix | .0412 | .2546 | .0825 | .2004 | .2844 |
| img | **ours** | .0470 (+8.5%) | .2955 (+13.4%) | .0927 (+6.9%) | .2290 (+11.2%) | .3211 (+11.5%) |

## GRPO-stage ablations — Arts

"Init" is the model GRPO starts from: mix-SFT, or a Stage-1 (MOPD) model from the original SFT → Stage 1 (MOPD) → Stage 2 (GRPO) pipeline.

| Init | Objective | both BLEU | both R-2 | both BS | text BLEU | text R-2 | text BS | img BLEU | img R-2 | img BS |
|---|---|---|---|---|---|---|---|---|---|---|
| mix-SFT | PPO + KD/CE | .0114 | .0442 | .1901 | .0082 | .0360 | .1780 | .0154 | .0449 | .2327 |
| mix-SFT | GRPO + KD/CE | .0407 | .0602 | .2784 | .0477 | .0348 | .2353 | .0428 | .0645 | .2889 |
| MOPD (per-sample) | GRPO + KD/CE | .0448 | .0633 | .2783 | .0532 | .0469 | .2557 | .0286 | .0569 | .2810 |
| MOPD (by scenario) | GRPO + KD/CE | .0472 | .0685 | .2747 | .0490 | .0484 | .2576 | .0428 | .0624 | .2878 |
| MOPD (by scenario) | GRPO | .0588 | .0795 | .2813 | .0572 | .0541 | .2757 | .0402 | .0669 | .2895 |
| **mix-SFT** | **GRPO (ours)** | .0557 | **.0848** | **.3082** | **.0646** | **.0558** | **.2922** | **.0436** | **.0703** | **.3082** |

## Reward ablation — Arts (GRPO + KD/CE objective; relative to composite)

| Reward | both BLEU | both CIDEr | text R-2 | text BS | img BLEU | img CIDEr |
|---|---|---|---|---|---|---|
| composite | .0407 | .2690 | .0348 | .2353 | .0428 | .3326 |
| BLEU only | +17.0% | +14.0% | +14.9% | +3.3% | −34.3% | −42.5% |
| ROUGE-2 only | −1.0% | −11.1% | +22.1% | +3.5% | −59.1% | −49.2% |
| BERTScore only | −5.2% | −8.1% | +24.4% | +11.3% | −30.6% | −36.6% |
| CIDEr only | −93.6% | −100% | −69.5% | BERTScore < 0 | −93.2% | −100% (collapse) |

## Multi-teacher on-policy distillation — Arts

| Model | both BLEU | both R-2 | both BS | text BLEU | text R-2 | text BS | img BLEU | img R-2 | img BS |
|---|---|---|---|---|---|---|---|---|---|
| ReviewLLM mix | .0422 | .0650 | .2847 | .0586 | .0416 | .2590 | .0451 | .0691 | .2978 |
| MOPD, per-sample teacher | .0448 | .0625 | .2768 | .0531 | .0465 | .2553 | .0286 | .0566 | .2812 |
| MOPD, per-sample, mix init | .0437 | .0654 | .2829 | .0574 | .0455 | .2555 | .0437 | .0655 | .2896 |
| MOPD, by-scenario teacher | .0474 | .0691 | .2773 | .0489 | .0481 | .2584 | .0428 | .0625 | .2868 |

## Scenario-specific capacity — Arts (BERTScore not rescaled)

| Model | both BLEU | both R-2 | text BLEU | text R-2 | img BLEU | img R-2 |
|---|---|---|---|---|---|---|
| ReviewLLM mix | .0422 | .0650 | .0586 | .0416 | .0451 | .0691 |
| + residual MLPs (MoE) | .0398 (−5.7%) | .0646 (−0.6%) | .0595 (+1.5%) | .0490 (+17.8%) | .0255 (−43.5%) | .0556 (−19.5%) |
| + per-scenario LM heads | .0219 (−48.1%) | .0401 (−38.3%) | .0196 (−66.6%) | .0398 (−4.3%) | .0226 (−49.9%) | .0449 (−35.0%) |
