#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接从 inference 输出的 jsonl（自带 id + output/outputs + ground_truth）算 metrics，
不依赖外部 reference_path / split_path，避免两套数据体系对不上的风险。

CIDEr 的 IDF 语料库通过 --cider_idf_pool_paths 传入同一个 run 的三个模态文件（用逗号分隔），
三个模态共用同一个 IDF，保证 CIDEr 在这三个模态之间是可比的；
不传的话，退化为只用当前这一个文件自己的 ground_truth 建 IDF（不同模态之间不可比，会打印警告）。
"""

import evaluate
import json
import pandas as pd
import numpy as np
import os
import argparse
import re
from collections import defaultdict
from transformers import AutoTokenizer
from bert_score import score as bertscore_score
from pycocoevalcap.cider.cider_scorer import CiderScorer as _CiderScorer


def compute_distinct_n(predictions, n=2):
    all_ngrams = []
    for pred in predictions:
        words = pred.strip().split()
        if len(words) < n:
            continue
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams) * 100


def build_cider_idf(references):
    scorer = _CiderScorer(n=4, sigma=6.0)
    for ref in references:
        scorer += (None, [ref.strip()])
    scorer.compute_doc_freq()
    scorer.ref_len = np.log(float(max(len(references), 1)))
    return scorer


def compute_cider_online(pred, ref, idf_scorer):
    try:
        scorer = _CiderScorer(n=4, sigma=6.0)
        scorer.document_frequency = idf_scorer.document_frequency
        scorer.ref_len = idf_scorer.ref_len
        scorer += (pred.strip(), [ref.strip()])
        return float(scorer.compute_cider()[0])
    except Exception:
        return 0.0


def compute_cider(predictions, references, idf_scorer=None):
    if idf_scorer is None:
        idf_scorer = build_cider_idf(references)
    scores = [compute_cider_online(p, r, idf_scorer)
              for p, r in zip(predictions, references)]
    return float(np.mean(scores)) if scores else 0.0


def compute_token_length_stats(texts, tokenizer):
    lengths = [len(tokenizer.tokenize(text.strip())) for text in texts]
    if not lengths:
        return {"mean": 0.0, "min": 0, "max": 0}
    return {"mean": float(np.mean(lengths)), "min": int(np.min(lengths)), "max": int(np.max(lengths))}


def compute_bertscore(predictions, references):
    P, R, F1 = bertscore_score(
        predictions, references,
        model_type="roberta-large", lang="en",
        rescale_with_baseline=True, verbose=True
    )
    return P.mean().item(), R.mean().item(), F1.mean().item()


def extract_visual_entities(text):
    visual_entities = set()
    text_lower = text.lower()

    visual_concepts = ['color', 'colour', 'shape', 'size', 'material', 'texture', 'pattern',
                      'appearance', 'design', 'style', 'look', 'finish', 'surface']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white',
              'gray', 'grey', 'brown', 'beige', 'tan', 'navy', 'silver', 'gold']
    size_patterns = [
        r'\b\d+\s*(inch|inches|cm|centimeter|centimeters|mm|millimeter|millimeters|ft|feet|foot)\b',
        r'\b(small|medium|large|xlarge|xl|xxl|xxxl|xs|xxs|big|tiny)\b',
        r'\b\d+\s*x\s*\d+\s*(inch|inches|cm|centimeter|centimeters)\b',
    ]
    size_descriptors = ['small', 'medium', 'large', 'big', 'tiny', 'compact', 'spacious', 'roomy']
    materials = ['cotton', 'polyester', 'leather', 'wood', 'metal', 'plastic', 'glass', 'ceramic',
                'fabric', 'silk', 'wool', 'denim', 'canvas', 'nylon', 'rubber', 'steel', 'aluminum',
                'marble', 'bamboo', 'velvet', 'suede', 'linen']
    shapes = ['round', 'square', 'rectangular', 'circular', 'oval', 'triangular', 'curved',
             'straight', 'angular', 'spherical', 'cylindrical', 'diamond', 'heart', 'star',
             'narrow', 'wide', 'thin', 'thick', 'flat']
    visual_attrs = ['bright', 'dark', 'light', 'shiny', 'matte', 'glossy', 'dull', 'transparent',
                   'opaque', 'smooth', 'rough', 'textured', 'patterned', 'striped',
                   'polished', 'reflective', 'metallic', 'clear', 'solid',
                   'grainy', 'silky', 'velvety', 'woven', 'knitted', 'floral', 'geometric',
                   'spotted', 'dotted', 'lined', 'painted', 'colored', 'multicolored',
                   'sturdy', 'flimsy', 'sharp', 'soft', 'hard', 'substantial', 'nice', 'beautiful', 'lovely']

    for concept in visual_concepts:
        if concept in text_lower:
            visual_entities.add(concept)
    for color in colors:
        if color in text_lower:
            visual_entities.add(color)
    for pattern in size_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                match_str = ' '.join([m for m in match if m]).strip()
                if match_str:
                    visual_entities.add(match_str)
            else:
                visual_entities.add(match)
    for size_desc in size_descriptors:
        if size_desc in text_lower:
            visual_entities.add(size_desc)
    for material in materials:
        if material in text_lower:
            visual_entities.add(material)
    for shape in shapes:
        if shape in text_lower:
            visual_entities.add(shape)
    for attr in visual_attrs:
        if attr in text_lower:
            visual_entities.add(attr)

    additional_visual_words = [
        'visible', 'edge', 'border', 'frame', 'outline', 'logo', 'decoration', 'detail',
        'tip', 'bent', 'sharp', 'room', 'space'
    ]
    product_visual_nouns = [
        'hair', 'pin', 'pins', 'clip', 'clips', 'bobby', 'band', 'bands', 'ties',
        'wire', 'mat', 'scissors', 'blade', 'die', 'yarn', 'fabric', 'marker', 'markers',
        'tape', 'masking', 'glue', 'ornament', 'ceramic', 'blank', 'loom', 'cutter', 'wax',
        'bag', 'pouch', 'pockets', 'zipper', 'strap', 'case', 'box', 'kit', 'set',
        'organizer', 'storage', 'jar', 'bottle', 'tube', 'quilt', 'quilts', 'pressing'
    ]
    for word in additional_visual_words:
        if word in text_lower:
            visual_entities.add(word)
    for noun in product_visual_nouns:
        if noun in text_lower:
            visual_entities.add(noun)

    return visual_entities


def compute_visual_entity_coverage(predictions, references):
    assert len(predictions) == len(references)
    coverages, precisions = [], []
    for pred, ref in zip(predictions, references):
        ref_entities = extract_visual_entities(ref)
        pred_entities = extract_visual_entities(pred)
        if len(ref_entities) == 0:
            continue
        covered = len(ref_entities.intersection(pred_entities))
        coverage = covered / len(ref_entities)
        coverages.append(coverage)
        if len(pred_entities) > 0:
            precision = len(ref_entities.intersection(pred_entities)) / len(pred_entities)
            precisions.append(precision)
    if len(coverages) == 0:
        return {"VisualEntityCoverage": 0.0, "VisualEntityPrecision": 0.0, "VisualEntityF1": 0.0}
    avg_coverage = np.mean(coverages)
    avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
    f1 = 2 * avg_coverage * avg_precision / (avg_coverage + avg_precision) if (avg_coverage + avg_precision) > 0 else 0.0
    return {
        "VisualEntityCoverage": avg_coverage * 100,
        "VisualEntityPrecision": avg_precision * 100,
        "VisualEntityF1": f1 * 100
    }


def load_self_contained(jsonl_path):
    """从 inference 输出的 jsonl 里，直接拿 id + prediction(s) + ground_truth，不查外部文件"""
    id2reference = {}
    prediction_pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred_id = str(item["id"])
            gt = item.get("ground_truth", "")
            id2reference[pred_id] = gt
            if "outputs" in item:
                for output in item["outputs"]:
                    prediction_pairs.append((pred_id, output))
            else:
                prediction_pairs.append((pred_id, item["output"]))
    return id2reference, prediction_pairs


def evaluate_all(args):
    id2reference, prediction_pairs = load_self_contained(args.pred_path)

    # ---- CIDEr IDF：可选在多个模态文件之间共享同一份语料库 ----
    if args.cider_idf_pool_paths:
        pool_refs = []
        for p in args.cider_idf_pool_paths.split(","):
            p = p.strip()
            if not p:
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    gt = item.get("ground_truth", "")
                    if gt:
                        pool_refs.append(gt)
        cider_idf = build_cider_idf(pool_refs)
        cider_idf_source = f"pooled {len(args.cider_idf_pool_paths.split(','))} files ({len(pool_refs)} refs)"
        print(f"[INFO] CIDEr IDF from {cider_idf_source}, ref_len={cider_idf.ref_len:.4f}")
    else:
        own_refs = [r for r in id2reference.values() if r]
        cider_idf = build_cider_idf(own_refs)
        cider_idf_source = f"self ({len(own_refs)} refs)"
        print(f"[WARN] --cider_idf_pool_paths 未设置，CIDEr IDF 只用当前文件自己的 ground_truth 建的，"
              f"和其他模态跑出来的 CIDEr 不可比。ref_len={cider_idf.ref_len:.4f}")

    references, predictions, missing_ids = [], [], []
    for pred_id, pred_text in prediction_pairs:
        ref = id2reference.get(pred_id, "")
        if ref:
            references.append(ref.strip())
            predictions.append(pred_text.strip())
        else:
            missing_ids.append(pred_id)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} ids have empty ground_truth: {missing_ids[:5]}...")
    assert len(references) == len(predictions), "Mismatch between references and predictions!"
    assert len(predictions) > 0, "predictions is empty"

    id2outs = defaultdict(list)
    for pid, out in prediction_pairs:
        if id2reference.get(pid):
            id2outs[pid].append(out.strip())

    sample_ids  = list(id2outs.keys())
    sample_refs = [id2reference[pid].strip() for pid in sample_ids]
    sample_outs = [id2outs[pid] for pid in sample_ids]
    n_draws = min(len(o) for o in sample_outs) if sample_outs else 0
    print(f"[INFO] #samples={len(sample_refs)}, n_draws={n_draws}")

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    length_tokenizer = AutoTokenizer.from_pretrained("${LLAMA_BASE_MODEL}")

    print("sample pred:", str(predictions[0])[:200])
    print("sample ref :", str(references[0])[:200])

    rouge_result = rouge.compute(predictions=predictions, references=references, use_aggregator=False)
    rouge_result = {k: float(np.mean(v)) for k, v in rouge_result.items()}

    chrf_draws = []
    for j in range(n_draws):
        preds_j = [outs[j] for outs in sample_outs]
        chrf_draws.append(chrf.compute(predictions=preds_j, references=sample_refs)["score"])
    chrf_score = float(np.mean(chrf_draws)) if chrf_draws else 0.0
    chrf_std = float(np.std(chrf_draws)) if chrf_draws else 0.0
    print(f"[INFO] chrF per-draw: {[f'{c:.2f}' for c in chrf_draws]} -> mean {chrf_score:.2f} ± {chrf_std:.2f}")

    bleu_draws = []
    for j in range(n_draws):
        preds_j = [outs[j] for outs in sample_outs]
        bleu_draws.append(bleu.compute(predictions=preds_j, references=sample_refs)["bleu"])
    bleu_score = float(np.mean(bleu_draws)) if bleu_draws else 0.0
    bleu_std = float(np.std(bleu_draws)) if bleu_draws else 0.0
    print(f"[INFO] BLEU per-draw: {[f'{b:.4f}' for b in bleu_draws]} -> mean {bleu_score:.4f} ± {bleu_std:.4f}")

    cider_draws = []
    for j in range(n_draws):
        preds_j = [outs[j] for outs in sample_outs]
        cider_draws.append(compute_cider(preds_j, sample_refs, cider_idf))
    cider_score = float(np.mean(cider_draws)) if cider_draws else 0.0
    cider_std = float(np.std(cider_draws)) if cider_draws else 0.0
    print(f"[INFO] CIDEr per-draw: {[f'{c:.4f}' for c in cider_draws]} -> mean {cider_score:.4f} ± {cider_std:.4f}")

    bertscore_p, bertscore_r, bertscore_f1 = compute_bertscore(predictions, references)

    d1_result = compute_distinct_n(predictions, n=1)
    d2_result = compute_distinct_n(predictions, n=2)

    avg_output_token_len = compute_token_length_stats(predictions, length_tokenizer)
    avg_reference_token_len = compute_token_length_stats(references, length_tokenizer)

    print("\n[INFO] Computing visual entity coverage...")
    visual_entity_metrics = compute_visual_entity_coverage(predictions, references)
    print(f"[INFO] Visual Entity Coverage: {visual_entity_metrics['VisualEntityCoverage']:.2f}%")
    print(f"[INFO] Visual Entity Precision: {visual_entity_metrics['VisualEntityPrecision']:.2f}%")
    print(f"[INFO] Visual Entity F1: {visual_entity_metrics['VisualEntityF1']:.2f}%")

    results = {
        "BLEU": bleu_score, "BLEU_std": bleu_std,
        "ROUGE-1": rouge_result["rouge1"], "ROUGE-2": rouge_result["rouge2"], "ROUGE-L": rouge_result["rougeL"],
        "BERTScore-F1": bertscore_f1,
        "chrF++": chrf_score, "chrF++_std": chrf_std,
        "CIDEr": cider_score, "CIDEr_std": cider_std,
        "CIDEr_idf_source": cider_idf_source,
        "Distinct-1": d1_result, "Distinct-2": d2_result,
        "AvgOutputTokenLength": avg_output_token_len["mean"],
        "MinOutputTokenLength": avg_output_token_len["min"],
        "MaxOutputTokenLength": avg_output_token_len["max"],
        "AvgReferenceTokenLength": avg_reference_token_len["mean"],
        "MinReferenceTokenLength": avg_reference_token_len["min"],
        "MaxReferenceTokenLength": avg_reference_token_len["max"],
        "n_draws": n_draws,
        **visual_entity_metrics
    }

    print("\n======== Evaluation Results ========")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("=====================================\n")

    if args.save_csv:
        df = pd.DataFrame([{
            "run_name": args.run_name, "checkpoint": args.checkpoint, "test_type": args.test_type,
            **results
        }])
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        if not os.path.exists(args.save_csv):
            df.to_csv(args.save_csv, index=False)
        else:
            df.to_csv(args.save_csv, mode='a', header=False, index=False)
        print(f"Appended evaluation results to {args.save_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_type", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument(
        "--cider_idf_pool_paths", type=str, default=None,
        help="逗号分隔的多个 jsonl 路径（同一个 run 的 both/text_only/img_only 三个文件），"
             "共用同一份 CIDEr IDF 语料库，保证三个模态的 CIDEr 互相可比。不传则只用当前文件自己建。"
    )
    args = parser.parse_args()
    evaluate_all(args)