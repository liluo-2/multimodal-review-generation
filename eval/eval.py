import evaluate
import json
import pandas as pd
import numpy as np
import os
import argparse
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bertscore_score
from peft import PeftModel, PeftConfig
import re
from pycocoevalcap.cider.cider import Cider


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


def compute_cider(predictions, references):
    """
    predictions: List[str]
    references: List[str]
    """
    gts = {i: [references[i].strip()] for i in range(len(references))}
    res = {i: [predictions[i].strip()] for i in range(len(predictions))}
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    return float(score)


def compute_perplexity(predictions, model, tokenizer, device="cuda"):
    model = model.to(device)
    model.eval()
    losses = []
    for text in predictions:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                losses.append(loss.item())
    return np.exp(np.mean(losses))


def compute_token_length_stats(texts, tokenizer):
    lengths = [len(tokenizer.tokenize(text.strip())) for text in texts]
    
    if not lengths:
        return {
            "mean": 0.0,
            "min": 0,
            "max": 0
        }
    
    return {
        "mean": float(np.mean(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths))
    }


def load_references_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['id'].astype(str), df['description']))


def load_review_references(jsonl_path: str, split_path: str):
    if split_path.endswith(".npz"):
        z = np.load(split_path, allow_pickle=False)
        split = {k: z[k] for k in z.files}
    else:
        split = np.load(split_path, allow_pickle=True).item()

    test_ids = set(int(i) for i in split[args.test_type])
    id2rev = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in test_ids:
                obj = json.loads(line)
                id2rev[str(idx)] = obj["reference"]["text"]
    return id2rev


def load_predictions_from_jsonl(jsonl_path):
    prediction_pairs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            pred_id = str(item["id"])

            if "outputs" in item:
                for output in item["outputs"]:
                    prediction_pairs.append((pred_id, output))

            else:
                prediction_pairs.append(
                    (pred_id, item["output"])
                )

    return prediction_pairs


def load_predictions_qwen(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = []
    for item in data:
        output = item["generated_response"]
        if output:
            predictions.append((str(item["id"]), output))
    return predictions


def load_predictions_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [(str(json.loads(line)['question_id']), json.loads(line)['text']) for line in f]

def compute_bertscore(predictions, references):
    """
    Official BERTScore (Zhang et al., ICLR 2020)

    Returns:
        precision_mean,
        recall_mean,
        f1_mean
    """

    P, R, F1 = bertscore_score(
        predictions,
        references,
        model_type="roberta-large",
        lang="en",
        device="cpu",
        rescale_with_baseline=True,
        verbose=True
    )

    return (
        P.mean().item(),
        R.mean().item(),
        F1.mean().item()
    )

def extract_visual_entities(text):
    """
    Extract visual-related entities from text.
    This includes colors, sizes, shapes, materials, and visual attributes.
    """
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
                   'grainy', 'silky', 'velvety',
                   'woven', 'knitted', 'floral', 'geometric',
                   'spotted', 'dotted', 'lined',
                   'painted', 'colored', 'multicolored',
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
        'organizer', 'storage', 'jar', 'bottle', 'tube',
        'quilt', 'quilts', 'pressing'
    ]
    
    for word in additional_visual_words:
        if word in text_lower:
            visual_entities.add(word)

    for noun in product_visual_nouns:
        if noun in text_lower:
            visual_entities.add(noun)

    return visual_entities


def compute_visual_entity_coverage(predictions, references):
    """
    Compute the coverage of visual entities mentioned in references that are also mentioned in predictions.
    Returns:
        - coverage: average percentage of visual entities from references that appear in predictions
        - precision: average percentage of visual entities in predictions that appear in references
        - f1: harmonic mean of coverage and precision
    """
    assert len(predictions) == len(references)
    
    coverages = []
    precisions = []
    
    for pred, ref in zip(predictions, references):
        ref_entities = extract_visual_entities(ref)
        pred_entities = extract_visual_entities(pred)
        
        if len(ref_entities) == 0:
            continue
        
        covered = len(ref_entities.intersection(pred_entities))
        coverage = covered / len(ref_entities) if len(ref_entities) > 0 else 0.0
        coverages.append(coverage)
        
        if len(pred_entities) > 0:
            precision = len(ref_entities.intersection(pred_entities)) / len(pred_entities)
            precisions.append(precision)
    
    if len(coverages) == 0:
        return {
            "VisualEntityCoverage": 0.0,
            "VisualEntityPrecision": 0.0,
            "VisualEntityF1": 0.0
        }
    
    avg_coverage = np.mean(coverages)
    avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
    f1 = 2 * avg_coverage * avg_precision / (avg_coverage + avg_precision) if (avg_coverage + avg_precision) > 0 else 0.0
    
    return {
        "VisualEntityCoverage": avg_coverage * 100,
        "VisualEntityPrecision": avg_precision * 100,
        "VisualEntityF1": f1 * 100
    }


def evaluate_all(reference_path, prediction_path, split_path, save_csv_path=None, ppl_model_path=None, ppl_tokenizer_path=None):
    save_csv_path = save_csv_path or args.save_csv if 'args' in globals() else None

    if split_path:
        id2reference = load_review_references(reference_path, split_path)
    else:
        id2reference = load_references_from_csv(reference_path)

    if args.model_type == "qwen-vl":
        print("Using Qwen-VL model for evaluation")
        prediction_pairs = load_predictions_qwen(prediction_path)
    elif args.model_type == "llama-3":
        prediction_pairs = load_predictions_from_jsonl(prediction_path)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    print("ref_len:", len(id2reference))
    print("pred_len (flatten):", len(prediction_pairs))

    # ---- Flatten view: used by ROUGE / BERTScore / Visual / chrF / length stats ----
    # Each (sample, output) pair becomes one row -> these metrics naturally become
    # per-(sample,output) averages.
    references, predictions, missing_ids = [], [], []
    for pred_id, pred_text in prediction_pairs:
        if pred_id in id2reference:
            references.append(id2reference[pred_id].strip())
            predictions.append(pred_text.strip())
        else:
            missing_ids.append(pred_id)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} ids not found in references: {missing_ids[:5]}...")
    assert len(references) == len(predictions), "Mismatch between references and predictions!"
    assert len(predictions) > 0, "predictions is empty"

    # ---- Per-sample view: used by BLEU / CIDEr (draw averaging) ----
    id2outs = defaultdict(list)
    for pid, out in prediction_pairs:
        if pid in id2reference:
            id2outs[pid].append(out.strip())

    sample_ids  = list(id2outs.keys())
    sample_refs = [id2reference[pid].strip() for pid in sample_ids]
    sample_outs = [id2outs[pid] for pid in sample_ids]  # list[list[str]]
    n_draws = min(len(o) for o in sample_outs) if sample_outs else 0
    print(f"[INFO] #samples={len(sample_refs)}, n_draws={n_draws}")

    # ---- Metric objects ----
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    length_tokenizer = AutoTokenizer.from_pretrained(
        "${LLAMA_BASE_MODEL}"
    )

    print("sample pred:", str(predictions[0])[:200])
    print("sample ref :", str(references[0])[:200])

    # ---- ROUGE / chrF (flatten = per-(sample,output) average for ROUGE; chrF is corpus over flatten) ----
    # NOTE: chrF here is corpus chrF on the flattened set. If you want chrF in the same
    # "draw averaging" protocol as BLEU/CIDEr, set USE_DRAW_FOR_CHRF=True below.
    rouge_result = rouge.compute(predictions=predictions, references=references)

    USE_DRAW_FOR_CHRF = True
    if USE_DRAW_FOR_CHRF and n_draws > 0:
        chrf_draws = []
        for j in range(n_draws):
            preds_j = [outs[j] for outs in sample_outs]
            chrf_draws.append(chrf.compute(predictions=preds_j, references=sample_refs)["score"])
        chrf_score = float(np.mean(chrf_draws))
        chrf_std = float(np.std(chrf_draws))
        print(f"[INFO] chrF per-draw: {[f'{c:.2f}' for c in chrf_draws]} -> mean {chrf_score:.2f} ± {chrf_std:.2f}")
    else:
        chrf_result = chrf.compute(predictions=predictions, references=references)
        chrf_score = chrf_result["score"]
        chrf_std = 0.0

    # ---- BLEU via draw averaging (Plan B) ----
    bleu_draws = []
    for j in range(n_draws):
        preds_j = [outs[j] for outs in sample_outs]
        bleu_draws.append(
            bleu.compute(predictions=preds_j, references=sample_refs)["bleu"]
        )
    bleu_score = float(np.mean(bleu_draws)) if bleu_draws else 0.0
    bleu_std = float(np.std(bleu_draws)) if bleu_draws else 0.0
    print(f"[INFO] BLEU per-draw: {[f'{b:.4f}' for b in bleu_draws]} -> mean {bleu_score:.4f} ± {bleu_std:.4f}")

    # ---- CIDEr via draw averaging ----
    cider_draws = []
    for j in range(n_draws):
        preds_j = [outs[j] for outs in sample_outs]
        cider_draws.append(compute_cider(preds_j, sample_refs))
    cider_score = float(np.mean(cider_draws)) if cider_draws else 0.0
    cider_std = float(np.std(cider_draws)) if cider_draws else 0.0
    print(f"[INFO] CIDEr per-draw: {[f'{c:.4f}' for c in cider_draws]} -> mean {cider_score:.4f} ± {cider_std:.4f}")

    # ---- BERTScore (flatten = per-(sample,output) average) ----
    bertscore_p, bertscore_r, bertscore_f1 = compute_bertscore(predictions, references)

    # ---- Distinct on flatten set (caveat: not directly comparable to n=1) ----
    d1_result = compute_distinct_n(predictions, n=1)
    d2_result = compute_distinct_n(predictions, n=2)

    # ---- Length stats on flatten set ----
    avg_output_token_len = compute_token_length_stats(predictions, length_tokenizer)
    avg_reference_token_len = compute_token_length_stats(references, length_tokenizer)

    print("\n[INFO] Computing visual entity coverage...")
    visual_entity_metrics = compute_visual_entity_coverage(predictions, references)
    print(f"[INFO] Visual Entity Coverage: {visual_entity_metrics['VisualEntityCoverage']:.2f}%")
    print(f"[INFO] Visual Entity Precision: {visual_entity_metrics['VisualEntityPrecision']:.2f}%")
    print(f"[INFO] Visual Entity F1: {visual_entity_metrics['VisualEntityF1']:.2f}%")

    if len(predictions) > 0:
        sample_ref_entities = extract_visual_entities(references[0])
        sample_pred_entities = extract_visual_entities(predictions[0])
        print(f"[DEBUG] Sample reference visual entities: {sample_ref_entities}")
        print(f"[DEBUG] Sample prediction visual entities: {sample_pred_entities}")
        print(f"[DEBUG] Sample matched entities: {sample_ref_entities.intersection(sample_pred_entities)}")

    results = {
        "BLEU": bleu_score,
        "BLEU_std": bleu_std,
        "ROUGE-1": rouge_result["rouge1"],
        "ROUGE-2": rouge_result["rouge2"],
        "ROUGE-L": rouge_result["rougeL"],
        "BERTScore-F1": bertscore_f1,
        "chrF++": chrf_score,
        "chrF++_std": chrf_std,
        "CIDEr": cider_score,
        "CIDEr_std": cider_std,
        "Distinct-1": d1_result,
        "Distinct-2": d2_result,
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
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=====================================\n")

    if save_csv_path:
        df = pd.DataFrame([{
            "dataset": args.dataset,
            "split_type": args.split_type,
            "train_type": args.train_type,
            "test_type": args.test_type,
            "prompt_type": args.prompt_type,
            **results
        }])
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        if not os.path.exists(save_csv_path):
            df.to_csv(save_csv_path, index=False)
        else:
            df.to_csv(save_csv_path, mode='a', header=False, index=False)
        print(f"Appended evaluation results to {save_csv_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--test_type", type=str, default="test")
    parser.add_argument("--split_type", type=str, default="length")
    parser.add_argument("--train_type", type=str, default="train")
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="qwen-vl")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--reference_path", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--pred_path", type=str, default=None)
    args = parser.parse_args()

    reference_path = args.reference_path
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    split_path = args.split_path
    prediction_path = args.pred_path

    evaluate_all(
        reference_path=reference_path,
        prediction_path=prediction_path,
        split_path=split_path,
        save_csv_path=args.save_csv,
        ppl_model_path=model_path,
        ppl_tokenizer_path=tokenizer_path
    )