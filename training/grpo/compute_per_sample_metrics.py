"""
compute_per_sample_metrics.py
=============================
Per-sample metrics for single vs mix teacher inference on train set.
Metrics: BLEU, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore (CLS cosine), CIDEr

Output columns:
  dataset, modality, id, ref, single_pred, mix_pred,
  single_X, mix_X, delta_X  (for each metric, delta = mix - single),
  single_norm_total, mix_norm_total, best_norm, better_teacher

Usage:
  python compute_per_sample_metrics.py \
    --datasets Arts Sports Toys \
    --base_dir ${DATA_ROOT}/output/revllm-img2txt \
    --ref_base ${MULTIMODAL_DATA_ROOT}/final-dataset \
    --bert_model_path ${MODEL_ROOT}/bert-base-uncased \
    --output_dir ${OUTPUT_ROOT}/results/per_sample_train
"""

import json, os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import evaluate
from transformers import AutoTokenizer, AutoModel
from pycocoevalcap.cider.cider import Cider

MODALITIES    = ["both", "img-only", "text-only"]
SCORE_METRICS = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "CIDEr"]  # 6


# ── Path helpers ───────────────────────────────────────────────────────────────

def get_single_path(base_dir, dataset, modality):
    name_map = {
        "both":      f"{dataset}_all_hist_both_split_img2txt.jsonl",
        "img-only":  f"{dataset}_all_hist_img-only_split_img2txt.jsonl",
        "text-only": f"{dataset}_all_hist_text-only_split_img2txt.jsonl",
    }
    return os.path.join(base_dir, dataset, name_map[modality])

def get_mix_path(base_dir, dataset, modality):
    name_map = {
        "both":      f"{dataset}_all_hist-revllm_mixed_split_both-img2txt.jsonl",
        "img-only":  f"{dataset}_all_hist-revllm_mixed_split_img_only-img2txt.jsonl",
        "text-only": f"{dataset}_all_hist-revllm_mixed_split_text_only-img2txt.jsonl",
    }
    return os.path.join(base_dir, dataset, name_map[modality])

def get_ref_path(ref_base, dataset):
    return os.path.join(ref_base, dataset, "final-desc-img.jsonl")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_predictions(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            sid = str(obj.get("id", i))
            if "outputs" in obj:
                data[sid] = obj["outputs"][0]
            else:
                data[sid] = obj.get("output", "").strip()
    return data

def load_references(path):
    """Load all IDs directly from final-desc-img.jsonl (no split filtering)."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            sid = str(obj.get("id", i))
            if "reference" in obj and isinstance(obj["reference"], dict):
                data[sid] = obj["reference"].get("text", "").strip()
            elif "description" in obj:
                data[sid] = obj["description"].strip()
    return data


# ── Metrics ────────────────────────────────────────────────────────────────────

_rouge  = evaluate.load("rouge")
_bleu   = evaluate.load("bleu")

def compute_text_metrics(pred, ref):
    r = _rouge.compute(predictions=[pred], references=[ref])
    try:
        b = _bleu.compute(predictions=[pred], references=[[ref]])
        bleu = b["bleu"]
    except Exception:
        bleu = 0.0
    return {
        "ROUGE-1": r["rouge1"],
        "ROUGE-2": r["rouge2"],
        "ROUGE-L": r["rougeL"],
        "BLEU":    bleu,
    }

BERT_TOK = None
BERT_MDL = None

def load_bert(path):
    global BERT_TOK, BERT_MDL
    if BERT_TOK is None:
        print(f"[INFO] Loading BERT: {path}")
        BERT_TOK = AutoTokenizer.from_pretrained(path)
        BERT_MDL = AutoModel.from_pretrained(path)
        BERT_MDL.eval()

def bertscore_batch(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        ep = BERT_TOK(p, return_tensors="pt", truncation=True, max_length=512, padding=True)
        er = BERT_TOK(r, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            vp = BERT_MDL(**ep).last_hidden_state[:, 0, :]
            vr = BERT_MDL(**er).last_hidden_state[:, 0, :]
        scores.append(F.cosine_similarity(vp, vr).item())
    return scores

def compute_cider_group(preds, refs):
    """Per-sample CIDEr using the whole group as IDF corpus."""
    gts = {i: [refs[i].strip()] for i in range(len(refs))}
    res = {i: [preds[i].strip()] for i in range(len(preds))}
    scorer = Cider()
    _, scores = scorer.compute_score(gts, res)
    return list(scores)


# ── Normalisation ──────────────────────────────────────────────────────────────

def decide_better_teacher(df_group):
    """
    For each metric, normalise using combined (single + mix) min/max within group.
    Sum normalised scores → single_norm_total, mix_norm_total.
    best_norm = max of the two.
    """
    df = df_group.copy()
    s_total = np.zeros(len(df))
    m_total = np.zeros(len(df))

    for metric in SCORE_METRICS:
        s_vals = df[f"single_{metric}"].values.astype(float)
        m_vals = df[f"mix_{metric}"].values.astype(float)
        combined = np.concatenate([s_vals, m_vals])
        lo, hi = combined.min(), combined.max()
        if hi - lo < 1e-9:
            s_norm = np.zeros_like(s_vals)
            m_norm = np.zeros_like(m_vals)
        else:
            s_norm = (s_vals - lo) / (hi - lo)
            m_norm = (m_vals - lo) / (hi - lo)
        s_total += s_norm
        m_total += m_norm

    df["single_norm_total"] = s_total
    df["mix_norm_total"]    = m_total
    df["best_norm"]         = np.maximum(s_total, m_total)
    df["better_teacher"]    = np.where(
        s_total > m_total, "single",
        np.where(m_total > s_total, "mix", "tie")
    )
    return df


# ── Per-modality processing ────────────────────────────────────────────────────

def process_one(dataset, modality, base_dir, ref_base):
    single_path = get_single_path(base_dir, dataset, modality)
    mix_path    = get_mix_path(base_dir, dataset, modality)
    ref_path    = get_ref_path(ref_base, dataset)

    for p in [single_path, mix_path, ref_path]:
        if not os.path.exists(p):
            print(f"[WARN] Not found, skipping: {p}")
            return None

    print(f"\n[INFO] {dataset}/{modality}")
    s_preds = load_predictions(single_path)
    m_preds = load_predictions(mix_path)
    refs    = load_references(ref_path)

    ids = sorted(
        set(s_preds) & set(m_preds) & set(refs),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"  Common IDs: {len(ids)}")
    if not ids:
        return None

    rows = []
    for idx, sid in enumerate(ids):
        if idx % 200 == 0:
            print(f"  [{idx}/{len(ids)}] ROUGE+BLEU ...", flush=True)
        ref = refs[sid]
        sp  = s_preds[sid]
        mp  = m_preds[sid]
        sm  = compute_text_metrics(sp, ref)
        mm  = compute_text_metrics(mp, ref)
        rows.append({
            "dataset": dataset, "modality": modality,
            "id": sid, "ref": ref,
            "single_pred": sp, "mix_pred": mp,
            **{f"single_{k}": sm[k] for k in sm},
            **{f"mix_{k}":    mm[k] for k in mm},
        })

    df = pd.DataFrame(rows)

    # BERTScore
    print(f"  BERTScore ({len(df)} x 2) ...", flush=True)
    sb = bertscore_batch(df["single_pred"].tolist(), df["ref"].tolist())
    mb = bertscore_batch(df["mix_pred"].tolist(),    df["ref"].tolist())
    df["single_BERTScore"] = sb
    df["mix_BERTScore"]    = mb

    # CIDEr (per group)
    print(f"  CIDEr ...", flush=True)
    sc = compute_cider_group(df["single_pred"].tolist(), df["ref"].tolist())
    mc = compute_cider_group(df["mix_pred"].tolist(),    df["ref"].tolist())
    df["single_CIDEr"] = sc
    df["mix_CIDEr"]    = mc

    # delta = mix - single for all metrics
    for m in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore", "CIDEr"]:
        df[f"delta_{m}"] = df[f"mix_{m}"] - df[f"single_{m}"]

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    load_bert(args.bert_model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    all_dfs = []
    for dataset in args.datasets:
        for modality in MODALITIES:
            df = process_one(dataset, modality, args.base_dir, args.ref_base)
            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No data generated.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Normalise and decide better_teacher per group
    print("\n[INFO] Normalising and deciding better_teacher ...")
    processed = []
    for (dataset, modality), grp in full_df.groupby(["dataset", "modality"]):
        processed.append(decide_better_teacher(grp))
    full_df = pd.concat(processed, ignore_index=True)

    # Reorder columns to required format
    col_order = (
        ["dataset", "modality", "id", "ref", "single_pred", "mix_pred"]
        + [c for m in ["ROUGE-1","ROUGE-2","ROUGE-L","BLEU","BERTScore","CIDEr"]
           for c in [f"single_{m}", f"mix_{m}", f"delta_{m}"]]
        + ["single_norm_total", "mix_norm_total", "best_norm", "better_teacher"]
    )
    full_df = full_df[[c for c in col_order if c in full_df.columns]]

    # Threshold
    p70 = full_df["best_norm"].quantile(0.70)
    print(f"\n[INFO] 70th pct of best_norm → reward_threshold = {p70:.4f}")
    print(f"[INFO] better_teacher distribution:\n{full_df['better_teacher'].value_counts()}")

    # Save per-sample detail
    detail_path = os.path.join(args.output_dir, "per_sample_detail_train.csv")
    full_df.to_csv(detail_path, index=False)
    print(f"\n[SAVED] {detail_path}  ({len(full_df)} rows)")

    # Save summary
    metric_cols = [c for c in full_df.columns if c.startswith(("single_","mix_","delta_"))]
    summary = (
        full_df.groupby(["dataset","modality"])[metric_cols + ["better_teacher"]]
        .agg({**{c: "mean" for c in metric_cols},
              "better_teacher": lambda x: (x=="mix").mean()})
        .rename(columns={"better_teacher": "mix_better_ratio"})
        .reset_index()
    )
    summary.to_csv(os.path.join(args.output_dir, "summary_train.csv"), index=False)
    print(f"[SAVED] summary_train.csv")

    # Save distribution
    dist = (
        full_df.groupby(["dataset","modality","better_teacher"])
        .size().reset_index(name="count")
    )
    dist.to_csv(
        os.path.join(args.output_dir, "teacher_selection_distribution_train.csv"),
        index=False,
    )
    print(f"[SAVED] teacher_selection_distribution_train.csv")
    print(f"\n✅  Update --reward_threshold {p70:.4f} in train_grpo.SBATCH")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",        nargs="+", default=["Arts","Sports","Toys"])
    parser.add_argument("--base_dir",        default="${DATA_ROOT}/output/revllm-img2txt")
    parser.add_argument("--ref_base",        default="${MULTIMODAL_DATA_ROOT}/final-dataset")
    parser.add_argument("--bert_model_path", default="${MODEL_ROOT}/bert-base-uncased")
    parser.add_argument("--output_dir",      default="${OUTPUT_ROOT}/results/per_sample_train")
    args = parser.parse_args()
    main(args)