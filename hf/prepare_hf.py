#!/usr/bin/env python3
r"""
Assemble HuggingFace upload folders for the MMRev project from local files.

Produces:
  <out>/mmrev-data/            -> HuggingFace *dataset* repo
  <out>/mmrev-llama3.1-8b-grpo/ -> HuggingFace *model* repo (LoRA adapter)

Usage (example):
  python prepare_hf.py \
      --final_dir      ./final          \   # folder containing final_Arts.jsonl, final_Sports.jsonl, final_Toys.jsonl
      --split_dir      ./split          \   # folder containing Arts_both_split.npz, ...
      --img2txt_dir    ./img2txt        \   # folder containing *img2txt*.jsonl files
      --teacher_dir    ./per_sample     \   # folder containing per_sample_detail_*_train*.csv
      --checkpoint_dir ./checkpoint-3147 \  # GRPO checkpoint folder
      --hf_user        YOUR_HF_USERNAME \
      --out            ./hf_upload
"""
import argparse, json, re, shutil, sys
from pathlib import Path

import numpy as np

DATASETS = ["Arts", "Sports", "Toys"]
SCENARIOS = {"both": "both", "text-only": "text_only", "img-only": "img_only"}
SPLITS = {"train_random": "train", "test_random": "test"}
KIT_DIR = Path(__file__).resolve().parent

ADAPTER_KEEP = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "trainer_state.json"]
ADAPTER_DROP = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]


def find_one(folder: Path, patterns):
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def clean_record(rec, idx, dataset, scenario, split):
    """Flatten one final.jsonl line into a stable, viewer-friendly schema."""
    ref = rec["reference"]
    imgs = ref.get("product_image") or []
    image_url = ""
    if imgs and isinstance(imgs[0], dict):
        image_url = imgs[0].get("large") or imgs[0].get("hi_res") or imgs[0].get("thumb") or ""
    history = [{
        "product_title": h.get("product_title") or "",
        "parent_asin":   h.get("parent_asin") or "",
        "rating":        float(h.get("rating") or 0.0),
        "timestamp":     int(h.get("timestamp") or 0),
        "review":        h.get("text") or "",
    } for h in rec["history"]]
    return {
        "id": str(idx),
        "dataset": dataset,
        "scenario": scenario,
        "split": split,
        "user_id": rec.get("user_id") or "",
        "history": history,
        "target_product_title": ref.get("product_title") or "",
        "target_product_desc":  ref.get("product_desc") or "",
        "target_parent_asin":   ref.get("parent_asin") or "",
        "target_image_url":     image_url,
        "target_rating":        float(ref.get("rating") or 0.0),
        "target_timestamp":     int(ref.get("timestamp") or 0),
        "reference_review":     ref.get("text") or "",
    }


def build_dataset(a, out: Path):
    root = out / "mmrev-data"
    if root.exists():
        shutil.rmtree(root)
    (root / "raw").mkdir(parents=True)
    (root / "splits").mkdir()
    (root / "image_descriptions").mkdir()
    (root / "teacher_scores").mkdir()

    final_dir, split_dir = Path(a.final_dir), Path(a.split_dir)
    stats = {}
    for ds in DATASETS:
        src = find_one(final_dir, [f"final_{ds}.jsonl", f"{ds}/final.jsonl", f"*{ds}*.jsonl"])
        if src is None:
            print(f"[WARN] final jsonl for {ds} not found in {final_dir}; skipping {ds}")
            continue
        (root / "raw" / ds).mkdir()
        shutil.copy2(src, root / "raw" / ds / "final.jsonl")
        lines = src.read_text(encoding="utf-8").splitlines()
        mix = {k: [] for k in SPLITS}
        for scen_file, scen in SCENARIOS.items():
            npz = split_dir / f"{ds}_{scen_file}_split.npz"
            if not npz.exists():
                print(f"[WARN] missing split {npz}"); continue
            shutil.copy2(npz, root / "splits" / npz.name)
            z = np.load(npz)
            for k in SPLITS:
                mix[k].extend(z[k].tolist())
            for key, split in SPLITS.items():
                ids = [int(x) for x in z[key]]
                dst = root / "data" / ds / scen / f"{split}.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "w", encoding="utf-8") as f:
                    for i in ids:
                        f.write(json.dumps(clean_record(json.loads(lines[i]), i, ds, scen, split),
                                           ensure_ascii=False) + "\n")
                stats[(ds, scen, split)] = len(ids)
        if all(mix.values()):
            np.savez(root / "splits" / f"{ds}_mix_split.npz", **{k: np.array(v) for k, v in mix.items()})

    # image descriptions: keep original file names, grouped by dataset
    if a.img2txt_dir:
        for p in sorted(Path(a.img2txt_dir).rglob("*.jsonl")):
            ds = next((d for d in DATASETS if d in p.name or d in p.parent.name), None)
            if ds is None:
                print(f"[WARN] cannot tell dataset of {p.name}; skipping"); continue
            name = re.sub(rf"_{ds}(?=\.jsonl$)", "", p.name)   # drop _Arts suffix
            (root / "image_descriptions" / ds).mkdir(exist_ok=True)
            shutil.copy2(p, root / "image_descriptions" / ds / name)

    # teacher scores (per-sample metric tables used for GRPO teacher look-up / reward normalization)
    if a.teacher_dir:
        for p in sorted(Path(a.teacher_dir).glob("*per_sample_detail_*.csv")):
            ds = next((d for d in DATASETS if d in p.name), "unknown")
            shutil.copy2(p, root / "teacher_scores" / f"per_sample_detail_{ds}_train.csv")

    card = (KIT_DIR / "dataset_card.md").read_text(encoding="utf-8").replace("{HF_USER}", a.hf_user)
    (root / "README.md").write_text(card, encoding="utf-8")
    return root, stats


def build_model(a, out: Path):
    root = out / "mmrev-llama3.1-8b-grpo"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    ck = Path(a.checkpoint_dir)
    for name in ADAPTER_KEEP:
        if (ck / name).exists():
            shutil.copy2(ck / name, root / name)
    cfg_path = root / "adapter_config.json"
    if not cfg_path.exists():
        sys.exit(f"[ERROR] adapter_config.json not found in {ck}")
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = "meta-llama/Llama-3.1-8B"   # replace cluster path
    cfg_path.write_text(json.dumps(cfg, indent=2))
    card = (KIT_DIR / "model_card.md").read_text(encoding="utf-8").replace("{HF_USER}", a.hf_user)
    (root / "README.md").write_text(card, encoding="utf-8")
    dropped = [n for n in ADAPTER_DROP if (ck / n).exists()]
    return root, dropped


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--img2txt_dir", default=None)
    ap.add_argument("--teacher_dir", default=None)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--hf_user", required=True)
    ap.add_argument("--out", default="./hf_upload")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    droot, stats = build_dataset(a, out)
    mroot, dropped = build_model(a, out)

    print("\n=== dataset split sizes (train/test) ===")
    for ds in DATASETS:
        row = [f"{s}: {stats.get((ds,s,'train'),0)}/{stats.get((ds,s,'test'),0)}" for s in SCENARIOS.values()]
        print(f"{ds:7s} " + "   ".join(row))
    for root in (droot, mroot):
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        print(f"\n{root}  ({human(size)})")
        for p in sorted(root.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(root)}  {human(p.stat().st_size)}")
    if dropped:
        print(f"\nNot copied (training-state only, not needed for inference): {', '.join(dropped)}")
    print("\nDone. Next: follow HF_UPLOAD_GUIDE.md to upload these two folders.")


if __name__ == "__main__":
    main()
