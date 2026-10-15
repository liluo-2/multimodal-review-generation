"""Build the mixed-scenario split used for mix-SFT and GRPO by taking the union
of the three per-scenario splits.

    python data/make_mix_split.py --split_dir data/splits --dataset Arts
    -> data/splits/Arts_mix_split.npz  (train_random / test_random)

Scenario of each sample is recovered at load time from the image-description
files: ids listed in the img-only description file are `img_only`, other ids
with an image description are `both`, and the rest are `text_only`.
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--split_dir", required=True)
ap.add_argument("--dataset", required=True)
a = ap.parse_args()
d = Path(a.split_dir)
out = {}
for key in ["train_random", "test_random"]:
    ids = []
    for scen in ["both", "text-only", "img-only"]:
        ids.extend(np.load(d / f"{a.dataset}_{scen}_split.npz")[key].tolist())
    assert len(ids) == len(set(ids)), "scenario splits overlap"
    out[key] = np.array(ids)
np.savez(d / f"{a.dataset}_mix_split.npz", **out)
print({k: len(v) for k, v in out.items()}, "->", d / f"{a.dataset}_mix_split.npz")
