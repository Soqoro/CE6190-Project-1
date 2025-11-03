from __future__ import annotations
import json, random, os
from pathlib import Path

def save_ids(ids: list[str], out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(sorted(ids), f)

def load_ids(path: str) -> list[str]:
    with open(path) as f: return json.load(f)

def make_percentage_split(all_ids: list[str], percent: int, seed: int, out_path: str):
    rnd = random.Random(seed)
    k = len(all_ids) if percent >= 100 else max(1, int(len(all_ids) * percent / 100.0))
    sel = sorted(rnd.sample(sorted(all_ids), k))
    save_ids(sel, out_path)
