"""Nhung van ban su kien TRAKE vao cache KhoSims (chay 1 lan qua chay_gon_ram)."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._console import safe_console
safe_console()
from scripts.experiment_cap_thoi_gian import KhoSims
from scripts.make_submission import split_events

gt = json.loads((ROOT / "data" / "gt_trake.json").read_text(encoding="utf-8"))
kho = KhoSims(str(ROOT / "data"), False)
n = 0
for m in gt:
    de = (m.get("boi_canh", "") + "\n"
          + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
    for ev in split_events(de):
        kho.lay(ev, "")
        n += 1
print(f"da cache sims cho {n} van ban su kien / {len(gt)} muc")
