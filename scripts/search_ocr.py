"""Find the frame that SAYS something. Reads the OCR cache, no model needed.

Complements search_transcripts.py: that one searches what was spoken, this one
searches what is written on screen. Between them they cover the two channels an
image embedding cannot represent at all.

    python scripts/search_ocr.py "Nguyễn Trung Trực"
    python scripts/search_ocr.py "nhân bánh cuốn" --videos L26
    python scripts/search_ocr.py --list-videos

Run scripts/run_ocr.py first; only frames it has read are searchable.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.ocr import MIN_CONF  # noqa: E402


def norm(s):
    return unicodedata.normalize("NFC", str(s or "")).lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="text you expect to see on screen")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--videos", default=None, help="restrict to ids starting with this")
    ap.add_argument("-n", "--top", type=int, default=15)
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--list-videos", action="store_true", help="what has been read so far")
    args = ap.parse_args()

    data = Path(args.data)
    ocr_dir = data / "ocr"
    files = sorted(ocr_dir.glob("*.json"))
    if not files:
        print("Chua doc OCR frame nao. Chay truoc:")
        print("  python scripts/run_ocr.py --queries round_p1/queries")
        return 2
    if args.videos:
        pre = tuple(x.strip() for x in args.videos.split(","))
        files = [f for f in files if f.stem.startswith(pre)]

    total = 0
    store = {}
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        store[f.stem] = d
        total += len(d)

    if args.list_videos or not args.query:
        withtext = sum(
            1
            for d in store.values()
            for v in d.values()
            if " ".join(t for t, c in v if c >= args.min_conf).strip()
        )
        print(f"{len(store)} video, {total} frame da doc, {withtext} co chu")
        return 0

    query = " ".join(args.query)
    needles = [norm(w) for w in query.split() if len(w) > 1]
    whole = norm(query)

    fps, urls, fname = {}, {}, {}
    meta = data / "metadata.json"
    if meta.exists():
        for m in json.loads(meta.read_text(encoding="utf-8")):
            fps.setdefault(m["video_id"], float(m["fps"]))
    z = data / "media-info-aic25-b1.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".json"):
                    urls[Path(n).stem] = json.loads(zf.read(n)).get("watch_url", "")

    hits = []
    for vid, d in store.items():
        for k, v in d.items():
            text = " ".join(t for t, c in v if c >= args.min_conf)
            low = norm(text)
            if not low:
                continue
            # the whole phrase is worth far more than scattered words, because
            # OCR mangles word boundaries more often than it mangles characters
            score = (10 if whole in low else 0) + sum(1 for w in needles if w in low)
            if score:
                hits.append((score, vid, int(k), text))
    hits.sort(key=lambda h: (-h[0], h[1], h[2]))

    print(f"\n{total} frame da doc · tim: {query!r} · {len(hits)} frame khop\n")
    for score, vid, frame, text in hits[: args.top]:
        sec = frame / fps.get(vid, 25.0)
        u = urls.get(vid, "").replace("https://youtube.com/watch?v=", "https://youtu.be/")
        link = f"{u}?t={max(0, int(sec) - 2)}" if u else ""
        exact = " (khop ca cum)" if whole in norm(text) else ""
        print(f"{vid}  frame {frame:<7d} {int(sec)//60:d}:{int(sec)%60:02d}  {link}{exact}")
        print(f"   {text[:170]}")
        print()
    if not hits:
        print("Khong frame nao da doc co chu do. Co the frame dung chua duoc OCR;")
        print("chay run_ocr.py voi --top lon hon, hoac tim bang search_transcripts.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
