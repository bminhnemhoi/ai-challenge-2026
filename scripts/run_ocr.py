"""Read the burned-in text on a round's candidate frames.

Run this once after make_submission and before opening review.html. It is a
background pass of a few minutes; everything is cached, so a second round over
the same videos is instant.

    python scripts/run_ocr.py --queries round_p1/queries --top 24

Then the review page shows what each frame actually says, and
scripts/search_ocr.py finds the frame that says a given thing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    detect_task,
    ranked_hits,
    read_en_override,
    read_query_text,
    split_events,
    split_qa,
)
from src.core.ocr import ColourIndex, OCRIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--top", type=int, default=24, help="candidates per query to read")
    ap.add_argument("--top-trake", type=int, default=6)
    ap.add_argument("--langs", default="vi,en")
    ap.add_argument("--no-colours", action="store_true",
                    help="skip the colour pass; it is free once the image is downloaded")
    ap.add_argument("--colours-only", action="store_true",
                    help="measure colour without OCR — no model to load, ~10 frames/s, "
                    "so it catches up a cache whose OCR pass predates the colour code")
    args = ap.parse_args()

    qdir = Path(args.queries)
    qfiles = sorted(
        p for p in qdir.glob("*.txt") if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    )

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}

    wanted: dict = {}
    trake_eng = None
    for qf in qfiles:
        text = read_query_text(qf) or ""
        task = detect_task(qf.name)
        en = read_en_override(qf)
        if task == "trake":
            from src.task3_trake import TRAKEEngine

            if trake_eng is None:
                trake_eng = TRAKEEngine(engine=eng).load_index()
            import re as _re

            first = bool(_re.search(r"đầu tiên|lần đầu|first", text, _re.IGNORECASE))
            chains = trake_eng.align_sequence(
                split_events(en or text), first_occurrence=first, top_k=args.top_trake
            )
            pairs = [(c["video_id"], f) for c in chains for f in c["sequence_frames"]]
        else:
            probe = split_qa(text)[0] if task == "qa" else text
            pairs = [(h.video_id, h.frame_idx) for h in ranked_hits(eng, probe, en)[: args.top]]
        for v, f in pairs:
            m = meta.get((v, int(f)))
            if m:
                wanted[(v, int(f))] = m["frame_filename"]
        print(f"  {qf.stem:24s} {len(pairs)} frame", flush=True)

    ocr = OCRIndex(args.data, langs=[x.strip() for x in args.langs.split(",")])

    # The detections carry a bounding box, so colour can be measured on the
    # SUBJECT rather than the whole frame — a lion-dance stage is red whatever
    # colour the lion is, so a global histogram settles nothing.
    colours = None
    det_lookup = None
    if not args.no_colours:
        import zipfile

        colours = ColourIndex(args.data)
        zpath = Path(args.data) / "objects-aic25-b1.zip"
        if zpath.exists():
            zf = zipfile.ZipFile(zpath)
            stem_of = {(m["video_id"], m["frame_idx"]): Path(m["frame_filename"]).stem
                       for m in eng.metadata}

            def det_lookup(v, f):  # noqa: F811
                stem = stem_of.get((v, int(f)))
                if not stem:
                    return None
                try:
                    import json as _json

                    return _json.loads(zf.read(f"objects/{v}/{stem}.json"))
                except Exception:  # noqa: BLE001
                    return None
    items = [(v, f, fn) for (v, f), fn in wanted.items()]

    if args.colours_only:
        if colours is None:
            print("--colours-only cung voi --no-colours thi khong lam gi ca")
            return 2
        left = [i for i in items if colours.get(i[0], i[1]) is None]
        print(f"\n{len(items)} frame, {len(items) - len(left)} da do mau, {len(left)} can do")
        if left:
            import io as _io
            import urllib.request as _rq
            from concurrent.futures import ThreadPoolExecutor as _TPE

            from PIL import Image as _Image

            from src.core.ocr import CDN as _CDN
            from src.core.ocr import UA as _UA

            def _one(job):
                v, f, fn = job
                try:
                    raw = _rq.urlopen(
                        _rq.Request(f"{_CDN}/{v}/{fn}", headers=_UA), timeout=40
                    ).read()
                    return v, f, _Image.open(_io.BytesIO(raw)).convert("RGB")
                except Exception:  # noqa: BLE001
                    return v, f, None

            t0 = time.time()
            with _TPE(max_workers=12) as ex:
                for i, (v, f, img) in enumerate(ex.map(_one, left), 1):
                    if img is not None:
                        colours.put(v, f, img, det_lookup(v, f) if det_lookup else None)
                    if i % 100 == 0:
                        colours.flush()
                        print(f"  {i}/{len(left)}  ({i/max(time.time()-t0,1e-9):.1f} frame/s)", flush=True)
            colours.flush()
        print(f"mau: {colours.n_frames} khung hinh")
        return 0

    todo = [i for i in items if ocr.get(i[0], i[1]) is None]
    print(f"\n{len(items)} frame tat ca, {len(items) - len(todo)} da doc truoc do, "
          f"{len(todo)} can doc (~{len(todo) * 4 / 60:.0f} phut)", flush=True)
    if not todo:
        return 0

    t0 = time.time()

    def progress(done, total):
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done}/{total}  ({rate:.1f} frame/s, con ~{(total - done) / max(rate, 1e-9) / 60:.0f} phut)",
              flush=True)

    ocr.read_frames(todo, progress=progress, colours=colours, detections=det_lookup)
    if colours is not None:
        colours.flush()
        print(f"mau: {colours.n_frames} khung hinh")

    withtext = sum(1 for v, f, _fn in items if ocr.text_of(v, f).strip())
    print(f"\nxong: {withtext}/{len(items)} frame co chu ({100 * withtext / max(len(items), 1):.0f}%)")
    print("Gio chay lai build_review_page.py de hien chu tren tung khung hinh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
