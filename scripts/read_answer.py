"""Answer a Q&A query off named frames, at the frame's FULL resolution.

The reranker downscales every frame to 512px, which is right for "is this the
scene" and wrong for "what does the sign say". Round 1 asked for a number on a
bridge sign, a number on a scale, and a count of markers on a distribution map;
at 512px the sign is a smudge and the markers merge, and the model answered with
an estimate it openly called unreliable. The keyframes on the CDN are ~1280px,
so the fix costs nothing but bytes: send the original.

Frames are named explicitly rather than searched, because by this point the
frame is already settled — this is the reading step, not the finding step.

    python scripts/read_answer.py --video L21_V006 --frames 561 \
        --question "Khong tinh bang chu giai, co bao nhieu vi tri dong dat cap do 4?"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.vlm import CDN, UA, VLMJudge, load_env  # noqa: E402


def full_frame(video_id: str, filename: str, max_side: int) -> bytes | None:
    """The keyframe as published, only downscaled if it exceeds `max_side`."""
    from PIL import Image

    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(f"{CDN}/{video_id}/{filename}", headers=UA), timeout=60
        ).read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        print(f"    khong tai duoc: {type(exc).__name__}: {str(exc)[:70]}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", required=True, help="comma-separated frame indices")
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--neighbours", type=int, default=0, help="also read n keyframes either side")
    ap.add_argument("--max-tokens", type=int, default=900,
                    help="thinking models spend most of their budget before the first word")
    args = ap.parse_args()

    load_env(Path(args.data).parent / ".env")
    load_env(".env")
    import os

    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Khong co GEMINI_API_KEY")
        return 2
    client = genai.Client(api_key=key)

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    arr = sorted(
        (int(m["frame_idx"]), m["frame_filename"])
        for m in meta
        if m["video_id"] == args.video
    )
    if not arr:
        print(f"khong co keyframe cho {args.video}")
        return 2

    wanted: list[tuple[int, str]] = []
    for raw in args.frames.split(","):
        f = int(raw.strip())
        i = min(range(len(arr)), key=lambda x: abs(arr[x][0] - f))
        lo, hi = max(0, i - args.neighbours), i + args.neighbours + 1
        wanted.extend(arr[lo:hi])
    seen: set[int] = set()
    wanted = [x for x in wanted if not (x[0] in seen or seen.add(x[0]))]

    print(f"{args.video}: doc {len(wanted)} khung hinh o toi da {args.max_side}px\n")
    for f, fn in wanted:
        blob = full_frame(args.video, fn, args.max_side)
        if not blob:
            continue
        try:
            r = client.models.generate_content(
                model=args.model,
                contents=[types.Part.from_bytes(data=blob, mime_type="image/jpeg"), args.question],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=args.max_tokens),
            )
            txt = " ".join((r.text or "").split())
            print(f"  frame {f:<7d} ({len(blob) // 1024} KB)  {txt[:900]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  frame {f:<7d} LOI {type(exc).__name__}: {str(exc)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
