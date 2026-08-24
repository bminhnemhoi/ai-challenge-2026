"""Search what is SAID in the corpus. Seconds, no model, no API.

The visual index cannot see a proper noun. Searching the transcripts by hand
found MĂNG TÂY CHIÊN BIA for query-p1-4 at rank 1 and CỦ NĂNG OM NẤM CHAY for
query-p1-18 — neither of which the visual ranking surfaced at all, and the second
was not even in its top six.

An automatic version of this was tried and dropped. Folded into the score it is
negative at every weight on the ground truth, and only +0.5% (noise) even when
gated on decisive evidence: those 60 queries are visual-scene descriptions
("a dark red sedan with a rear spoiler") that nobody says out loud. Worse, a
machine guessing which passage the operator wants gets it wrong both ways — it
missed query-p1-4, whose discriminative unit is the BIGRAM "măng tây", and
volunteered passages about unrelated robots for query-p1-21.

So this is a tool, not a stage in a pipeline. The operator knows what to search
for; the machine does not.

    python scripts/search_transcripts.py "măng tây chiên bột"
    python scripts/search_transcripts.py "củ năng" --videos L26
    python scripts/search_transcripts.py "Nguyễn Trung Trực" -n 8

Each hit prints the video, the passage, and a YouTube link that opens two
seconds before the words are spoken.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.transcripts import TranscriptIndex, tokenise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="+", help="what you expect to hear")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--transcripts", default=str(ROOT.parent / "transcripts_full"))
    ap.add_argument("-n", "--top", type=int, default=6)
    ap.add_argument("--videos", default=None, help="restrict to ids starting with this, e.g. L26")
    ap.add_argument("--passages", type=int, default=2, help="passages to show per video")
    args = ap.parse_args()

    query = " ".join(args.query)
    data = Path(args.data)
    tx = TranscriptIndex().load_dir(
        *[Path(d) for d in args.transcripts.split(",") if d.strip()], data / "captions"
    )
    if not tx.n_videos:
        print("Khong tim thay loi thoai nao. Chay: python scripts/fetch_captions.py")
        return 2

    urls = {}
    fps = {}
    z = data / "media-info-aic25-b1.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".json"):
                    urls[Path(n).stem] = json.loads(zf.read(n)).get("watch_url", "")
    meta = data / "metadata.json"
    if meta.exists():
        for m in json.loads(meta.read_text(encoding="utf-8")):
            fps.setdefault(m["video_id"], float(m["fps"]))

    restrict = None
    if args.videos:
        pre = tuple(x.strip() for x in args.videos.split(","))
        restrict = [v for v in tx.docs if v.startswith(pre)]

    scores = tx.score_videos(query, restrict=restrict)
    if not scores:
        print(f"Khong co video nao nhac toi: {query!r}")
        return 1

    terms = {t for t in tokenise(query) if "_" not in t}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: args.top]
    print(f"\n{tx.n_videos} video co loi thoai · tim: {query!r}\n")

    for rank, (vid, sc) in enumerate(ranked, 1):
        title = tx.titles.get(vid, "")
        print(f"{rank}. {vid}  (bm25 {sc:.1f})  {title[:62]}")
        segs = tx.segments.get(vid, [])
        # the passages that actually contain a query word, best first, so the
        # operator can see WHY this video came up instead of trusting a number
        scored = []
        for i in range(len(segs)):
            chunk = segs[i : i + 5]
            text = " ".join(x for _t, x in chunk)
            low = text.lower()
            hits = sum(1 for t in terms if t in low)
            if hits:
                scored.append((hits, -i, chunk[0][0], text))
        scored.sort(reverse=True)
        seen_at = []
        for _h, _i, at, text in scored:
            if any(abs(at - a) < 25 for a in seen_at):
                continue
            seen_at.append(at)
            u = urls.get(vid, "")
            link = (
                u.replace("https://youtube.com/watch?v=", "https://youtu.be/")
                + f"?t={max(0, int(at) - 2)}"
                if u
                else ""
            )
            f = int(at * fps.get(vid, 25.0))
            print(f"     {int(at)//60:d}:{int(at)%60:02d}  frame {f:<7d} {link}")
            print(f"        …{text[:150]}…")
            if len(seen_at) >= args.passages:
                break
        if not scored:
            print("     (khop tren tu chung, khong co doan nao chua tu khoa)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
