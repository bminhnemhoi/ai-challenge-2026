"""Pull YouTube auto-captions for the corpus — the text channel we do not have.

Every retrieval decision this system makes comes from one modality: what a
keyframe looks like. The AIC 2025 team that scored 79/88 (MERVIN, arXiv:2605.16120)
ran three text channels alongside the visual one. We run zero.

Captions are the cheapest of the three and need no API key: the organisers'
media-info gives a YouTube URL for all 873 videos, and the Vietnamese
auto-captions come back clean and timestamped — 506 segments and 19k characters
for a 20-minute news bulletin, at roughly 3 seconds a video.

They matter most where the picture cannot help. query-p1-21 asks for research at
a university in Lausanne into insect flight for robotics; the visual model put a
Vietnamese lifestyle video first, ahead of the right kind of video by 0.003 —
1.8% — because a news anchor at a desk looks like a news anchor at a desk. A
spoken word does not have that problem.

    python scripts/fetch_captions.py                 # everything, resumable
    python scripts/fetch_captions.py --prefix L21,L22   # just the news bulletins
    python scripts/fetch_captions.py --search "Lausanne,rô bốt"

Output: data/captions/<video_id>.json  ->  [[start_seconds, text], ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

UA = {"User-Agent": "Mozilla/5.0"}


def watch_urls(data_dir: Path) -> dict:
    z = data_dir / "media-info-aic25-b1.zip"
    out = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"):
                continue
            j = json.loads(zf.read(n))
            url = j.get("watch_url")
            if url:
                out[Path(n).stem] = url
    return out


class BotCheck(RuntimeError):
    """YouTube demanded a sign-in. Not a per-video problem — the whole sweep is blocked."""


def fetch_one(video_id: str, url: str, langs, cookies=None, sleep=0.0) -> list | None:
    """[[start_seconds, text], ...] or None when the video has no captions.

    json3 is requested specifically: it carries per-cue start times, which is
    what makes a caption hit usable as a *timestamp* rather than only as a
    video-level filter.
    """
    import yt_dlp

    if sleep:
        time.sleep(sleep)
    opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": list(langs),
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
    }
    if cookies:
        if str(cookies).lower().endswith(".txt"):
            # duong du phong khi Chrome khoa cookie DB (yt-dlp #7271):
            # xuat cookies.txt bang extension "Get cookies.txt LOCALLY" roi
            # --cookies-from-browser duong\dan\cookies.txt
            opts["cookiefile"] = str(cookies)
        else:
            opts["cookiesfrombrowser"] = (cookies,)
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # After a few hundred anonymous requests YouTube starts demanding a
        # sign-in. Every later video then "has no captions", which would be
        # cached as a permanent negative — so this is raised as its own error
        # and the sweep stops instead of poisoning the cache.
        if "not a bot" in msg or "Please sign in" in msg or "cookies" in msg:
            raise BotCheck(msg) from exc
        raise

    tracks = None
    for src in ("subtitles", "automatic_captions"):
        for lang in langs:
            t = (info.get(src) or {}).get(lang)
            if t:
                tracks = t
                break
        if tracks:
            break
    if not tracks:
        return None

    href = next((t["url"] for t in tracks if t.get("ext") == "json3"), None)
    if not href:
        return None
    raw = urllib.request.urlopen(urllib.request.Request(href, headers=UA), timeout=40).read()
    events = json.loads(raw).get("events", [])
    segs = []
    for e in events:
        if not e.get("segs"):
            continue
        text = "".join(s.get("utf8", "") for s in e["segs"]).strip()
        if text:
            segs.append([round(e.get("tStartMs", 0) / 1000, 2), text])
    return segs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="default: <data>/captions")
    ap.add_argument("--prefix", default=None, help="comma-separated video prefixes, e.g. L21,L22")
    ap.add_argument("--langs", default="vi,en", help="preferred caption languages, in order")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search", default=None, help="after fetching, grep the cache for these terms")
    ap.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="chrome / edge / firefox — needed once YouTube starts asking for a sign-in, "
        "which it does after a few hundred anonymous requests. Log in to YouTube in that "
        "browser first, then re-run; the sweep resumes where it stopped.",
    )
    ap.add_argument(
        "--retry-empty",
        action="store_true",
        help="re-ask for videos cached as having no captions. Needed after a run that hit "
        "the bot check: everything requested while blocked looked caption-less and was "
        "cached as a permanent negative.",
    )
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait before each request; 1.0 with --workers 2 "
                    "usually stays under the anonymous limit")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out else data_dir / "captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    urls = watch_urls(data_dir)
    if args.prefix:
        keep = tuple(x.strip() for x in args.prefix.split(","))
        urls = {k: v for k, v in urls.items() if k.startswith(keep)}
    def cached(k):
        f = out_dir / f"{k}.json"
        if not f.exists():
            return False
        return not (args.retry_empty and f.stat().st_size <= 4)

    todo = {k: v for k, v in urls.items() if not cached(k)}
    if args.limit:
        todo = dict(list(todo.items())[: args.limit])

    have = len(urls) - len(todo)
    print(f"{len(urls)} video, {have} da co san, {len(todo)} can tai", flush=True)

    if todo:
        t0 = time.time()
        done = failed = blocked = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_one, k, v, langs, args.cookies_from_browser, args.sleep): k
                    for k, v in todo.items()}
            for i, f in enumerate(as_completed(futs), 1):
                vid = futs[f]
                try:
                    segs = f.result()
                except BotCheck:
                    blocked += 1
                    continue          # do NOT cache: this video was never asked
                except Exception as exc:  # noqa: BLE001 - one dead video must not stop the sweep
                    segs = None
                    if failed < 3:
                        print(f"  {vid}: {type(exc).__name__}: {str(exc)[:80]}")
                # an empty list is cached too, so a re-run does not retry a
                # video that genuinely has no captions
                (out_dir / f"{vid}.json").write_text(
                    json.dumps(segs if segs is not None else [], ensure_ascii=False),
                    encoding="utf-8",
                )
                if segs:
                    done += 1
                else:
                    failed += 1
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"  {i}/{len(todo)}  {done} co phu de, {failed} khong  "
                          f"({rate:.1f}/s, con ~{(len(todo)-i)/max(rate,1e-9)/60:.0f} phut)", flush=True)

        if blocked:
            print(f"\n!! YouTube chan {blocked} video: no doi dang nhap.")
            print("   Dang nhap YouTube tren Chrome/Edge roi chay lai voi:")
            print("     python scripts/fetch_captions.py --cookies-from-browser chrome")
            print("   Hoac cham hon:  --workers 2 --sleep 1.0")
            print("   Nhung video da tai xong duoc giu lai, chay lai se tiep tuc tu do.")

    files = sorted(out_dir.glob("*.json"))
    withtext = [p for p in files if p.stat().st_size > 4]
    total_chars = sum(
        len(" ".join(t for _s, t in json.loads(p.read_text(encoding="utf-8")))) for p in withtext
    )
    print(f"\n{len(withtext)}/{len(files)} video co phu de, tong {total_chars/1e6:.1f} trieu ky tu")

    if args.search:
        terms = [t.strip().lower() for t in args.search.split(",") if t.strip()]
        print(f"\ntim '{', '.join(terms)}':")
        hits = 0
        for p in withtext:
            segs = json.loads(p.read_text(encoding="utf-8"))
            for s, t in segs:
                low = t.lower()
                for term in terms:
                    if term in low:
                        print(f"  {p.stem}  giay {s:7.1f}  ...{t[:80]}")
                        hits += 1
                        break
        print(f"  {hits} lan xuat hien")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
