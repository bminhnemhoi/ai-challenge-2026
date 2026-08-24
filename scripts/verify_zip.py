"""Check any submission zip before uploading it.

review.html now builds the upload itself, in the browser, so there is a path to
the organisers that never passes through make_submission.py. That path needs the
same last gate as every other one — a format rejection still costs one of the
three attempts per round, and the browser cannot know which queries this round
was supposed to contain.

    python scripts/verify_zip.py ~/Downloads/submission.zip --queries round_p1/queries

Loading the 780 MB index is not needed for this, so it answers in under a second.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    csv_name_for_query,
    verify_submission_zip,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", help="the file you are about to upload")
    ap.add_argument("--queries", default=None, help="query folder, to check none is missing")
    ap.add_argument(
        "--allow-blank-answers",
        action="store_true",
        help="format smoke-test only; blank Q&A answers score 0",
    )
    args = ap.parse_args()

    zp = Path(args.zip).expanduser()
    if not zp.is_file():
        print(f"ERROR: no such file: {zp}")
        return 2

    expect = None
    if args.queries:
        qdir = Path(args.queries)
        expect = {
            csv_name_for_query(p.name)
            for p in qdir.glob("*.txt")
            if not p.name.lower().endswith((".en.txt", ".vi.txt"))
        }
        if not expect:
            print(f"ERROR: no query .txt files in {qdir}")
            return 2

    problems = verify_submission_zip(
        zp, expect_names=expect, allow_blank_answers=args.allow_blank_answers
    )

    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        rows = sum(
            len([ln for ln in z.read(n).decode("utf-8-sig").splitlines() if ln.strip()])
            for n in names
        )
        qa = [n for n in names if "-qa" in Path(n).name.lower()]
        blank = 0
        for n in qa:
            first = z.read(n).decode("utf-8-sig").splitlines()[0].split(",")
            if len(first) < 3 or not first[2].strip():
                blank += 1

    print(f"{zp}  ({zp.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(names)} CSV, {rows} rows total, {rows / max(1, len(names)):.0f} per query")
    if expect:
        print(f"  expected {len(expect)} queries from {args.queries}")
    if blank:
        print(f"  {blank}/{len(qa)} Q&A queries have a blank answer on row 1")

    if problems:
        print("\nFORMAT PROBLEMS — do not upload this:")
        for p in problems:
            print("  -", p)
        return 1
    if rows < len(names) * MAX_ROWS:
        # not an error, but leaving rank slots empty is free score thrown away
        print(f"\n  note: {len(names) * MAX_ROWS - rows} of the "
              f"{len(names) * MAX_ROWS} available rank slots are unused. "
              "Extra rows can never lower a score.")
    print("\nformat check passed. This zip is safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
