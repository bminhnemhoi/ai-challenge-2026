"""Rebuild and re-verify the zip after hand corrections.

Any time a CSV is edited — by pin_video.py, or by typing a Q&A answer into
column 3 — the archive has to be rebuilt and checked again.  A format
rejection still costs one of the three attempts per round, so this never skips
the verification step.

    python scripts/repackage.py --out round_p1/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.submission import package_submission, verify_submission_zip  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the make_submission --out directory")
    ap.add_argument("--queries", default=None, help="query folder, to check none is missing")
    ap.add_argument(
        "--allow-blank-answers",
        action="store_true",
        help="format smoke-test only; blank Q&A answers score 0",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    csv_dir = out_dir / "csv"
    if not csv_dir.is_dir():
        print(f"ERROR: {csv_dir} does not exist")
        return 2

    expect = None
    if args.queries:
        from src.core.submission import csv_name_for_query

        qdir = Path(args.queries)
        expect = {
            csv_name_for_query(p.name)
            for p in qdir.glob("*.txt")
            if not p.name.lower().endswith((".en.txt", ".vi.txt"))
        }

    zip_path = out_dir / "submission.zip"
    package_submission(csv_dir, zip_path)
    problems = verify_submission_zip(
        zip_path, expect_names=expect, allow_blank_answers=args.allow_blank_answers
    )

    files = sorted(csv_dir.glob("*.csv"))
    print(f"packed {len(files)} CSV files -> {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    if problems:
        print("\nFORMAT PROBLEMS — do not upload this:")
        for p in problems:
            print("  -", p)
        return 1
    print("format check passed. This zip is safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
