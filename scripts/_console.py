"""Make stdout safe for Vietnamese on a Windows console.

Python picks its stdout encoding from the active code page.  On a default
Vietnamese/US Windows install that is cp1252, and printing a single "ạ" raises
UnicodeEncodeError.  In these scripts the crash would land *between* writing
some CSVs and repackaging the zip, leaving a half-updated submission — so the
console is forced to UTF-8 with replacement instead of being allowed to fail.

Import for the side effect, before anything prints:

    from scripts._console import safe_console; safe_console()
"""

from __future__ import annotations

import sys


def safe_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a pipe or a captured buffer; nothing to fix
            pass
