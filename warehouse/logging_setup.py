"""Tee stdout / stderr to a rolling log file so every print() + traceback
lands somewhere the user can copy.

Call `install()` once at process start (server.py does this)."""
from __future__ import annotations

import atexit
import io
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "server.log"


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


_installed = False
_log_file = None


def install() -> Path:
    """Redirect stdout+stderr to both the terminal AND the log file. Idempotent."""
    global _installed, _log_file
    if _installed:
        return LOG_PATH

    _log_file = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
    _log_file.write(f"\n\n===== session start {datetime.now().isoformat(timespec='seconds')} =====\n")

    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_out, _log_file)
    sys.stderr = _Tee(real_err, _log_file)

    atexit.register(_log_file.close)
    _installed = True
    print(f"[logs] tee installed → {LOG_PATH}")
    return LOG_PATH


def tail(n: int = 200) -> str:
    if not LOG_PATH.exists():
        return ""
    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])
