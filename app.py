"""Entry point.

    streamlit run app.py          conversational web interface
    python app.py "your question" one-shot answer in the terminal
    python app.py                 interactive terminal session

Running the module directly keeps `python -m src.ui.cli` from re-importing the
package mid-execution, which is where the RuntimeWarning came from.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _running_under_streamlit() -> bool:
    """True only when `streamlit run` is driving this file.

    Detected from sys.modules rather than by importing streamlit: importing it
    outside a script run emits a bare-mode warning, which would appear on every
    terminal invocation.
    """
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_under_streamlit():
    from src.ui.streamlit_app import main as run_web
    run_web()
elif __name__ == "__main__":
    from src.ui.cli import main as run_cli
    raise SystemExit(run_cli(sys.argv[1:]))
