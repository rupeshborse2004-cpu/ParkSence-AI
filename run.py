#!/usr/bin/env python
"""
ParkSense AI - single entry point.

    python run.py            # build artefacts if missing, then open the dashboard
    python run.py build      # (re)build all data/model artefacts only
    python run.py dashboard  # launch the Streamlit dashboard only
    python run.py all        # force a rebuild, then launch the dashboard

Designed so a judge can clone the repo and run one command.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
APP = os.path.join(ROOT, "app", "dashboard.py")
KPI = os.path.join(ROOT, "outputs", "processed", "kpis.json")
MODEL = os.path.join(ROOT, "outputs", "models", "risk_lgbm.pkl")

PY = sys.executable  # use the exact interpreter that launched this script


def _artefacts_ready() -> bool:
    return os.path.exists(KPI) and os.path.exists(MODEL)


def build() -> None:
    print(">> Building ParkSense AI artefacts ...")
    sys.path.insert(0, SRC)
    import build_all  # noqa: E402  (imported lazily so `dashboard` is fast)
    build_all.main()


def dashboard() -> None:
    if not _artefacts_ready():
        print(">> Artefacts missing - building them first ...")
        build()
    print(">> Launching dashboard at http://localhost:8501  (Ctrl+C to stop)")
    subprocess.run([PY, "-m", "streamlit", "run", APP], cwd=ROOT, check=False)


def main() -> None:
    cmd = (sys.argv[1].lower() if len(sys.argv) > 1 else "run")
    if cmd == "build":
        build()
    elif cmd == "dashboard":
        dashboard()
    elif cmd == "all":
        build()
        dashboard()
    elif cmd in ("run", "start"):
        dashboard()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
