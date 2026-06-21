"""
Streamlit entrypoint for GitHub / Streamlit Cloud deployment.

Use:
    streamlit run streamlit_app.py
"""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "app" / "dashboard.py"), run_name="__main__")
