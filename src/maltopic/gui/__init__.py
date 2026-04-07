"""
MALTopic GUI - A Streamlit-based web interface for the MALTopic topic modeling library.

Launch the GUI by running:
    maltopic-gui
    # or
    python -m maltopic.gui
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the MALTopic Streamlit GUI application."""
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.maxUploadSize=200",
            "--browser.gatherUsageStats=false",
            "--client.toolbarMode=minimal",
        ],
        check=False,
    )
