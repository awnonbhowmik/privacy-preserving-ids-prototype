"""
app.py — Project root launch helper.

Streamlit's multipage app discovers pages/ relative to the file passed to
`streamlit run`. Because pages/ lives inside app/, the correct launch command
is always:

    streamlit run app/main.py

Running this file directly executes that command for convenience.

Usage:
    python app.py        # launches the app via subprocess
    # OR directly:
    streamlit run app/main.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    entry = Path(__file__).parent / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(entry)]
    print(f"Launching: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
