"""
scripts/run_scripted.py
=======================
Convenience entry point to run the standard scripted benchmark suite.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiments import main

if __name__ == "__main__":
    main()
