"""
scripts/analyze.py
==================
Convenience entry point to analyze results and regenerate figures.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_results import main

if __name__ == "__main__":
    main()
