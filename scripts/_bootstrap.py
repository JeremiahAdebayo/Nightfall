"""
Puts the repo root on sys.path so scripts work when run directly
(`python scripts/train.py`) as well as when the package is installed
(`pip install -e .`).

Why this is needed: running `python scripts/X.py` puts scripts/ on
sys.path, not the repo root, so `import nightfall...` fails unless the
package has been installed. Import this module (not a copy-pasted path
snippet) at the top of every script that needs it -- the previous layout
duplicated the same three sys.path lines in four files with three
slightly different comments, which is exactly how they drifted out of
sync with the import names they were guarding.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
