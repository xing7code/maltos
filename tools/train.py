from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _THIS_DIR]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train import cli as _train_cli

for _name, _value in vars(_train_cli).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value


if __name__ == "__main__":
    main()
