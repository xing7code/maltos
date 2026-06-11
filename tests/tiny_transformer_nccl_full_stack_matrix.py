from __future__ import annotations

import sys

from tiny_transformer_full_stack_matrix_runner import main


if __name__ == "__main__":
    if "--backend" not in sys.argv:
        sys.argv.extend(["--backend", "nccl"])
    main()
