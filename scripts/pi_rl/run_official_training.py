#!/usr/bin/env python3
"""Install observations before the official entrypoint creates Ray actors."""

import runpy
import sys
from pathlib import Path

from training_observer import register


def main() -> None:
    entry = Path(sys.argv[1]).resolve()
    register()
    sys.argv = [str(entry), *sys.argv[2:]]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
