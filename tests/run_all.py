"""Раннер тестов без pytest: запускает все функции test_* из tests/test_*.py.

  ~/Coding/Python/CERBER/.venv/bin/python -m tests.run_all
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent))
    failed = 0
    total = 0
    for mod_info in sorted(pkgutil.iter_modules([str(root)]), key=lambda m: m.name):
        if not mod_info.name.startswith("test_"):
            continue
        mod = importlib.import_module(f"tests.{mod_info.name}")
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            total += 1
            t0 = time.time()
            try:
                fn()
                print(f"  ok    {mod_info.name}.{name}  ({time.time() - t0:.2f}s)")
            except Exception:
                failed += 1
                print(f"  FAIL  {mod_info.name}.{name}")
                traceback.print_exc()
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
