#!/usr/bin/env python3
"""
Patch OpenNRE for compatibility with modern transformers and Windows.

Run once after installing OpenNRE:
    python patch_opennre.py

Fixes:
1. UTF-8 encoding for data file loading (Windows defaults to cp1251/cp1252)
2. AdamW import (removed from transformers >=4.x, use torch.optim.AdamW)
3. num_workers=0 (avoids CUDA multiprocessing OOM on Windows)
"""

import importlib
import re
from pathlib import Path


def patch():
    import opennre.framework.data_loader as dl
    import opennre.framework.sentence_re as sre

    dl_path = Path(dl.__file__)
    sre_path = Path(sre.__file__)

    patched = []

    # --- data_loader.py ---
    dl_text = dl_path.read_text(encoding="utf-8")
    dl_orig = dl_text

    # Fix 1: encoding='utf-8' on all open(path) calls
    dl_text = dl_text.replace("f = open(path)", "f = open(path, encoding='utf-8')")

    # Fix 3: num_workers default 8 -> 0
    dl_text = dl_text.replace("num_workers=8", "num_workers=0")

    if dl_text != dl_orig:
        dl_path.write_text(dl_text, encoding="utf-8")
        patched.append(str(dl_path))

    # --- sentence_re.py ---
    sre_text = sre_path.read_text(encoding="utf-8")
    sre_orig = sre_text

    # Fix 2: Replace transformers.AdamW with torch.optim.AdamW
    sre_text = sre_text.replace(
        "from transformers import AdamW",
        "from torch.optim import AdamW",
    )
    # Remove correct_bias kwarg (not supported by torch AdamW)
    sre_text = sre_text.replace(
        "self.optimizer = AdamW(grouped_params, correct_bias=False)",
        "self.optimizer = AdamW(grouped_params)",
    )

    if sre_text != sre_orig:
        sre_path.write_text(sre_text, encoding="utf-8")
        patched.append(str(sre_path))

    if patched:
        print(f"Patched {len(patched)} file(s):")
        for p in patched:
            print(f"  {p}")
    else:
        print("Already patched, nothing to do.")


if __name__ == "__main__":
    patch()
