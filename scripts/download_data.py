#!/usr/bin/env python
"""Downloads and caches MAGE + builds the generalization splits + pulls the
RAID adversarial-attack sample. Run this once before `scripts/train.py`."""
from forensics.data.raid import build_raid_sample
from forensics.data.splits import build_splits

if __name__ == "__main__":
    build_splits()
    build_raid_sample()
