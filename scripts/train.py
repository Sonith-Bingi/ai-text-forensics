#!/usr/bin/env python
"""Full training pipeline: data -> features -> encoder CV -> blender + calibration.
Every stage is disk-cached (see forensics/pipeline.py), so re-running after an
interruption resumes rather than restarting."""
from forensics.pipeline import train_all

if __name__ == "__main__":
    train_all()
