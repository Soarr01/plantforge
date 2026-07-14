"""PLANTFORGE — a procedural control-plant corpus for in-context system identification.

Gap adversarially verified 2026-07-12 (two independent verifications); gate PASSED the
same day: in-context SysID transformers trained on Wiener-Hammerstein-only white-noise
data (the field's current private-generator default) degrade 32.6x crossing nonlinearity
family and 10.6x crossing sampling rate. The corpus fixes both axes: 5 control-plant
nonlinearity families x 4 excitation classes x exact-ZOH multi-rate ground truth, with
per-instance Fisher identifiability annotations.
"""
__version__ = "0.1.0"
