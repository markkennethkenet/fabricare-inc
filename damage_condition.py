"""V6 care features: identified fabric + detected fabric condition + extracted image features.

The V6 damage-condition classifier labels a whole photo as one of DAMAGE_CLASSES. The care models
(Decision Tree, Random Forest, SVM) read the fabric and condition as one-hot inputs plus the same colour
and GLCM texture features as care_model. The reference laundry mode is care_model's referenced rule set,
with the detected condition standing in for the detector's findings: a hole, stain, pilling or needle mark
counts as that finding; "normal" and "wrinkle" count as none (a wrinkle does not change the wash method).
A model trained on these labels learns that reference, so its accuracy measures agreement with the rules.
"""
from __future__ import annotations

import numpy as np

import care_model

DAMAGE_CLASSES = ("hole", "needlemark", "normal", "pilling", "stain", "wrinkle")
CONDITION_FINDINGS = {"hole": "hole", "needlemark": "needlemark", "pilling": "pilling", "stain": "stain"}

FEATURE_NAMES = (
    [f"fabric_{name}" for name in care_model.FABRICS]
    + [f"condition_{name}" for name in DAMAGE_CLASSES]
    + ["luminance_mean", "luminance_std", "saturation_mean", "dark_fraction"]
    + [f"glcm_{prop}" for prop in care_model.GLCM_PROPERTIES]
)


def condition_features(condition):
    if condition not in DAMAGE_CLASSES:
        raise ValueError(f"Unknown fabric condition: {condition!r}")
    return {f"condition_{name}": 1.0 if name == condition else 0.0 for name in DAMAGE_CLASSES}


def feature_row(fabric_type, condition, color, glcm):
    """One care case as a dictionary covering FEATURE_NAMES."""
    row = {**care_model.fabric_features(fabric_type), **condition_features(condition), **color, **glcm}
    missing = [name for name in FEATURE_NAMES if name not in row]
    if missing:
        raise ValueError(f"Missing care features: {', '.join(missing)}")
    return row


def feature_vector(row):
    return np.asarray([[row[name] for name in FEATURE_NAMES]], dtype=np.float64)


def rule_row(row):
    """The care_model row the referenced rules read: the condition becomes one damage finding."""
    condition = next(name for name in DAMAGE_CLASSES if row[f"condition_{name}"] >= 0.5)
    damage = {f"{kind}_{stat}": 0.0 for kind in care_model.DAMAGE_TYPES for stat in ("count", "max_area")}
    finding = CONDITION_FINDINGS.get(condition)
    if finding:
        damage[f"{finding}_count"] = 1.0
    fabric = {key: row[key] for key in row if key.startswith("fabric_")}
    color = {key: row[key] for key in ("luminance_mean", "luminance_std", "saturation_mean", "dark_fraction")}
    glcm = {key: row[key] for key in row if key.startswith("glcm_")}
    return {**fabric, **damage, **color, **glcm}


def reference_laundry_mode(row):
    """(mode, reasons) from care_model's referenced rules for this fabric and condition."""
    return care_model.reference_laundry_mode(rule_row(row))
