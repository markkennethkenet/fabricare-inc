"""Data handling for the V7 damage detector: pre-label naming, garment-grouped splits, and class mapping.

The app's YOLOv8 detector predicts four classes in this order: hole, needlemark, pilling, stain. Photos the team
labels in Roboflow keep a garment prefix in their file names ("tshirt_003__front__IMG_123.jpg"), which survives
Roboflow's renaming ("tshirt_003__front__IMG_123_jpg.rf.<hash>.jpg"), so every photo of one garment can be kept
in one split. Nothing here touches Drive or the network.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

APP_CLASSES = ("hole", "needlemark", "pilling", "stain")
SPLITS = ("train", "valid", "test")
TARGET = {"train": 0.7, "valid": 0.15, "test": 0.15}
# The public 6-class set (docs/damage-yolov8-dataset.md): drop classes the app does not detect.
PUBLIC_CLASSES = ("hole", "needlemark", "pilling", "stain", "missing_yarn", "wrinkle")
PUBLIC_TO_APP = {i: APP_CLASSES.index(name) for i, name in enumerate(PUBLIC_CLASSES) if name in APP_CLASSES}

_ROBOFLOW = re.compile(r"_(?:jpe?g|png|heic)\.rf\.[0-9a-f]+$", re.I)
_SAFE = re.compile(r"[^a-z0-9]+")


def safe(text):
    return _SAFE.sub("_", text.strip().lower()).strip("_") or "x"


def prelabel_name(garment, view, original_name):
    """File name for a photo sent to Roboflow: garment and view prefixes, original stem, .jpg."""
    stem = original_name.rsplit(".", 1)[0]
    return f"{safe(garment)}__{safe(view)}__{safe(stem)}.jpg"


def garment_of(file_name):
    """Garment group of a labelled photo; photos without the prefix form their own group."""
    stem = _ROBOFLOW.sub("", file_name.rsplit(".", 1)[0])
    if "__" in stem:
        return stem.split("__", 1)[0]
    return "photo:" + stem.lower()


def _order(group):
    return hashlib.sha256(group.encode("utf-8")).hexdigest()


def split_by_garment(file_names):
    """{file name: split}, whole garments per split, aiming for 70/15/15 of photos (deterministic)."""
    groups = defaultdict(list)
    for name in file_names:
        groups[garment_of(name)].append(name)
    placement, filled = {}, Counter()
    for group in sorted(groups, key=_order):
        size = len(groups[group])
        total = sum(filled.values()) + size
        split = max(SPLITS, key=lambda s: (TARGET[s] * total - filled[s], -SPLITS.index(s)))
        filled[split] += size
        for name in groups[group]:
            placement[name] = split
    return placement


def remap_label_lines(text, mapping):
    """YOLO label text with class ids remapped; lines whose class is not in mapping are dropped."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if cls in mapping:
            out.append(" ".join([str(mapping[cls])] + parts[1:]))
    return "\n".join(out) + ("\n" if out else "")


def roboflow_mapping(names):
    """Map a Roboflow export's class list (any order, any case) onto APP_CLASSES; unknown classes are dropped."""
    aliases = {"needle_mark": "needlemark", "needle_marks": "needlemark", "needlemarks": "needlemark",
               "holes": "hole", "stains": "stain", "pill": "pilling", "pills": "pilling"}
    mapping = {}
    for index, name in enumerate(names):
        key = "_".join(str(name).strip().lower().replace("-", " ").split())   # "Needle Marks" -> "needle_marks"
        key = aliases.get(key, key)
        if key in APP_CLASSES:
            mapping[index] = APP_CLASSES.index(key)
    return mapping


def online_name(folder, original_name):
    """Name for an Online_Dataset photo: no "__", so each online photo is its own group when splitting."""
    stem = original_name.rsplit(".", 1)[0]
    return f"online-{safe(folder).replace('_', '-')}-{safe(stem)}.jpg"


def original_name(file_name):
    """A labelled photo's name before Roboflow renamed it ("x_jpg.rf.<hash>.jpg" -> "x.jpg")."""
    stem = _ROBOFLOW.sub("", file_name.rsplit(".", 1)[0])
    return f"{stem}.jpg"


def round_zip_name(round_number):
    """Pre-label zip for a labelling round; round 1 keeps the original name."""
    return "v7_prelabels.zip" if round_number <= 1 else f"v7_prelabels_round{round_number}.zip"


def newest_weights(run_dir, base_weights):
    """The most recently trained V7 detector (RUN/train*/weights/best.pt), or the app's detector if none yet."""
    from pathlib import Path

    trained = [p for p in Path(run_dir).glob("train*/weights/best.pt") if p.is_file()]
    return max(trained, key=lambda p: p.stat().st_mtime) if trained else Path(base_weights)
