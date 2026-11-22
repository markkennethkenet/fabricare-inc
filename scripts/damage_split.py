"""Split the folder-labelled FabriCare damage photos for the V6 damage-condition classifier.

Drive layout (FabriCare_Damage_Dataset):
  Online_Dataset/<Class>/*.jpg                    public photos, no garment ids
  Own_Collected_Dataset/<Tshirt_NNN>/<Class>/*.jpg  team photos of six T-shirts

Own photos are split by T-shirt, so no garment is seen in both training and testing. Online photos are
grouped by file name (copies and augmented versions of one source share a group) and each group goes
wholly to one split, aiming for 70/20/10 per class. "Multiple_Damage" photos do not say which damages they
show, so they are kept apart as a separate check set. Nothing here touches Drive.
"""
import hashlib
import re
from collections import Counter, defaultdict

SPLITS = ("train", "valid", "test")
TARGET = {"train": 0.7, "valid": 0.2, "test": 0.1}
DAMAGE_CLASSES = ("hole", "needlemark", "normal", "pilling", "stain", "wrinkle")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
MULTIPLE = "multiple"

_FOLDER_CLASS = {
    "holes": "hole", "hole": "hole",
    "needle_marks": "needlemark", "needle_mark": "needlemark", "needlemarks": "needlemark",
    "normal": "normal", "normal_fabric": "normal",
    "pilling": "pilling",
    "stains": "stain", "stain": "stain",
    "wrinkles": "wrinkle", "wrinkle": "wrinkle",
    "multiple_damage": MULTIPLE,
}
# Own T-shirts: the largest goes to training; the others are held out whole.
OWN_GARMENT_SPLIT = {"tshirt_001": "train", "tshirt_002": "train", "tshirt_003": "valid",
                     "tshirt_004": "valid", "tshirt_005": "test", "tshirt_006": "test"}

_COPY = re.compile(r"\s*\(\d+\)$")
_ROBOFLOW = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-f]+$")
_AUGMENT = re.compile(r"[_-](?:aug|flip|rot|copy)\w*$")


def folder_class(name):
    """Class for a Drive folder name such as 'Holes ', ' Stains' or 'Normal_Fabric'; None if unknown."""
    return _FOLDER_CLASS.get(re.sub(r"\s+", "_", name.strip()).casefold())


def online_group(label, filename):
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = _AUGMENT.sub("", _ROBOFLOW.sub("", _COPY.sub("", stem.strip()))).casefold()
    return f"online:{label}:{stem}"


def _order(group):
    return hashlib.sha256(group.encode("utf-8")).hexdigest()


def assign_groups(sizes):
    """Place whole groups so the photos land near 70/20/10 (deterministic hash order)."""
    placement, filled = {}, Counter()
    for group in sorted(sizes, key=_order):
        total = sum(filled.values()) + sizes[group]
        split = max(SPLITS, key=lambda s: (TARGET[s] * total - filled[s], -SPLITS.index(s)))
        placement[group] = split
        filled[split] += sizes[group]
    return placement


def build_split(scanned):
    """Return (records, report).

    scanned: dicts with source ("online" | "own"), garment (own only), folder (class folder name),
    filename, path and sha256.
    """
    report = Counter()
    labelled = []
    for item in scanned:
        label = folder_class(item["folder"])
        if label is None:
            report[f"unknown_folder:{item['folder'].strip()}"] += 1
            continue
        labelled.append({**item, "label": label})

    by_sha = defaultdict(list)
    for item in labelled:
        by_sha[item["sha256"]].append(item)
    unique = []
    for sha, items in sorted(by_sha.items()):
        if len({i["label"] for i in items}) > 1:
            report["excluded_conflicting_labels"] += len(items)
            continue
        if len(items) > 1:
            report["duplicate_copy_ignored"] += len(items) - 1
        unique.append(min(items, key=lambda i: (i["source"] != "own", i["path"])))

    records = []
    online_sizes = defaultdict(Counter)
    for item in unique:
        if item["label"] == MULTIPLE:
            records.append({**item, "split": MULTIPLE, "group_id": f"own:{item.get('garment', 'online')}"})
            continue
        if item["source"] == "own":
            garment = item["garment"].strip().casefold()
            split = OWN_GARMENT_SPLIT.get(garment, "test")
            records.append({**item, "split": split, "group_id": f"own:{garment}"})
        else:
            group = online_group(item["label"], item["filename"])
            online_sizes[item["label"]][group] += 1
            records.append({**item, "split": None, "group_id": group})

    placement = {}
    for sizes in online_sizes.values():
        placement.update(assign_groups(sizes))
    for record in records:
        if record["split"] is None:
            record["split"] = placement[record["group_id"]]

    groups = defaultdict(set)
    for r in records:
        groups[r["group_id"]].add(r["split"])
    crossing = [g for g, s in groups.items() if len(s) > 1]
    if crossing:
        raise ValueError(f"Groups cross splits: {crossing[:5]}")
    for r in records:
        report[f"{r['split']}:{r['label']}"] += 1
    return records, dict(sorted(report.items()))
