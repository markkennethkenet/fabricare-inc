"""Build the V5 fabric split: the audited V3 split, extended with photos added to Drive since.

Photos already in the V3 manifest keep their split, so V5 test results stay comparable with V4 and no
earlier training photo leaks into validation or test. New photos are grouped by garment (same rule as V3)
and each new group goes wholly to one split, aiming for 70/20/10 photos per fabric.
Nothing here touches Drive; the notebook passes in what it scanned.
"""
import hashlib
import re
from collections import Counter, defaultdict

SPLITS = ("train", "valid", "test")
TARGET = {"train": 0.7, "valid": 0.2, "test": 0.1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

_GARMENT = re.compile(r"^(?P<fabric>[a-z]+)_g(?P<number>\d+)_\d+$")
_CAMERA = re.compile(r"^(?:img|pxl|vid)_(?P<stamp>\d{8}_\d{6})")
_COPY = re.compile(r"\s*\(\d+\)$")
_ROBOFLOW = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-f]+$")


def group_id(fabric, filename):
    """Garment group of a photo; photos of one garment must stay in one split."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = _ROBOFLOW.sub("", _COPY.sub("", stem.strip())).lower()
    garment = _GARMENT.match(stem)
    if garment and garment["fabric"] == fabric:
        return f"{fabric}:garment_{int(garment['number']):04d}"
    camera = _CAMERA.match(stem)
    if camera:
        # Burst frames and "_1" copies share the camera timestamp.
        return f"{fabric}:img_{camera['stamp']}"
    return f"{fabric}:{stem}"


def _order(group):
    return hashlib.sha256(group.encode("utf-8")).hexdigest()


def assign_new_groups(sizes_by_fabric):
    """Place whole groups so each fabric's new photos land near 70/20/10.

    sizes_by_fabric: {fabric: {group_id: photo_count}}. Groups are visited in a hash order (deterministic,
    independent of upload order) and each goes to the split furthest below its target share.
    Targets apply to the new photos only, so the V3 split keeps its own proportions.
    """
    placement = {}
    for fabric, sizes in sizes_by_fabric.items():
        filled = Counter()
        for group in sorted(sizes, key=_order):
            total = sum(filled.values()) + sizes[group]
            split = max(SPLITS, key=lambda s: (TARGET[s] * total - filled[s], -SPLITS.index(s)))
            placement[group] = split
            filled[split] += sizes[group]
    return placement


def build_split(v3_rows, scanned, excluded_sha256=(), polyester_additions=()):
    """Return (records, report).

    v3_rows: rows of the audited V3 manifest (filename, fabric, sha256, group_id, split).
    scanned: dicts with fabric, filename, sha256 and path for every image now in Drive raw/<fabric>/.
    excluded_sha256: photos the V3 audit set aside (cross-class quarantine, duplicate copies).
    polyester_additions: verified polyester photos that V4 used for training only (sha256 values).
    """
    excluded = set(excluded_sha256)
    additions = set(polyester_additions)
    by_sha = defaultdict(list)
    for item in scanned:
        by_sha[item["sha256"]].append(item)

    report = Counter()
    conflicting = set()
    current = {}
    for sha, items in by_sha.items():
        fabrics = {i["fabric"] for i in items}
        if len(fabrics) > 1:
            conflicting.add(sha)  # the same photo filed under two fabrics: label unknown
            report["excluded_cross_class_copy"] += len(items)
            continue
        if len(items) > 1:
            report["duplicate_copy_ignored"] += len(items) - 1
        current[sha] = min(items, key=lambda i: i["filename"])

    records, seen = [], set()
    for row in v3_rows:
        sha = row["sha256"]
        photo = current.get(sha)
        if photo is None:
            report["v3_photo_missing_from_drive"] += 1
            continue
        if photo["fabric"] != row["fabric"]:
            report["v3_photo_moved_to_other_fabric"] += 1
        records.append({**photo, "split": row["split"], "group_id": row["group_id"], "source": "v3"})
        seen.add(sha)

    new_sizes = defaultdict(Counter)
    pending = []
    for sha, photo in sorted(current.items()):
        if sha in seen:
            continue
        if sha in excluded:
            report["excluded_by_v3_audit"] += 1
            continue
        if sha in additions and photo["fabric"] == "polyester":
            records.append({**photo, "split": "train", "group_id": "polyester_addition:" + sha[:16],
                            "source": "polyester_addition"})
            seen.add(sha)
            continue
        group = group_id(photo["fabric"], photo["filename"])
        new_sizes[photo["fabric"]][group] += 1
        pending.append((photo, group))

    # A new photo can share a garment group with V3 photos (another burst frame): follow their split.
    v3_group_split = {r["group_id"]: r["split"] for r in records}
    free = {f: {g: n for g, n in sizes.items() if g not in v3_group_split} for f, sizes in new_sizes.items()}
    placement = {**assign_new_groups(free), **v3_group_split}
    for photo, group in pending:
        records.append({**photo, "split": placement[group], "group_id": group, "source": "new"})
        report[f"new_{placement[group]}"] += 1

    groups = defaultdict(set)
    for r in records:
        groups[r["group_id"]].add(r["split"])
    crossing = [g for g, s in groups.items() if len(s) > 1]
    if crossing:
        raise ValueError(f"Garment groups cross splits: {crossing[:5]}")
    report["new_photos"] = len(pending)
    return records, dict(report)
