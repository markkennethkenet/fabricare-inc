#!/usr/bin/env python3
"""Build FabriCare's YOLO detection dataset from four source ZIP archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import yaml
from PIL import Image


CLASSES = ("hole", "needlemark", "pilling", "stain", "missing_yarn", "wrinkle")
ALIASES = {name.replace("_", ""): name for name in CLASSES}
ALIASES.update(holes="hole", needlemarks="needlemark", stains="stain", spot="stain",
               wrinkles="wrinkle", defectfree=None, normal=None)
SPLITS = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Record:
    source: str
    archive: ZipFile
    member: str
    source_split: str
    split: str
    group: str
    boxes: tuple[tuple[int, float, float, float, float], ...]
    encoding: str
    dimensions: tuple[int, int] | None = None


def canonical_class(raw):
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    if key != "defectfree":
        key = key.removesuffix("defect")
    if key not in ALIASES:
        raise ValueError(f"Unsupported class {raw!r}")
    name = ALIASES[key]
    return None if name is None else CLASSES.index(name)


def photo_group(member):
    # Roboflow gives each version a different .rf. hash.
    return PurePosixPath(member).stem.lower().split(".rf.")[0]


def make_box(class_id, left, top, right, bottom):
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("Non-finite box coordinate")
    left, top, right, bottom = (
        min(1.0, max(0.0, value)) for value in (left, top, right, bottom)
    )
    if right - left < 1e-7 or bottom - top < 1e-7:
        raise ValueError("Empty box after clipping to image bounds")
    return (class_id, (left + right) / 2, (top + bottom) / 2,
            right - left, bottom - top)


def read_yolo(archive, source, directory=None):
    members = {name for name in archive.namelist()
               if directory is None or directory in name}
    configs = [yaml.safe_load(archive.read(n)) for n in sorted(members)
               if PurePosixPath(n).name == "data.yaml"]
    if not configs:
        raise ValueError(f"{source}: no data.yaml found")
    names = configs[0]["names"]
    if any(config["names"] != names for config in configs):
        raise ValueError(f"{source}: class lists disagree")
    names = names if isinstance(names, dict) else dict(enumerate(names))
    mapping = {int(index): canonical_class(name) for index, name in names.items()}
    records = []
    for member in sorted(members):
        path = PurePosixPath(member)
        if (path.suffix.lower() not in IMAGE_TYPES or len(path.parts) < 3
                or path.parts[-2] != "images" or path.parts[-3] not in SPLITS):
            continue
        split = SPLITS[path.parts[-3]]
        label = path.parent.parent / "labels" / path.with_suffix(".txt").name
        if str(label) not in members:
            raise ValueError(f"{source}: missing label for {member}")
        boxes = []
        encoding = "yolo_boxes"
        for line in archive.read(str(label)).decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            raw_id, *raw_coords = line.split()
            class_id = mapping[int(raw_id)]
            if class_id is None:
                continue  # A defect-free marker supplies no detection target.
            coords = list(map(float, raw_coords))
            if len(coords) == 4:
                x, y, w, h = coords
                coords = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs, ys = coords[::2], coords[1::2]
                coords = (min(xs), min(ys), max(xs), max(ys))
                encoding = "yolo_polygons_to_boxes"
            else:
                raise ValueError(f"{source}: unsupported label geometry in {label}")
            try:
                boxes.append(make_box(class_id, *coords))
            except ValueError as error:
                raise ValueError(f"{source}: {label}: {error}") from error
        records.append(Record(source, archive, member, split, split,
                              photo_group(member), tuple(boxes), encoding))
    if not records:
        raise ValueError(f"{source}: no YOLO images found")
    return records


def read_original_spots(archive):
    yolo_exports = {}
    for record in read_yolo(archive, "fabricspotdefect", "/Augmented/YOLOv8/"):
        yolo_exports.setdefault((record.split, record.group), record)
    records = []
    for member in sorted(archive.namelist()):
        path = PurePosixPath(member)
        if ("Orginal" not in path.parts or path.name != "_annotations.coco.json"
                or path.parent.name not in SPLITS):
            continue
        split = SPLITS[path.parent.name]
        content = json.loads(archive.read(member))
        categories = {category["id"]: category["name"] for category in content["categories"]}
        by_image = defaultdict(list)
        for annotation in content["annotations"]:
            # The release also declares an unused supercategory called Hole-EnBF.
            class_id = canonical_class(categories[annotation["category_id"]])
            by_image[annotation["image_id"]].append((class_id, annotation["bbox"]))
        for image in content["images"]:
            filename = str(path.parent / image["file_name"])
            width, height = image["width"], image["height"]
            group = photo_group(filename)
            with Image.open(io.BytesIO(archive.read(filename))) as photo:
                if photo.size != (width, height):
                    fallback = yolo_exports[(split, group)]
                    records.append(Record(fallback.source, archive, fallback.member, split,
                                          split, group, fallback.boxes, "spot_yolo_fallback"))
                    continue
            boxes = tuple(make_box(class_id, x / width, y / height,
                                   (x + w) / width, (y + h) / height)
                          for class_id, (x, y, w, h) in by_image[image["id"]])
            records.append(Record("fabricspotdefect", archive, filename, split, split,
                                  group, boxes, "coco_boxes", (width, height)))
    if not records:
        raise ValueError("No original FabricSpotDefect COCO images found")
    return records


def select_copies(records, seed):
    wrinkle_validation = sorted({r.group for r in records
                                 if r.source == "wrinkles" and r.split == "valid"})
    if len(wrinkle_validation) < 2:
        raise ValueError("Wrinkle validation needs at least two source photos")
    random.Random(seed).shuffle(wrinkle_validation)
    wrinkle_test = set(wrinkle_validation[:len(wrinkle_validation) // 2])
    groups = set()
    skipped = Counter()
    selected = []
    for record in sorted(records, key=lambda r: (SPLIT_PRIORITY[r.split], r.source, r.member)):
        key = (record.source, record.group)
        if key in groups:
            skipped[record.source] += 1
            continue
        groups.add(key)
        split = "test" if record.source == "wrinkles" and record.split == "valid" and record.group in wrinkle_test else record.split
        selected.append((split, record))
    return selected, skipped


def build(output, sources, seed):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True)
    records = read_original_spots(sources["fabricspotdefect"])
    for source in ("fabricare", "fabrid_v2", "wrinkles"):
        records.extend(read_yolo(sources[source], source))
    selected, skipped_copies = select_copies(records, seed)
    counts = defaultdict(Counter)
    source_counts = defaultdict(Counter)
    encodings = Counter()
    duplicates = Counter()
    seen = set()
    for split in SPLIT_PRIORITY:
        for kind in ("images", "labels"):
            (staging / split / kind).mkdir(parents=True)

    with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        manifest = csv.writer(stream)
        manifest.writerow(("image", "split", "source", "archive", "source_member",
                           "source_split", "photo_group", "pixel_sha256", "encoding", "box_count"))
        for split, record in sorted(selected, key=lambda item: (SPLIT_PRIORITY[item[0]], item[1].source, item[1].member)):
            data = record.archive.read(record.member)
            with Image.open(io.BytesIO(data)) as image:
                if record.dimensions is not None and image.size != record.dimensions:
                    raise ValueError(f"COCO dimensions disagree with image: {record.member}")
                pixels = image.convert("RGB")
                digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            if digest in seen:
                duplicates[record.source] += 1
                continue
            seen.add(digest)
            extension = PurePosixPath(record.member).suffix.lower()
            filename = f"{record.source}_{digest[:24]}{extension}"
            relative = f"{split}/images/{filename}"
            (staging / relative).write_bytes(data)
            lines = [" ".join([str(cls), *(f"{value:.8f}" for value in coords)])
                     for cls, *coords in record.boxes]
            labels = staging / split / "labels" / (Path(filename).stem + ".txt")
            labels.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split]["images"] += 1
            counts[split]["background_images"] += int(not lines)
            counts[split].update(CLASSES[box[0]] for box in record.boxes)
            source_counts[record.source][split] += 1
            encodings[record.encoding] += 1
            manifest.writerow((relative, split, record.source, Path(record.archive.filename).name,
                               record.member, record.source_split, record.group,
                               digest, record.encoding, len(lines)))

    if set(source_counts) != set(sources):
        raise ValueError("One or more requested sources contributed no images")
    for split in SPLIT_PRIORITY:
        if any(not counts[split][name] for name in CLASSES):
            raise ValueError(f"{split}: one or more classes have no boxes")
    (staging / "data.yaml").write_text(yaml.safe_dump({
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": dict(enumerate(CLASSES)),
    }, sort_keys=False), encoding="utf-8")
    summary = {
        "seed": seed, "classes": CLASSES,
        "splits": counts, "images_by_source_and_split": source_counts,
        "images_by_annotation_conversion": encodings,
        "source_photo_repeats_removed": skipped_copies,
        "exact_decoded_pixel_duplicates_removed": duplicates,
    }
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    guide = Path(__file__).resolve().parents[1] / "docs" / "damage-yolov8-dataset.md"
    shutil.copyfile(guide, staging / "README.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("fabrid", "spot", "wrinkles", "fabricare", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with (ZipFile(args.fabrid) as fabrid, ZipFile(args.spot) as spot,
          ZipFile(args.wrinkles) as wrinkles, ZipFile(args.fabricare) as fabricare):
        summary = build(args.output.resolve(), {
            "fabrid_v2": fabrid, "fabricspotdefect": spot,
            "wrinkles": wrinkles, "fabricare": fabricare,
        }, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
