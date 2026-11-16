#!/usr/bin/env python3
"""Assemble labeled fabric image folders into a provenance-tracked six-class set.

The script copies source images into train/valid/test folders, hashes files to
remove byte-identical duplicates, and writes a CSV manifest and class counts.
It never edits or deletes source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASSES = ("cotton", "polyester", "denim", "linen", "wool", "silk")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {
    "train": "train", "training": "train",
    "valid": "valid", "validation": "valid", "val": "valid",
    "test": "test", "testing": "test",
}
LABEL_ALIASES = {
    "cotton": "cotton",
    "polyester": "polyester",
    "denim": "denim",
    "linen": "linen",
    "wool": "wool",
    "silk": "silk",
}


@dataclass(frozen=True)
class ImageRecord:
    source_name: str
    source_root: Path
    source_path: Path
    source_label: str
    label: str
    source_split: str | None
    sha256: str


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


NORMALIZED_LABELS = {normalize_name(name): name for name in LABEL_ALIASES}


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for each --source.")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both source name and path are required.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Source directory does not exist: {path}")
    return name.strip(), path


def image_label_and_split(path: Path, root: Path) -> tuple[str, str, str | None] | None:
    parents = list(path.parents)
    try:
        root_index = parents.index(root)
    except ValueError:
        return None
    relative_parents = parents[:root_index]
    label_info = None
    split = None
    for parent in relative_parents:
        normalized = normalize_name(parent.name)
        if label_info is None and normalized in NORMALIZED_LABELS:
            label_info = (parent.name, NORMALIZED_LABELS[normalized])
        if split is None and parent.name.casefold() in SPLIT_NAMES:
            split = SPLIT_NAMES[parent.name.casefold()]
    if label_info is None:
        return None
    return label_info[0], label_info[1], split


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_sources(sources: list[tuple[str, Path]]) -> tuple[list[ImageRecord], Counter, list[str]]:
    records: list[ImageRecord] = []
    ignored: Counter = Counter()
    errors: list[str] = []
    for source_name, root in sources:
        found = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            found += 1
            info = image_label_and_split(path, root)
            if info is None:
                ignored[f"{source_name}:unrecognized_label_folder"] += 1
                continue
            source_label, label, source_split = info
            try:
                digest = digest_file(path)
            except OSError as exc:
                errors.append(f"Could not read {path}: {exc}")
                continue
            records.append(ImageRecord(
                source_name, root, path, source_label, label, source_split, digest
            ))
        if found == 0:
            errors.append(f"No supported image files found under {root}")
    return records, ignored, errors


def deduplicate(records: list[ImageRecord]) -> tuple[list[ImageRecord], Counter, list[str]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    duplicates: Counter = Counter()
    errors: list[str] = []
    for record in records:
        grouped[record.sha256].append(record)

    kept = []
    split_priority = {"test": 3, "valid": 2, "train": 1, None: 0}
    for same_image in grouped.values():
        labels = {record.label for record in same_image}
        if len(labels) > 1:
            errors.append(
                "Same file bytes have conflicting labels: "
                + " vs ".join(f"{record.source_path} ({record.label})" for record in same_image)
            )
            continue
        chosen = max(same_image, key=lambda record: split_priority[record.source_split])
        kept.append(chosen)
        duplicates[chosen.label] += len(same_image) - 1
    return kept, duplicates, errors


def split_records(records: list[ImageRecord], seed: int) -> dict[str, str]:
    """Keep source validation/test holdouts and split source training images by class."""
    rng = random.Random(seed)
    split_by_hash: dict[str, str] = {}
    by_label: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.source_split in {"valid", "test"}:
            prior = split_by_hash.get(record.sha256)
            if prior is None or (prior == "valid" and record.source_split == "test"):
                split_by_hash[record.sha256] = record.source_split
        else:
            by_label[record.label].append(record)

    for label in CLASSES:
        candidates = [record for record in by_label[label] if record.sha256 not in split_by_hash]
        rng.shuffle(candidates)
        n = len(candidates)
        if n:
            test_n = max(1, round(n * 0.10)) if n >= 10 else (1 if n >= 3 else 0)
            valid_n = max(1, round(n * 0.10)) if n >= 10 else (1 if n >= 3 else 0)
            for index, record in enumerate(candidates):
                if index < test_n:
                    split_by_hash[record.sha256] = "test"
                elif index < test_n + valid_n:
                    split_by_hash[record.sha256] = "valid"
                else:
                    split_by_hash[record.sha256] = "train"
    return split_by_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, type=parse_source,
        metavar="NAME=PATH", help="Source dataset root; repeat for each source.",
    )
    parser.add_argument("--output", required=True, type=Path, help="New output directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records, ignored, errors = scan_sources(args.source)
    records, duplicates, duplicate_errors = deduplicate(records)
    errors.extend(duplicate_errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if not records:
        print("No images with one of the six supported class-folder labels were found.", file=sys.stderr)
        return 2

    counts: dict[str, Counter] = {split: Counter() for split in ("train", "valid", "test")}
    split_by_hash = split_records(records, args.seed)
    manifest_rows = []
    for record in records:
        split = split_by_hash.get(record.sha256, "train")
        relative_output = Path(split) / record.label / f"{record.sha256[:16]}_{record.source_path.name}"
        target = output / relative_output
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.source_path, target)
        counts[split][record.label] += 1
        manifest_rows.append({
            "output_path": relative_output.as_posix(),
            "label": record.label,
            "source_name": record.source_name,
            "source_path": str(record.source_path),
            "source_label": record.source_label,
            "source_split": record.source_split or "",
            "split": split,
            "sha256": record.sha256,
        })

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output / "labels.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    summary = {
        "classes": list(CLASSES),
        "seed": args.seed,
        "unique_images": len(records),
        "exact_duplicate_images_removed": dict(duplicates),
        "ignored_images_by_source": dict(ignored),
        "counts_by_split": {
            split: {label: counts[split][label] for label in CLASSES}
            for split in ("train", "valid", "test")
        },
        "sources": [{"name": name, "path": str(path)} for name, path in args.source],
        "limitations": [
            "Only byte-identical duplicate images are removed automatically.",
            "Review source labels and group photos of the same garment before final evaluation.",
            "Mixed-fiber images are outside this six-class single-label set.",
        ],
    }
    (output / "counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Built {len(records)} unique images at {output}")
    for split in ("train", "valid", "test"):
        print(f"{split}: " + ", ".join(f"{label}={counts[split][label]}" for label in CLASSES))
    missing = [label for label in CLASSES if counts["train"][label] == 0]
    untested = [label for label in CLASSES if counts["test"][label] == 0]
    if missing or untested:
        print(f"WARNING: missing training classes={missing}; untested classes={untested}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
