"""Measure the scanner CNN's accuracy and retake stability, with and without mirror averaging.

Evaluate the V3 held-out split:

    .venv-api/Scripts/python.exe scripts/evaluate_fabric_scans.py ^
        --raw C:/data/FabriCare_Dataset_V3/raw ^
        --manifest C:/data/FabriCare_Dataset_V3/reports/split_manifest_70_20_10.csv

Evaluate your own phone photos, sorted into one folder per fabric
(cotton, denim, linen, polyester, silk, wool):

    .venv-api/Scripts/python.exe scripts/evaluate_fabric_scans.py --folder C:/data/phone_scans

Each photo goes through the scanner's own preprocessing. "Retakes" are simulated
re-captures of the same photo (a slightly different crop, light level, tilt, and
JPEG compression). Consistency is the share of retakes that keep the original
photo's label, which tracks the "same garment, different result" problem.
Simulated retakes do not replace real phone scans of independent garments.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_runtime import SIX_FABRIC_LABELS, FabricMobileNetV2, prepare_fabric_rgb  # noqa: E402

LABELS = list(SIX_FABRIC_LABELS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="V3 split manifest CSV (needs --raw)")
    source.add_argument("--folder", type=Path, help="Folder with one subfolder of photos per fabric")
    parser.add_argument("--raw", type=Path, help="V3 raw folder with the six fabric subfolders")
    parser.add_argument("--split", default="test", help="Manifest split to evaluate (default: test)")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model" / "mobilenetv2_6class")
    parser.add_argument("--retakes", type=int, default=4, help="Simulated retakes per photo (0 to skip)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="Evaluate at most this many photos")
    parser.add_argument("--output", type=Path, help="Write the full JSON report here")
    args = parser.parse_args()
    if args.manifest and not args.raw:
        parser.error("--manifest needs --raw")
    return args


def manifest_photos(manifest, raw, split):
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    photos = []
    for row in rows:
        name = row["filename"]
        if not name or Path(name).name != name or row["fabric"] not in LABELS:
            raise ValueError(f"Invalid manifest row: {row}")
        path = raw / row["fabric"] / name
        if not path.is_file():
            path = path.with_name(path.name.replace(" (1)", ""))
        content = path.read_bytes()
        if row.get("sha256") and hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Photo differs from the manifest: {path}")
        photos.append((str(path), row["fabric"], content))
    return photos


def folder_photos(folder):
    photos = []
    for label in LABELS:
        directory = folder / label
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                photos.append((str(path), label, path.read_bytes()))
    if not photos:
        raise ValueError(f"No photos found in {folder}/<fabric>/ for fabrics {', '.join(LABELS)}")
    return photos


def encoded(image, quality=None):
    output = io.BytesIO()
    if quality is None:
        image.save(output, format="PNG")
    else:
        image.save(output, format="JPEG", quality=quality)
    output.seek(0)
    return output


def oriented(content):
    with Image.open(io.BytesIO(content)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def retake(image, rng):
    """One simulated re-capture: new crop, light, tilt, and compression."""
    width, height = image.size
    area = rng.uniform(0.80, 1.0)
    aspect = rng.uniform(0.9, 1.1)
    crop_w = min(width, max(8, round(width * np.sqrt(area * aspect))))
    crop_h = min(height, max(8, round(height * np.sqrt(area / aspect))))
    left = int(rng.integers(0, width - crop_w + 1))
    top = int(rng.integers(0, height - crop_h + 1))
    image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.rotate(rng.uniform(-6, 6), resample=Image.Resampling.BILINEAR, fillcolor=(127, 127, 127))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    return encoded(image, quality=int(rng.integers(70, 92)))


def model_inputs(source):
    pixels = np.asarray(prepare_fabric_rgb(source), dtype=np.float32)
    return np.stack([pixels, pixels[:, ::-1, :]])


def scores_for(model, sources):
    """Return (single-view, mirror-averaged) probabilities for each source."""
    single, mirrored = [], []
    for start in range(0, len(sources), 16):
        batch = np.concatenate([model_inputs(source) for source in sources[start:start + 16]])
        output = np.asarray(model(batch, training=False), dtype=np.float32).reshape(-1, 2, len(LABELS))
        single.append(output[:, 0])
        average = output.mean(axis=1)
        mirrored.append(average / average.sum(axis=1, keepdims=True))
    return np.concatenate(single), np.concatenate(mirrored)


def summary(truth, scores):
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    guess = scores.argmax(axis=1)
    report = classification_report(truth, guess, labels=np.arange(len(LABELS)), target_names=LABELS,
                                   zero_division=0, output_dict=True)
    top = np.sort(scores, axis=1)
    return {
        "accuracy": float(accuracy_score(truth, guess)),
        "macro_f1": float(f1_score(truth, guess, labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
        "recall": {label: float(report[label]["recall"]) for label in LABELS},
        "mean_top_score": float(top[:, -1].mean()),
        "mean_top_two_margin": float((top[:, -1] - top[:, -2]).mean()),
    }


def main():
    args = arguments()
    photos = (manifest_photos(args.manifest, args.raw, args.split) if args.manifest
              else folder_photos(args.folder))
    if args.limit:
        photos = photos[:args.limit]
    truth = np.asarray([LABELS.index(label) for _, label, _ in photos])
    classifier = FabricMobileNetV2(args.model_dir)
    model = classifier.model
    if list(classifier.labels) != LABELS:
        # Model output order comes from class_names.json; map it to LABELS.
        order = [list(classifier.labels).index(label) for label in LABELS]
        model = (lambda network: lambda batch, training: np.asarray(network(batch, training=training))[:, order])(
            classifier.model)

    clean_single, clean_mirror = scores_for(model, [encoded(oriented(content)) for _, _, content in photos])
    report = {
        "photos": len(photos),
        "source": str(args.manifest or args.folder),
        "split": args.split if args.manifest else None,
        "model": classifier.name,
        "clean": {"single_view": summary(truth, clean_single), "mirror_average": summary(truth, clean_mirror)},
    }

    if args.retakes > 0:
        rng = np.random.default_rng(args.seed)
        retake_truth, retake_owner, sources = [], [], []
        for index, (_, label, content) in enumerate(photos):
            image = oriented(content)
            for _ in range(args.retakes):
                sources.append(retake(image, rng))
                retake_truth.append(LABELS.index(label))
                retake_owner.append(index)
        retake_single, retake_mirror = scores_for(model, sources)
        owner = np.asarray(retake_owner)
        report["retakes"] = {"per_photo": args.retakes}
        for name, clean, retaken in (("single_view", clean_single, retake_single),
                                     ("mirror_average", clean_mirror, retake_mirror)):
            same = retaken.argmax(axis=1) == clean.argmax(axis=1)[owner]
            per_photo_stable = np.asarray([same[owner == index].all() for index in range(len(photos))])
            report["retakes"][name] = {
                **summary(np.asarray(retake_truth), retaken),
                "consistency": float(same.mean()),
                "photos_with_no_label_change": float(per_photo_stable.mean()),
            }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{report['photos']} photos - {report['model']}")
    for name in ("single_view", "mirror_average"):
        clean = report["clean"][name]
        line = f"{name:15s} accuracy {clean['accuracy']:.2%}  macro-F1 {clean['macro_f1']:.3f}"
        if "retakes" in report:
            retaken = report["retakes"][name]
            line += (f"  | retake accuracy {retaken['accuracy']:.2%}"
                     f"  consistency {retaken['consistency']:.2%}"
                     f"  photos never flipping {retaken['photos_with_no_label_change']:.2%}")
        print(line)
    worst = min(report["clean"]["mirror_average"]["recall"].items(), key=lambda item: item[1])
    print(f"Lowest recall with mirror averaging: {worst[0]} {worst[1]:.2%}")


if __name__ == "__main__":
    main()
