"""Write docs/fabricare_v7_damage_detector_colab.ipynb from the cell sources below.

Run from the repository root: python scripts/build_v7_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "fabricare_v7_damage_detector_colab.ipynb"
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# FabriCare V7 · Retrain the damage detector on our own photos

The app's YOLOv8 detector (`model/best.pt`: hole, needlemark, pilling, stain) was trained mostly on factory
close-ups. On phone photos of whole garments it boxes floor beside sleeves, reads prints as stains and misses small
slits. This notebook fine-tunes it on our own labelled photos. Follow `docs/damage-annotation-guide.md`.

* **Part A · Pre-label.** Runs the current detector on our photos and writes `v7_prelabels.zip` to Drive for
  correcting in Roboflow. Run it once, before labelling.
* **Part B · Train and compare.** After the corrected Roboflow export is in
  `FabriCare_Damage_Dataset/Annotated_Dataset/`, fine-tunes the detector and tests the old and new detector on the
  same held-out garments: precision, recall, mAP@0.5, mAP@0.5:0.95 per class.

**Run:** GPU runtime (L4 or T4) → *Runtime → Run all*. Part B stops with a message until the labels exist.
""")

code(r"""
# 1. Libraries (the API's Ultralytics version, so the exported detector loads unchanged).
import importlib.metadata as md, subprocess, sys
try:
    installed = md.version("ultralytics")
except md.PackageNotFoundError:
    installed = None
if installed != "8.4.104":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.104"])
import torch
print("ultralytics", md.version("ultralytics"), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

code(r"""
# 2. Drive, code and paths.
import csv, json, os, random, shutil, time, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from google.colab import drive

drive.mount("/content/drive")
REPO = Path("/content/fabricare")
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "training/v7",
                           "https://github.com/markkennethkenet/fabricare-inc.git", str(REPO)])
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import garment_mask
import v7_detector_data as v7
from ultralytics import YOLO

DATASET = Path("/content/drive/MyDrive/FabriCare_Damage_Dataset")
PUBLIC_ZIP = Path("/content/drive/MyDrive/FabriCare_Damage_YOLOv8_6Class.zip")
RUN = Path("/content/drive/MyDrive/Fabricare_training_v7/run_v7")
BASE_WEIGHTS = REPO / "model" / "best.pt"
RUN.mkdir(parents=True, exist_ok=True)
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
base = YOLO(str(BASE_WEIGHTS))
assert tuple(base.names[i] for i in range(len(base.names))) == v7.APP_CLASSES, base.names
print("Base detector classes:", base.names)
""")

md(r"""
## Part A · Pre-label our photos for Roboflow
""")

code(r"""
# 3. Run the current detector on Own_Collected_Dataset and Phone_Scans; keep boxes on the garment; zip for Roboflow.
PRELABEL_ZIP = DATASET / "v7_prelabels.zip"

def photos():
    for garment in sorted(p for p in (DATASET / "Own_Collected_Dataset").glob("*") if p.is_dir()):
        for folder in sorted(p for p in garment.iterdir() if p.is_dir()):
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in SUFFIXES:
                    yield garment.name, folder.name, path, False      # close-ups: no outline filter
    for garment in sorted(p for p in (DATASET / "Phone_Scans").glob("*") if p.is_dir()):
        for path in sorted(garment.rglob("*")):
            if path.suffix.lower() in SUFFIXES:
                yield garment.name, "phone", path, True               # whole-garment photos

if PRELABEL_ZIP.exists():
    print("Pre-labels already made:", PRELABEL_ZIP, "(delete it to rebuild)")
else:
    out = Path("/content/v7_prelabels"); shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True); (out / "labels").mkdir()
    counts, boxes = Counter(), Counter()
    for garment, view, path, whole in photos():
        try:
            with Image.open(path) as raw:
                rgb = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as error:
            print("skip unreadable:", path, error); continue
        rgb.thumbnail((1920, 1920))
        name = v7.prelabel_name(garment, view, path.name)
        rgb.save(out / "images" / name, quality=92)
        image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        result = base.predict(image, imgsz=640, conf=0.25, verbose=False)[0]
        outline = garment_mask.silhouette(image) if whole else None
        lines = []
        h, w = image.shape[:2]
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bbox = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            if outline is not None and garment_mask.share_on_garment(outline, bbox) < 0.6:
                continue
            cls = int(b.cls)
            lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
            boxes[v7.APP_CLASSES[cls]] += 1
        (out / "labels" / (name.rsplit(".", 1)[0] + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        counts["whole" if whole else "close-up"] += 1
    (out / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(v7.APP_CLASSES)) +
                                   "nc: 4\ntrain: images\nval: images\n")
    shutil.make_archive(str(PRELABEL_ZIP.with_suffix("")), "zip", out)
    print("Photos:", dict(counts), "| pre-label boxes:", dict(boxes))
    print("Saved:", PRELABEL_ZIP, "→ unzip and upload the folder to Roboflow (see docs/damage-annotation-guide.md).")
""")

code(r"""
# 3b. Send the pre-labels straight from Drive to Roboflow (nothing is stored on your laptop).
#     First add your Roboflow API key in Colab's Secrets panel (key icon, left) as ROBOFLOW_API_KEY,
#     and turn on "Notebook access". Find the key in Roboflow: Settings → API Keys.
ROBOFLOW_WORKSPACE = ""                    # the <workspace> part of app.roboflow.com/<workspace>; "" = your default
ROBOFLOW_PROJECT = "fabricare-damage-v7"   # project id: lowercase letters, digits and dashes
SEND_TO_ROBOFLOW = True

def roboflow_workspace():
    try:
        import roboflow  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "roboflow==1.7.3"])
    from google.colab import userdata
    try:
        key = userdata.get("ROBOFLOW_API_KEY")
    except Exception:
        raise SystemExit("Add your Roboflow API key in Colab's Secrets panel as ROBOFLOW_API_KEY "
                         "and allow this notebook to read it, then rerun this cell.")
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    return rf.workspace(ROBOFLOW_WORKSPACE) if ROBOFLOW_WORKSPACE else rf.workspace()

SENT = DATASET / "v7_prelabels_sent_to_roboflow.txt"
if not SEND_TO_ROBOFLOW:
    print("Skipped: upload v7_prelabels.zip to Roboflow yourself.")
elif SENT.exists():
    print("Already sent to Roboflow:", SENT.read_text().strip())
else:
    folder = Path("/content/v7_prelabels")
    if not (folder / "images").is_dir():
        shutil.rmtree(folder, ignore_errors=True)
        zipfile.ZipFile(PRELABEL_ZIP).extractall(folder)
    count = len(list((folder / "images").iterdir()))
    workspace = roboflow_workspace()
    workspace.upload_dataset(str(folder), ROBOFLOW_PROJECT, num_workers=8, project_type="object-detection",
                             batch_name="v7 pre-labels", num_retries=2)
    link = f"https://app.roboflow.com/{workspace.url}/{ROBOFLOW_PROJECT}"
    SENT.write_text(f"{count} photos to {link} on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"Sent {count} photos with their pre-label boxes.")
    print(f"Correct them at {link}/annotate (see docs/damage-annotation-guide.md).")
""")

md(r"""
## Part B · Train on the corrected labels and compare with the current detector
""")

code(r"""
# 3c. Fetch the corrected export from Roboflow into Drive (Annotated_Dataset/). Set the version number you
#     generated in Roboflow (Generate → Auto-Orient only, no augmentation). Leave None if you put a zip there yourself.
ROBOFLOW_VERSION = None
if ROBOFLOW_VERSION:
    target = DATASET / "Annotated_Dataset" / f"roboflow_{ROBOFLOW_PROJECT}_v{int(ROBOFLOW_VERSION)}"
    if (target / "data.yaml").exists():
        print("Already fetched:", target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        roboflow_workspace().project(ROBOFLOW_PROJECT).version(int(ROBOFLOW_VERSION)).download(
            "yolov8", location=str(target), overwrite=True)
        print("Fetched version", ROBOFLOW_VERSION, "into", target)
""")

code(r"""
# 4. Read the Roboflow YOLOv8 export, map classes, and split by garment (70/15/15).
annotated = DATASET / "Annotated_Dataset"
exports = sorted(annotated.glob("*.zip")) + sorted(p for p in annotated.glob("*") if p.is_dir()) if annotated.exists() else []
if not exports:
    raise SystemExit("Part B needs the corrected Roboflow export (YOLOv8 format) in "
                     f"{annotated}. Label the photos first: docs/damage-annotation-guide.md.")
src = Path("/content/v7_export"); shutil.rmtree(src, ignore_errors=True); src.mkdir()
for export in exports:
    target = src / export.stem
    if export.suffix == ".zip":
        zipfile.ZipFile(export).extractall(target)
    else:
        shutil.copytree(export, target)

own = []   # (image path, label text)
for yaml_path in src.rglob("data.yaml"):
    names = []
    text = yaml_path.read_text()
    block = text.split("names:", 1)[1]
    if block.lstrip().startswith("["):
        names = [n.strip().strip("'\"") for n in block.split("]", 1)[0].lstrip()[1:].split(",")]
    else:
        names = [line.split(":", 1)[1].strip().strip("'\"") for line in block.splitlines()
                 if ":" in line and line.strip().split(":", 1)[0].strip().isdigit()]
    mapping = v7.roboflow_mapping(names)
    print(yaml_path.parent.name, "classes", names, "→ app ids", mapping)
    for image in yaml_path.parent.rglob("*"):
        if image.suffix.lower() in SUFFIXES and image.parent.name == "images":
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            text = label.read_text() if label.exists() else ""
            own.append((image, v7.remap_label_lines(text, mapping)))
assert own, "No labelled images found in the export."
split = v7.split_by_garment([p.name for p, _ in own])

DS = Path("/content/v7ds"); shutil.rmtree(DS, ignore_errors=True)
stats = defaultdict(Counter)
for image, label in own:
    s = split[image.name]
    for kind in ("images", "labels"):
        (DS / "own" / s / kind).mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, DS / "own" / s / "images" / image.name)
    (DS / "own" / s / "labels" / (image.stem + ".txt")).write_text(label)
    stats[s]["images"] += 1
    stats[s]["background"] += not label.strip()
    for line in label.splitlines():
        stats[s][v7.APP_CLASSES[int(line.split()[0])]] += 1
for s in v7.SPLITS:
    print(s, dict(stats[s]), "| garments:", len({v7.garment_of(p.name) for p, _ in own if split[p.name] == s}))

train_dirs, public_test = [str(DS / "own" / "train" / "images")], None
if PUBLIC_ZIP.exists():
    pub = Path("/content/public"); shutil.rmtree(pub, ignore_errors=True)
    zipfile.ZipFile(PUBLIC_ZIP).extractall(pub)
    root = next(pub.rglob("data.yaml")).parent
    for s in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            (DS / "public" / s / kind).mkdir(parents=True, exist_ok=True)
        for image in (root / s / "images").iterdir():
            label = root / s / "labels" / (image.stem + ".txt")
            shutil.copy2(image, DS / "public" / s / "images" / image.name)
            (DS / "public" / s / "labels" / (image.stem + ".txt")).write_text(
                v7.remap_label_lines(label.read_text() if label.exists() else "", v7.PUBLIC_TO_APP))
    train_dirs += [str(DS / "public" / "train" / "images"), str(DS / "public" / "valid" / "images")]
    public_test = DS / "public" / "test" / "images"
    print("Public dataset added for training; its test split is reported separately.")
else:
    print("Public dataset not on Drive; training on our photos only.")

def yaml_for(name, train, val, test):
    path = DS / f"{name}.yaml"
    path.write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(v7.APP_CLASSES)) +
                    f"train: {json.dumps(train)}\nval: {json.dumps(str(val))}\ntest: {json.dumps(str(test))}\n")
    return path

DATA = yaml_for("v7", train_dirs, DS / "own" / "valid" / "images", DS / "own" / "test" / "images")
PUBLIC_DATA = yaml_for("public", train_dirs, public_test, public_test) if public_test else None
""")

code(r"""
# 5. Test helper and the current detector's baseline on the same held-out garments.
def evaluate(weights, data, tag):
    model = YOLO(str(weights))
    m = model.val(data=str(data), split="test", imgsz=960, conf=0.001, batch=8, plots=False, verbose=False,
                  project=str(RUN / "eval"), name=tag, exist_ok=True)
    by_class = {int(c): i for i, c in enumerate(m.ap_class_index)}
    rows = {}
    for cid, name in enumerate(v7.APP_CLASSES):
        if cid in by_class:
            p, r, ap50, ap = m.class_result(by_class[cid])
            rows[name] = {"precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50_95": float(ap),
                          "boxes": int(m.nt_per_class[cid])}
    overall = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
               "mAP50_95": float(m.box.map)}
    return {"overall": overall, "per_class": rows}

baseline = {"own_test": evaluate(BASE_WEIGHTS, DATA, "baseline_own")}
if PUBLIC_DATA:
    baseline["public_test"] = evaluate(BASE_WEIGHTS, PUBLIC_DATA, "baseline_public")
print("Current detector on our test garments:", json.dumps(baseline["own_test"]["overall"], indent=1))
""")

code(r"""
# 6. Fine-tune from the current detector (resumes after a disconnect). 960 px helps small holes and slits.
last = RUN / "train" / "weights" / "last.pt"
best = RUN / "train" / "weights" / "best.pt"
if best.exists() and (RUN / "train_done.txt").exists():
    print("Already trained:", best)
else:
    model = YOLO(str(last)) if last.exists() else YOLO(str(BASE_WEIGHTS))
    model.train(data=str(DATA), epochs=80, patience=20, imgsz=960, batch=16, seed=42, deterministic=True,
                project=str(RUN), name="train", exist_ok=True, resume=last.exists(), plots=True,
                close_mosaic=10, degrees=10, fliplr=0.5, flipud=0.2, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4)
    (RUN / "train_done.txt").write_text(datetime.now(timezone.utc).isoformat())
""")

code(r"""
# 7. Compare old and new detectors on the same test garments, then export.
new = {"own_test": evaluate(best, DATA, "v7_own")}
if PUBLIC_DATA:
    new["public_test"] = evaluate(best, PUBLIC_DATA, "v7_public")

print(f"{'test set':12s} {'model':10s} {'precision':>9s} {'recall':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
for key in new:
    for tag, result in (("current", baseline[key]), ("V7", new[key])):
        o = result["overall"]
        print(f"{key:12s} {tag:10s} {o['precision']:9.3f} {o['recall']:7.3f} {o['mAP50']:7.3f} {o['mAP50_95']:9.3f}")
print("\nPer class on our test garments (current → V7, mAP50):")
for name in v7.APP_CLASSES:
    a, b = baseline["own_test"]["per_class"].get(name), new["own_test"]["per_class"].get(name)
    if a or b:
        print(f"  {name:10s} {a['mAP50'] if a else float('nan'):.3f} → {b['mAP50'] if b else float('nan'):.3f}"
              f"  ({(b or a)['boxes']} test boxes)")

export = RUN / "export"; shutil.rmtree(export, ignore_errors=True); export.mkdir()
shutil.copy2(best, export / "best.pt")
report = {"created": datetime.now(timezone.utc).isoformat(), "classes": list(v7.APP_CLASSES),
          "split_counts": {s: dict(stats[s]) for s in v7.SPLITS}, "public_data": bool(PUBLIC_DATA),
          "baseline": baseline, "v7": new, "train_image_size": 960,
          "better_on_own_test": new["own_test"]["overall"]["mAP50"] > baseline["own_test"]["overall"]["mAP50"]}
(export / "v7_detector_report.json").write_text(json.dumps(report, indent=2))
archive = shutil.make_archive(str(RUN / "fabricare_v7_detector"), "zip", export)
print("\nBetter than the current detector on our photos:", report["better_on_own_test"])
print("Saved:", archive)
""")


def main():
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in CELLS]}
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT), "with", len(CELLS), "cells")


if __name__ == "__main__":
    main()
