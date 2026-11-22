"""Write docs/fabricare_labelling_export_colab.ipynb: pack the damage photos on Drive into zips for labelling.

Run from the repository root: python scripts/build_labelling_export_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "fabricare_labelling_export_colab.ipynb"
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# FabriCare · Pack the damage photos for manual labelling

Collects every photo in `My Drive/FabriCare_Damage_Dataset` (Own_Collected_Dataset, Phone_Scans, Online_Dataset),
fixes their rotation, shrinks very large phone photos, and writes numbered zip parts to
`FabriCare_Damage_Dataset/For_Labelling/`, each small enough to download and to upload to Roboflow. No boxes are
pre-drawn: you draw every box yourself (`docs/damage-annotation-guide.md`).

File names keep the garment, so the training notebook can keep each garment in one split:
`tshirt_001__holes__img_123.jpg` (own photos), `yellow_dress__phone__img_456.jpg` (phone scans) and
`online-holes-img_789.jpg` (online photos, each its own group). Keep these names when uploading.

**Run:** *Runtime → Run all* (a CPU runtime is enough). Then download the zips from Drive, or drag them into
Roboflow (*Upload Data* accepts zips), and delete them from your laptop afterwards.
""")

code(r"""
# 1. Settings.
SOURCE_NAME = "FabriCare_Damage_Dataset"   # folder in My Drive
FOLDERS = {                                # which subfolders to include
    "Own_Collected_Dataset": True,         # Tshirt_NNN/<class>/*.jpg
    "Phone_Scans": True,                   # <garment>/*.jpg (whole-garment photos)
    "Online_Dataset": True,                # <class>/*.jpg (2,200+ public close-ups)
}
MAX_SIDE = 1920        # longest side in pixels; None keeps the original size (much larger zips)
PART_MB = 450          # target size of each zip part
REBUILD = False        # True deletes For_Labelling/ and packs again
""")

code(r"""
# 2. Mount Drive and pack the photos into zip parts.
import csv, io, os, re, shutil, subprocess, sys, zipfile
from collections import Counter
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")
try:
    import pillow_heif
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow-heif"])
    import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image, ImageOps

SOURCE = Path("/content/drive/MyDrive") / SOURCE_NAME
OUT = SOURCE / "For_Labelling"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
if not SOURCE.is_dir():
    raise SystemExit(f"Not found: {SOURCE}. Add the shared folder to My Drive (Organize → Add shortcut) and rerun.")
if REBUILD:
    shutil.rmtree(OUT, ignore_errors=True)
if (OUT / "DONE.txt").exists():
    print((OUT / "DONE.txt").read_text())
    raise SystemExit("Already packed. Set REBUILD = True to pack again.")
OUT.mkdir(parents=True, exist_ok=True)

def safe(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "x"

def photos():
    # (source, garment, class/view, path, archive name)
    if FOLDERS.get("Own_Collected_Dataset"):
        for garment in sorted(p for p in (SOURCE / "Own_Collected_Dataset").glob("*") if p.is_dir()):
            for folder in sorted(p for p in garment.iterdir() if p.is_dir()):
                for path in sorted(folder.iterdir()):
                    if path.suffix.lower() in SUFFIXES:
                        yield ("own", garment.name, folder.name.strip(), path,
                               f"{safe(garment.name)}__{safe(folder.name)}__{safe(path.stem)}.jpg")
    if FOLDERS.get("Phone_Scans"):
        for garment in sorted(p for p in (SOURCE / "Phone_Scans").glob("*") if p.is_dir()):
            for path in sorted(garment.rglob("*")):
                if path.suffix.lower() in SUFFIXES:
                    yield ("phone", garment.name, "phone", path,
                           f"{safe(garment.name)}__phone__{safe(path.stem)}.jpg")
    if FOLDERS.get("Online_Dataset"):
        for folder in sorted(p for p in (SOURCE / "Online_Dataset").glob("*") if p.is_dir()):
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in SUFFIXES:
                    # No "__": each online photo is its own group when the data is split later.
                    yield ("online", "", folder.name.strip(), path,
                           f"online-{safe(folder.name).replace('_', '-')}-{safe(path.stem)}.jpg")

stage = Path("/content/labelling_parts"); shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
part, part_bytes, archive = 0, 0, None
rows, used, counts, skipped = [], set(), Counter(), []

def next_part():
    global part, part_bytes, archive
    if archive is not None:
        archive.close()
        shutil.copy2(stage / f"labelling_part_{part:02d}.zip", OUT)
        print(f"  saved labelling_part_{part:02d}.zip ({part_bytes / 2**20:.0f} MB)", flush=True)
    part += 1
    part_bytes = 0
    archive = zipfile.ZipFile(stage / f"labelling_part_{part:02d}.zip", "w", zipfile.ZIP_STORED)

next_part()
for n, (source, garment, label, path, name) in enumerate(photos(), 1):
    base, k = name, 2
    while name in used:                      # two photos with the same name: add a number
        name = base.replace(".jpg", f"_{k}.jpg"); k += 1
    used.add(name)
    try:
        with Image.open(path) as raw:
            rgb = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as error:
        skipped.append((str(path), f"{type(error).__name__}: {error}")); continue
    if MAX_SIDE:
        rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO(); rgb.save(buffer, "JPEG", quality=92)
    data = buffer.getvalue()
    if part_bytes and part_bytes + len(data) > PART_MB * 2**20:
        next_part()
    archive.writestr(name, data)
    part_bytes += len(data)
    counts[source] += 1
    rows.append({"part": f"labelling_part_{part:02d}.zip", "file": name, "source": source, "garment": garment,
                 "folder_label": label, "original": str(path.relative_to(SOURCE)),
                 "width": rgb.width, "height": rgb.height})
    if n % 250 == 0:
        print(f"packed {n} photos", flush=True)
next_part()                                  # saves the last part
archive.close(); (stage / f"labelling_part_{part:02d}.zip").unlink()   # drop the empty extra part

with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, list(rows[0]) if rows else ["file"])
    writer.writeheader(); writer.writerows(rows)
summary = (f"{len(rows)} photos in {part - 1} zip parts: " + ", ".join(f"{k} {v}" for k, v in counts.items()) +
           (f". Skipped {len(skipped)} unreadable files (see skipped.csv)." if skipped else "."))
if skipped:
    with (OUT / "skipped.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([("file", "error"), *skipped])
(OUT / "DONE.txt").write_text(summary + "\n")
print(summary)
print("Saved in", OUT, "— download the zips from Drive or drag them into Roboflow (Upload Data).")
""")


def main():
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in CELLS]}
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT), "with", len(CELLS), "cells")


if __name__ == "__main__":
    main()
