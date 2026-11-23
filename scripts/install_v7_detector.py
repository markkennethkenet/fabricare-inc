"""Install the V7 damage detector export into model/best.pt after checking it.

    .venv-api/Scripts/python.exe scripts/install_v7_detector.py C:/path/to/fabricare_v7_detector(.zip) [--force]

The export comes from docs/fabricare_v7_damage_detector_colab.ipynb. The replaced detector is kept as
model/best_previous.pt and the very first one as
model/best_before_v7.pt. Installation stops if the new detector did not beat the current one on the team's
held-out garments, unless --force is given.
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("FABRICARE_MODEL_DIR", ROOT / "model"))
CORE_CLASSES = ("hole", "needlemark", "pilling", "stain")
# Detectors the app accepts: the original four classes, or V7.1 with discoloration added as a fifth.
ACCEPTED_CLASSES = (CORE_CLASSES, CORE_CLASSES + ("discoloration",))


def unpack(source):
    source = Path(source)
    if source.is_dir():
        return source
    target = Path(tempfile.mkdtemp(prefix="fabricare_v7_"))
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise ValueError(f"Unsafe path in export archive: {member}")
        archive.extractall(target)
    return target


def main(source, force=False):
    export = unpack(source)
    weights, report_path = export / "best.pt", export / "v7_detector_report.json"
    if not weights.is_file() or not report_path.is_file():
        raise FileNotFoundError("Export must contain best.pt and v7_detector_report.json.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    from ultralytics import YOLO

    names = YOLO(str(weights)).names
    if tuple(names[i] for i in range(len(names))) not in ACCEPTED_CLASSES:
        raise RuntimeError(f"Detector classes {names} are not one of {ACCEPTED_CLASSES}.")
    old, new = report["baseline"]["own_test"]["overall"], report["v7"]["own_test"]["overall"]
    print(f"Our test garments, mAP50: current {old['mAP50']:.3f} -> V7 {new['mAP50']:.3f}; "
          f"recall {old['recall']:.3f} -> {new['recall']:.3f}")
    if not report.get("better_on_own_test") and not force:
        raise SystemExit("V7 did not beat the current detector on our photos; not installed (use --force to override).")
    current = MODEL / "best.pt"
    backup = MODEL / "best_before_v7.pt"
    previous = MODEL / "best_previous.pt"   # the detector this install replaces (an earlier V7 on later rounds)
    if current.is_file():
        if not backup.is_file():
            shutil.copy2(current, backup)
        shutil.copy2(current, previous)
    shutil.copy2(weights, current)
    shutil.copy2(report_path, MODEL / "v7_detector_report.json")
    print(f"Installed {current} (replaced detector kept at {previous.name}; the original at {backup.name}). "
          "Restart the API to use it.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    main(args[0], force="--force" in sys.argv)
