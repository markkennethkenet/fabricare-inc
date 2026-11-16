"""Install the V4 training export into the API's model folders and verify every model loads.

    .venv-api/Scripts/python.exe scripts/install_v4_models.py C:/path/to/fabricare_v4_export(.zip)

The export comes from docs/fabricare_v4_training_colab.ipynb. Existing V3 GLCM models are kept
under model/glcm_models/v3_backup/. The CNN is re-saved without optimizer state, which roughly
halves its size and keeps it loadable by the API's TensorFlow and Keras versions.
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# FABRICARE_MODEL_DIR lets a test install into a scratch folder instead of the real models.
MODEL = Path(os.environ.get("FABRICARE_MODEL_DIR", ROOT / "model"))
GITHUB_FILE_LIMIT = 100 * 1024 * 1024


def unpack(source):
    source = Path(source)
    if source.is_dir():
        return source
    target = Path(tempfile.mkdtemp(prefix="fabricare_v4_"))
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            # Reject absolute paths and parent references inside the archive.
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise ValueError(f"Unsafe path in export archive: {member}")
        archive.extractall(target)
    return target


def main(source):
    export = unpack(source)
    report = json.loads((export / "training_report.json").read_text(encoding="utf-8"))
    for required in ("fabric_cnn/class_names.json", "fabric_cnn/model_info.json",
                     "glcm_models/glcm_svm.joblib", "glcm_models/glcm_decision_tree.joblib",
                     "glcm_models/metrics.json", "care/care_recommender.joblib"):
        if not (export / required).is_file():
            raise FileNotFoundError(f"Export is missing {required}")

    import numpy as np
    import tensorflow as tf
    from model_runtime import FabricCNN

    source, cnn_dir = export / "fabric_cnn", MODEL / "fabric_cnn"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    half = [FabricCNN.architecture, FabricCNN.half_weights]
    for stale in [FabricCNN.filename, *half]:
        (cnn_dir / stale).unlink(missing_ok=True)
    if all((source / name).is_file() for name in half):
        for name in half:
            shutil.copy2(source / name, cnn_dir / name)
    elif (source / FabricCNN.filename).is_file():
        model = tf.keras.models.load_model(source / FabricCNN.filename, compile=False)
        model.save(cnn_dir / FabricCNN.filename)  # uncompiled: no optimizer state
        if (cnn_dir / FabricCNN.filename).stat().st_size > GITHUB_FILE_LIMIT:
            (cnn_dir / FabricCNN.filename).unlink()
            (cnn_dir / FabricCNN.architecture).write_text(model.to_json(), encoding="utf-8")
            np.savez_compressed(cnn_dir / FabricCNN.half_weights, *[w.astype(np.float16) for w in model.get_weights()])
            print("CNN stored as 16-bit weights to stay under GitHub's file size limit.")
    else:
        raise FileNotFoundError("Export has no fabric_cnn.keras or 16-bit CNN weights.")
    for name in ("class_names.json", "model_info.json"):
        shutil.copy2(source / name, cnn_dir / name)

    glcm_dir = MODEL / "glcm_models"
    backup = glcm_dir / "v3_backup"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("glcm_svm.joblib", "glcm_decision_tree.joblib"):
        if (glcm_dir / name).is_file() and not (backup / name).is_file():
            shutil.copy2(glcm_dir / name, backup / name)
    for name in ("glcm_svm.joblib", "glcm_decision_tree.joblib", "metrics.json"):
        shutil.copy2(export / "glcm_models" / name, glcm_dir / name)

    care_dir = MODEL / "care"
    care_dir.mkdir(exist_ok=True)
    for path in (export / "care").glob("*.joblib"):
        shutil.copy2(path, care_dir / path.name)
    shutil.copy2(export / "training_report.json", MODEL / "v4_training_report.json")

    # Verify with the API's own loaders.
    import care_model
    from model_runtime import FabricGLCM72

    cnn = FabricCNN(cnn_dir)
    svm, tree = FabricGLCM72(glcm_dir, "svm"), FabricGLCM72(glcm_dir, "glcm_dt")
    care = care_model.CareRecommender(care_dir / "care_recommender.joblib")
    sample = next((ROOT / "model").glob("*.jpg"), None) or next(MODEL.glob("*.jpg"), None)
    if sample is not None:
        for classifier in (cnn, svm, tree):
            result = classifier.classify(str(sample))
            print(f"{classifier.name}: {result['fabric_type']} ({result['confidence']:.1f}% score)")
    print("Care model:", care.name)

    oversized = [p for p in MODEL.rglob("*") if p.is_file() and p.stat().st_size > GITHUB_FILE_LIMIT]
    for path in oversized:
        print(f"WARNING: {path} is {path.stat().st_size / 2**20:.0f} MB, over GitHub's 100 MB limit.")

    best = report["fabric_test"]["cnn"][report["selected_cnn"]]["mirror_average"]["test"]
    print(f"Installed {cnn.name}: test accuracy {best['accuracy']:.2%}, macro-F1 {best['macro_f1']:.3f}")
    print(f"GLCM + SVM {svm.test_accuracy}% · GLCM + Decision Tree {tree.test_accuracy}% (test accuracy)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
