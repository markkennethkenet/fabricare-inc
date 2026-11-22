"""Write docs/fabricare_v6_damage_care_colab.ipynb from the cell sources below.

Run from the repository root: python scripts/build_v6_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "fabricare_v6_damage_care_colab.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# FabriCare V6 · Damage condition and care recommendation

**Objective 2.** Train, test and compare a **Decision Tree**, a **Random Forest** and an **SVM** that recommend a
laundry mode from the identified fabric type, the detected fabric condition and extracted image features.

1. **Detected fabric condition.** A ConvNeXt-Tiny classifier is trained on `FabriCare_Damage_Dataset` to label a
   photo as hole, needle mark, normal, pilling, stain or wrinkle. The photos are sorted into class folders without
   boxes, so this is an image-level condition classifier, not a box detector.
2. **Identified fabric type.** The V5 ConvNeXt-Tiny fabric model in the repository (86.8% test accuracy).
3. **Extracted image features.** Colour (luminance, saturation, dark share) and the GLCM texture summary.
4. **Care models.** Each photo is one case; its label is the referenced laundry mode from `care_model`'s rules.
   Accuracy therefore measures agreement with those rules.

**Splits.** Own T-shirt photos are split by garment (T-shirts 1–2 train, 3–4 validation, 5–6 test). Online photos
are grouped by file name and split 70/20/10 per class. `Multiple_Damage` photos are a separate check set. Fabric
photos keep their V5 split, so the fabric model never saw the care test photos.

**How to run.** GPU runtime (L4 or T4). Run cell 1 (installs the API's library versions, restarts once), then
*Runtime → Run all*. Results and the export go to `My Drive/Fabricare_training_v6/run_v6`; finished stages are
skipped on a rerun.
""")

code(r"""
# 1. Match the FabriCare API's library versions so exported models load unchanged.
import importlib.metadata as md, os, subprocess, sys

PINS = {"tensorflow": "2.21.0", "keras": "3.15.1", "scikit-learn": "1.6.1", "scikit-image": "0.26.0",
        "joblib": "1.5.3", "ultralytics": "8.4.104"}
EXTRAS = {"flask": "3.1.3", "flask-cors": "6.0.5", "mysql-connector-python": "9.7.0",
          "python-dotenv": "1.2.2", "pi-heif": "1.4.0"}

def installed(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

missing = [f"{n}=={v}" for n, v in {**PINS, **EXTRAS}.items() if installed(n) != v]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    print("Installed:", missing)
    print("Restarting the runtime. When it reconnects, choose Runtime → Run all.")
    os.kill(os.getpid(), 9)
print("Library versions match the API:", {n: installed(n) for n in PINS})
""")

code(r"""
# 2. Mount Drive, fetch the FabriCare code, scan the damage photos and build the splits.
import csv, hashlib, io, json, shutil, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from google.colab import drive

drive.mount("/content/drive")
REPO_URL = "https://github.com/markkennethkenet/fabricare-inc.git"
REPO_BRANCH = "training/v6"
REPO = Path("/content/fabricare")
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import care_model, damage_condition, damage_split
from model_runtime import SIX_FABRIC_LABELS, extract_glcm72_features
from pi_heif import register_heif_opener
register_heif_opener()

DAMAGE_ROOT = Path("/content/drive/MyDrive/FabriCare_Damage_Dataset")
V3_ROOT = Path("/content/drive/MyDrive/FabriCare_Dataset_V3")
V5_SPLIT = Path("/content/drive/MyDrive/Fabricare_training_v5/run_v5/v5_split.csv")
RUN = Path("/content/drive/MyDrive/Fabricare_training_v6") / "run_v6"
STAGE = Path("/content/stage_v6")
SPLITS = ("train", "valid", "test")
DAMAGE_CLASSES = list(damage_condition.DAMAGE_CLASSES)
FABRICS = list(SIX_FABRIC_LABELS)
SEED = 42
RUN.mkdir(parents=True, exist_ok=True)
STAGE.mkdir(parents=True, exist_ok=True)
for needed in (DAMAGE_ROOT, V3_ROOT / "raw", V5_SPLIT):
    if not needed.exists():
        raise FileNotFoundError(f"Missing {needed}. Add the Drive folder (or run the V5 notebook) first.")

SCAN_CACHE = RUN / "scan_cache.json"
scan_cache = json.loads(SCAN_CACHE.read_text()) if SCAN_CACHE.is_file() else {}
def fingerprint(path):
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    if key not in scan_cache:
        scan_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return scan_cache[key]

def images(folder):
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in damage_split.IMAGE_SUFFIXES]

entries = []
for folder in sorted(p for p in (DAMAGE_ROOT / "Online_Dataset").iterdir() if p.is_dir()):
    entries += [{"source": "online", "folder": folder.name, "path": p} for p in images(folder)]
for garment in sorted(p for p in (DAMAGE_ROOT / "Own_Collected_Dataset").iterdir() if p.is_dir()):
    for folder in sorted(p for p in garment.iterdir() if p.is_dir()):
        entries += [{"source": "own", "garment": garment.name, "folder": folder.name, "path": p} for p in images(folder)]
started = time.monotonic()
with ThreadPoolExecutor(max_workers=16) as pool:
    hashes = list(pool.map(lambda e: fingerprint(e["path"]), entries))
SCAN_CACHE.write_text(json.dumps(scan_cache))
print(f"Fingerprinted {len(entries)} damage photos in {time.monotonic() - started:.0f}s")
scanned = [{**e, "path": str(e["path"]), "filename": e["path"].name, "sha256": h} for e, h in zip(entries, hashes)]
damage_records, damage_report = damage_split.build_split(scanned)
print("Damage split report:", json.dumps(damage_report, indent=1))
with (RUN / "damage_split.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, ["split", "label", "source", "garment", "group_id", "sha256", "filename"],
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(damage_records)

# Fabric photos keep the V5 split, so the fabric CNN never saw the care test photos.
fabric_records = [{"source": "fabric", "split": r["split"], "fabric": r["fabric"], "sha256": r["sha256"],
                   "filename": r["filename"], "path": str(V3_ROOT / "raw" / r["fabric"] / r["filename"])}
                  for r in csv.DictReader(V5_SPLIT.read_text(encoding="utf-8").splitlines())]
print("Fabric photos from the V5 split:", dict(Counter(r["split"] for r in fabric_records)))
""")

code(r"""
# 3. Stage every photo once: a 224 px grey-padded copy, colour features and the 72 GLCM features.
def stage(record):
    key = record["sha256"]
    padded, meta = STAGE / "padded" / f"{key}.png", STAGE / "meta" / f"{key}.json"
    if meta.is_file() and padded.is_file():
        return {**record, **json.loads(meta.read_text())}
    try:
        content = Path(record["path"]).read_bytes()
        with Image.open(io.BytesIO(content)) as raw:
            rgb = ImageOps.exif_transpose(raw).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return {**record, "unreadable": f"{type(error).__name__}: {error}"}
    for folder in ("padded", "meta"):
        (STAGE / folder).mkdir(parents=True, exist_ok=True)
    ImageOps.pad(rgb, (224, 224), method=Image.Resampling.BILINEAR, color=(127, 127, 127)).save(padded)
    info = {"padded": str(padded), "color": care_model.color_features(io.BytesIO(content)),
            "glcm": care_model.glcm_summary(extract_glcm72_features(str(padded))[0])}
    meta.write_text(json.dumps(info))
    return {**record, **info}

def stage_all(records, label):
    started, out = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, item in enumerate(pool.map(stage, records), 1):
            out.append(item)
            if i % 500 == 0:
                print(f"{label}: staged {i}/{len(records)} in {time.monotonic() - started:.0f}s", flush=True)
    bad = [i for i in out if "unreadable" in i]
    for item in bad:
        print("  skipped unreadable:", item["path"], "|", item["unreadable"])
    return [i for i in out if "unreadable" not in i]

damage_items = stage_all(damage_records, "damage")
fabric_items = stage_all(fabric_records, "fabric")
damage_by_split = defaultdict(list)
for item in damage_items:
    damage_by_split[item["split"]].append(item)
for split in (*SPLITS, "multiple"):
    chosen = damage_by_split[split]
    print(f"damage {split:8s} {len(chosen):5d}", dict(sorted(Counter(i["label"] for i in chosen).items())),
          "| own:", sum(i["source"] == "own" for i in chosen))
""")

md(r"""
## Detected fabric condition · ConvNeXt-Tiny damage classifier
""")

code(r"""
# 4. Classifier helpers (same training recipe as the V5 fabric CNN).
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

assert tf.config.list_physical_devices("GPU"), "Select a GPU runtime before training."
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.utils.set_random_seed(SEED)

def arrays(items, classes, key):
    return [i["padded"] for i in items], np.asarray([classes.index(i[key]) for i in items], np.int32)

def dataset(items, classes, key, training=False, batch=32):
    paths, labels = arrays(items, classes, key)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    def decode(path, label):
        image = tf.cast(tf.ensure_shape(tf.io.decode_png(tf.io.read_file(path), channels=3), (224, 224, 3)), tf.float32)
        return image, tf.one_hot(label, len(classes))
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)

def predict(model, items):
    # Mirror-averaged probabilities, as the API scores a photo.
    ds = tf.data.Dataset.from_tensor_slices([i["padded"] for i in items]).map(
        lambda p: tf.cast(tf.io.decode_png(tf.io.read_file(p), channels=3), tf.float32)).batch(64)
    single = model.predict(ds, verbose=0)
    flipped = model.predict(ds.map(lambda x: tf.reverse(x, axis=[2])), verbose=0)
    both = (single + flipped) / 2
    return both / both.sum(axis=1, keepdims=True)

def metrics(truth, guess, classes):
    ids = list(range(len(classes)))
    report = classification_report(truth, guess, labels=ids, target_names=list(classes), output_dict=True,
                                   zero_division=0)
    return {"accuracy": float(accuracy_score(truth, guess)),
            "macro_f1": float(f1_score(truth, guess, labels=ids, average="macro", zero_division=0)),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                          for c in classes},
            "confusion_matrix_actual_by_predicted": confusion_matrix(truth, guess, labels=ids).tolist()}

FILL = 127.0
def augmentation():
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.04, fill_mode="constant", fill_value=FILL, seed=SEED + 1),
        layers.RandomZoom((-0.2, 0.15), fill_mode="constant", fill_value=FILL, seed=SEED + 2),
        layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=FILL, seed=SEED + 3),
        layers.RandomBrightness(0.15, value_range=(0, 255), seed=SEED + 4),
        layers.RandomContrast(0.2, value_range=(0, 255), seed=SEED + 5),
    ], name="retake_augmentation")

def build(classes):
    backbone = tf.keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet", input_shape=(224, 224, 3),
                                                  include_preprocessing=True)
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3), name="rgb_0_to_255")
    x = backbone(augmentation()(inputs), training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax", name="condition_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="fabricare_damage_convnext_tiny"), backbone

class SelectOnValidation(tf.keras.callbacks.Callback):
    def __init__(self, items, path, patience, history):
        super().__init__()
        self.items, self.path, self.patience, self.history = items, path, patience, history
        self.truth = arrays(items, DAMAGE_CLASSES, "label")[1]
        self.best, self.wait = max([h["valid_macro_f1"] for h in history] or [-1.0]), 0
    def on_epoch_end(self, epoch, logs=None):
        score = metrics(self.truth, predict(self.model, self.items).argmax(1), DAMAGE_CLASSES)["macro_f1"]
        self.history.append({"epoch": len(self.history) + 1, "valid_macro_f1": score,
                             **{k: float(v) for k, v in (logs or {}).items()}})
        print(f"epoch {len(self.history)} valid macro-F1 {score:.4f}", flush=True)
        if score > self.best:
            self.best, self.wait = score, 0
            self.model.save(self.path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

def class_weights(items):
    counts = Counter(DAMAGE_CLASSES.index(i["label"]) for i in items)
    total = sum(counts.values())
    return {k: float(min(3.0, np.sqrt(total / (len(DAMAGE_CLASSES) * n)))) for k, n in counts.items()}
""")

code(r"""
# 5. Train the damage classifier (skipped if already finished), then test it.
folder = RUN / "damage_classifier"
folder.mkdir(parents=True, exist_ok=True)
best_path, done = folder / "best.keras", folder / "result.json"
if done.is_file():
    damage_train = json.loads(done.read_text())
    print("Already trained; validation macro-F1", round(damage_train["valid"]["macro_f1"], 4))
else:
    started, history = time.monotonic(), []
    model, backbone = build(DAMAGE_CLASSES)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    train_ds = dataset(damage_by_split["train"], DAMAGE_CLASSES, "label", training=True)
    valid_ds = dataset(damage_by_split["valid"], DAMAGE_CLASSES, "label")
    weights = class_weights(damage_by_split["train"])
    model.compile(tf.keras.optimizers.Adam(1e-3), loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=valid_ds, epochs=15, class_weight=weights, verbose=2,
              callbacks=[SelectOnValidation(damage_by_split["valid"], best_path, 4, history)])
    model = tf.keras.models.load_model(best_path, compile=False)
    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model) and l.name != "retake_augmentation")
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    schedule = tf.keras.optimizers.schedules.CosineDecay(3e-5, 30 * len(train_ds))
    model.compile(tf.keras.optimizers.AdamW(schedule, weight_decay=1e-4), loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=valid_ds, epochs=30, class_weight=weights, verbose=2,
              callbacks=[SelectOnValidation(damage_by_split["valid"], best_path, 6, history)])
    model = tf.keras.models.load_model(best_path, compile=False)
    truth = arrays(damage_by_split["valid"], DAMAGE_CLASSES, "label")[1]
    damage_train = {"valid": metrics(truth, predict(model, damage_by_split["valid"]).argmax(1), DAMAGE_CLASSES),
                    "history": history, "minutes": (time.monotonic() - started) / 60}
    done.write_text(json.dumps(damage_train, indent=2))
    tf.keras.backend.clear_session()

damage_model = tf.keras.models.load_model(best_path, compile=False)
test_items = damage_by_split["test"]
test_truth = arrays(test_items, DAMAGE_CLASSES, "label")[1]
test_probs = predict(damage_model, test_items)
own = np.asarray([i["source"] == "own" for i in test_items])
damage_test = {"test": metrics(test_truth, test_probs.argmax(1), DAMAGE_CLASSES),
               "test_own_tshirts": metrics(test_truth[own], test_probs[own].argmax(1), DAMAGE_CLASSES),
               "test_online": metrics(test_truth[~own], test_probs[~own].argmax(1), DAMAGE_CLASSES)}
multiple = damage_by_split["multiple"]
if multiple:
    guesses = predict(damage_model, multiple).argmax(1)
    damage_test["multiple_damage_check"] = {
        "photos": len(multiple), "flagged_as_damaged": float(np.mean([DAMAGE_CLASSES[g] != "normal" for g in guesses])),
        "predicted": dict(Counter(DAMAGE_CLASSES[g] for g in guesses))}
for name in ("test", "test_own_tshirts", "test_online"):
    m = damage_test[name]
    print(f"{name:17s} n={sum(v['support'] for v in m['per_class'].values()):4.0f} accuracy {m['accuracy']:.2%} "
          f"macro-F1 {m['macro_f1']:.3f}")
print("per-class F1:", {c: round(v["f1-score"], 3) for c, v in damage_test["test"]["per_class"].items()})
print("Multiple-damage check:", damage_test.get("multiple_damage_check"))
""")

md(r"""
## Objective 2 · Care recommendation: Decision Tree vs Random Forest vs SVM
Each photo is one case: fabric (one-hot), condition (one-hot), colour and GLCM features. The reference label comes
from `care_model`'s referenced rules. Where a photo has no ground truth for an input, the trained model supplies
it: the fabric of a damage photo comes from the V5 fabric CNN, and the condition of a fabric photo comes from the
damage classifier. Test results are reported with these reference inputs and end to end, where both models supply
every input.
""")

code(r"""
# 6. Fabric and condition predictions from the two CNNs.
def load_fabric_cnn(folder=REPO / "model" / "fabric_cnn"):
    model = tf.keras.models.model_from_json((folder / "fabric_cnn_architecture.json").read_text(encoding="utf-8"))
    with np.load(folder / "fabric_cnn_weights_fp16.npz") as stored:
        model.set_weights([stored[f"arr_{i}"].astype(np.float32) for i in range(len(stored.files))])
    return model

care_items = [i for i in fabric_items + damage_items if i["split"] in SPLITS]
fabric_cnn = load_fabric_cnn()
fabric_pred = predict(fabric_cnn, care_items).argmax(1)
del fabric_cnn
tf.keras.backend.clear_session()
damage_model = tf.keras.models.load_model(best_path, compile=False)
condition_pred = predict(damage_model, care_items).argmax(1)
for item, f, c in zip(care_items, fabric_pred, condition_pred):
    item["fabric_pred"], item["condition_pred"] = FABRICS[f], DAMAGE_CLASSES[c]
fabric_check = [i for i in care_items if i["source"] == "fabric" and i["split"] == "test"]
print("V5 fabric CNN on the fabric test photos:",
      f"{np.mean([i['fabric_pred'] == i['fabric'] for i in fabric_check]):.2%}")
print("Conditions the classifier gives the fabric photos:",
      dict(Counter(i["condition_pred"] for i in care_items if i["source"] == "fabric")))
""")

code(r"""
# 7. Build care cases, train each model with validation-based settings, and test.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

CARE_CLASSES = list(care_model.CLASSES)
def reference_inputs(item):
    fabric = item["fabric"] if item["source"] == "fabric" else item["fabric_pred"]
    condition = item["label"] if item["source"] != "fabric" else item["condition_pred"]
    return fabric, condition

def case(item, end_to_end=False):
    fabric, condition = (item["fabric_pred"], item["condition_pred"]) if end_to_end else reference_inputs(item)
    return damage_condition.feature_row(fabric, condition, item["color"], item["glcm"])

care_x, care_y, care_e2e = {}, {}, {}
for split in SPLITS:
    items = [i for i in care_items if i["split"] == split]
    rows = [case(i) for i in items]
    care_x[split] = np.vstack([damage_condition.feature_vector(r) for r in rows])
    care_y[split] = np.asarray([CARE_CLASSES.index(damage_condition.reference_laundry_mode(r)[0]) for r in rows])
    care_e2e[split] = np.vstack([damage_condition.feature_vector(case(i, True)) for i in items])
    print(split, len(items), dict(Counter(CARE_CLASSES[c] for c in care_y[split])))
test_source = np.asarray([i["source"] for i in care_items if i["split"] == "test"])

care_specs = {
    "Decision Tree": (lambda **p: DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **p),
                      {"criterion": ["gini", "entropy"], "max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 3, 5]}),
    "Random Forest": (lambda **p: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                         random_state=SEED, n_jobs=-1, **p),
                      {"max_depth": [8, 16, None], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.5]}),
    "SVM": (lambda **p: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True,
                                                            random_state=SEED, **p)),
            {"C": [1, 10, 100], "gamma": ["scale", 0.03, 0.1]}),
}
CARE_IDS = list(range(len(CARE_CLASSES)))
def complete(x, y):
    # Fit on every class index so classes_ always lists all five modes in order.
    absent = [c for c in CARE_IDS if c not in set(y)]
    return (np.vstack([x, np.zeros((len(absent), x.shape[1]))]), np.concatenate([y, absent])) if absent else (x, y)

def subset(guess, mask):
    return metrics(care_y["test"][mask], guess[mask], CARE_CLASSES) if mask.any() else None

care_results, care_models = {}, {}
for name, (make, grid) in care_specs.items():
    started, trials = time.monotonic(), []
    for params in ParameterGrid(grid):
        model = make(**params).fit(*complete(care_x["train"], care_y["train"]))
        trials.append((f1_score(care_y["valid"], model.predict(care_x["valid"]), labels=CARE_IDS, average="macro",
                                zero_division=0), params, model))
    score, params, model = max(trials, key=lambda t: t[0])
    care_models[name] = model
    reference, e2e = model.predict(care_x["test"]), model.predict(care_e2e["test"])
    care_results[name] = {
        "params": params, "valid_macro_f1": float(score), "fit_seconds": time.monotonic() - started,
        "test_reference_inputs": metrics(care_y["test"], reference, CARE_CLASSES),
        "test_end_to_end": metrics(care_y["test"], e2e, CARE_CLASSES),
        "test_end_to_end_damage_photos": subset(e2e, test_source != "fabric"),
        "test_end_to_end_fabric_photos": subset(e2e, test_source == "fabric")}
    r = care_results[name]
    print(f"{name:14s} valid F1 {score:.3f} | test acc {r['test_reference_inputs']['accuracy']:.2%} / "
          f"F1 {r['test_reference_inputs']['macro_f1']:.3f} (reference inputs) | "
          f"{r['test_end_to_end']['accuracy']:.2%} / {r['test_end_to_end']['macro_f1']:.3f} (end to end) | {params}",
          flush=True)
best_care = max(care_results, key=lambda k: care_results[k]["valid_macro_f1"])
print("Selected care model on validation:", best_care)
""")

md(r"""
## Export
""")

code(r"""
# 8. Save the damage classifier, the three care models and a full report; zip for download.
import importlib.metadata
export = RUN / "export"
if export.exists():
    shutil.rmtree(export)
(export / "damage_classifier").mkdir(parents=True)
(export / "care_v6").mkdir()

half = [w.astype(np.float16) for w in damage_model.get_weights()]
rebuilt = tf.keras.models.model_from_json(damage_model.to_json())
rebuilt.set_weights([w.astype(np.float32) for w in half])
half_check = metrics(test_truth, predict(rebuilt, test_items).argmax(1), DAMAGE_CLASSES)
np.savez_compressed(export / "damage_classifier" / "damage_cnn_weights_fp16.npz", *half)
(export / "damage_classifier" / "damage_cnn_architecture.json").write_text(damage_model.to_json())
(export / "damage_classifier" / "class_names.json").write_text(json.dumps(DAMAGE_CLASSES, indent=2))
(export / "damage_classifier" / "model_info.json").write_text(json.dumps({
    "name": "ConvNeXt-Tiny damage condition (V6)", "classes": DAMAGE_CLASSES,
    "validation_accuracy": round(damage_train["valid"]["accuracy"] * 100, 2),
    "test_accuracy": round(half_check["accuracy"] * 100, 2), "test_macro_f1": round(half_check["macro_f1"], 4),
    "weights": "float16", "inference": "mirror_average", "input": "224x224 RGB, 0-255, gray (127) padding"},
    indent=2))
for name, model in care_models.items():
    joblib.dump({"model": model, "feature_names": list(damage_condition.FEATURE_NAMES), "classes": CARE_CLASSES,
                 "name": name, "metrics": care_results[name]},
                export / "care_v6" / f"care_{name.lower().replace(' ', '_')}.joblib")
shutil.copy2(export / "care_v6" / f"care_{best_care.lower().replace(' ', '_')}.joblib",
             export / "care_v6" / "care_recommender.joblib")

report = {
    "created": datetime.now(timezone.utc).isoformat(),
    "damage_dataset": "https://drive.google.com/drive/folders/1kormVmRAGOwAi93jjcThhvj1Di5-lD46",
    "damage_split_report": damage_report,
    "damage_counts": {s: dict(Counter(i["label"] for i in damage_by_split[s])) for s in (*SPLITS, "multiple")},
    "care_counts": {s: dict(Counter(CARE_CLASSES[c] for c in care_y[s])) for s in SPLITS},
    "damage_classifier": {"validation": damage_train["valid"], **damage_test, "test_16bit": half_check},
    "care": care_results, "selected_care_model": best_care,
    "care_feature_names": list(damage_condition.FEATURE_NAMES),
    "versions": {n: importlib.metadata.version(n) for n in ("tensorflow", "keras", "scikit-learn", "scikit-image",
                                                            "joblib", "numpy")},
}
(export / "training_report_v6.json").write_text(json.dumps(report, indent=2))
archive = shutil.make_archive(str(RUN / "fabricare_v6_export"), "zip", export)
print("16-bit damage classifier test accuracy:", f"{half_check['accuracy']:.2%}")
print("Zip:", archive)
print()
print(f"{'model':14s} {'acc (ref)':>10s} {'F1 (ref)':>9s} {'acc (e2e)':>10s} {'F1 (e2e)':>9s}")
for name, r in care_results.items():
    print(f"{name:14s} {r['test_reference_inputs']['accuracy']:10.2%} {r['test_reference_inputs']['macro_f1']:9.3f} "
          f"{r['test_end_to_end']['accuracy']:10.2%} {r['test_end_to_end']['macro_f1']:9.3f}")
print("Damage classifier test:", f"{damage_test['test']['accuracy']:.2%}", "macro-F1",
      round(damage_test["test"]["macro_f1"], 3))
""")


def main():
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [{**cell, "source": cell["source"].splitlines(keepends=True)} for cell in CELLS],
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT), "with", len(CELLS), "cells")


if __name__ == "__main__":
    main()
