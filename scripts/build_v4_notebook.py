"""Write docs/fabricare_v4_training_colab.ipynb from the cell sources below.

Run from the repository root: python scripts/build_v4_notebook.py
Keeping the cells here makes the notebook reviewable as ordinary Python.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "fabricare_v4_training_colab.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# FabriCare V4 · Train, test and compare every model

This notebook covers the two machine-learning objectives of the FabriCare study:

1. **Fabric identification.** Train, test and compare a CNN, GLCM + Decision Tree and GLCM + SVM on the six fabrics.
   Three CNN backbones (MobileNetV2, EfficientNetV2-B0, ConvNeXt-Tiny) are trained with retake-style augmentation and
   the best one is chosen on validation data only.
2. **Care recommendation.** Train, test and compare a Decision Tree, Random Forest and SVM that suggest a laundry mode
   from the identified fabric, the detected condition (YOLOv8 damage findings) and extracted image features
   (colour and GLCM texture). Labels come from the referenced care rules in `care_model.py`; accuracy measures
   agreement with those rules.

**Data.** All V3 photos from the audited garment-grouped split manifest, plus the 292 verified polyester additions in
the training split. Test photos are read only in the final evaluation. The 11 phone photos in `FabriCare_Dataset_V3/test`
(one cotton garment) are a separate real-world consistency check.

**How to run.** Select *Runtime → Change runtime type → T4 GPU*. Run cell 1; it installs the API's exact library
versions and restarts the runtime once. Then choose *Runtime → Run all*. Authorize Google Drive when asked.
Every stage saves to `My Drive/Fabricare_training_v4/<run>` and is skipped on a rerun, so a disconnect loses little.
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
# 2. Mount Drive, fetch the FabriCare code, and check the audited V3 split.
import csv, hashlib, io, json, shutil, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from google.colab import drive

drive.mount("/content/drive")
REPO_URL = "https://github.com/markkennethkenet/fabricare-inc.git"
REPO_BRANCH = "training/v4"
REPO = Path("/content/fabricare")
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import care_model
from model_runtime import SIX_FABRIC_LABELS, extract_glcm72_features

V3_URL = "https://drive.google.com/drive/folders/1cHixSkWAvL3gK1Mhuw8s6G0E7RMYx0-8"
V3_ROOT = Path("/content/drive/MyDrive/FabriCare_Dataset_V3")
MANIFEST_SHA256 = "f164042dcb4a0b4876fc5474532df1bb7b25403703e4a70852f19ada64c287ea"
CLASS_NAMES = list(SIX_FABRIC_LABELS)
SPLITS = ("train", "valid", "test")
SEED = 42
RUN = Path("/content/drive/MyDrive/Fabricare_training_v4") / "run_v4"
STAGE = Path("/content/stage_v4")
RUN.mkdir(parents=True, exist_ok=True)
STAGE.mkdir(parents=True, exist_ok=True)

manifest_path = V3_ROOT / "reports" / "split_manifest_70_20_10.csv"
if not manifest_path.is_file():
    raise FileNotFoundError(f"Open {V3_URL}, choose Organize → Add shortcut → My Drive, keep the name "
                            f"{V3_ROOT.name}, then rerun this cell.")
manifest_bytes = manifest_path.read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
    raise ValueError("The split manifest differs from the audited V3 revision.")
rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"), newline="")))
groups = defaultdict(set)
for row in rows:
    assert row["split"] in SPLITS and row["fabric"] in CLASS_NAMES, row
    groups[row["group_id"]].add(row["split"])
assert all(len(s) == 1 for s in groups.values()), "A garment group crosses splits."
print("V3 manifest:", Counter((r["split"], r["fabric"]) for r in rows))

selection = json.loads((REPO / "research" / "polyester_v3" / "selected_images.json").read_text(encoding="utf-8"))
known = {r["sha256"] for r in rows}
additions = [{"split": "train", "fabric": "polyester", "filename": item["name"], "sha256": item["sha256"],
              "group_id": "polyester_addition:" + item["sha256"][:16]}
             for item in selection if item["sha256"] not in known]
print("Verified polyester additions for training:", len(additions))
phone_paths = sorted(p for p in (V3_ROOT / "test").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print("Phone photos (cotton garment):", len(phone_paths))
""")

code(r"""
# 3. Stage every photo once: a 224 px padded copy (scanner input), a ≤1920 px JPEG (detector input)
#    and colour features from the original. Source photos on Drive are never modified.
def source_path(record):
    path = V3_ROOT / "raw" / record["fabric"] / record["filename"]
    return path if path.is_file() else path.with_name(path.name.replace(" (1)", ""))

def stage(record):
    key = record["sha256"]
    padded, full = STAGE / "padded" / f"{key}.png", STAGE / "full" / f"{key}.jpg"
    meta = STAGE / "meta" / f"{key}.json"
    if meta.is_file() and padded.is_file() and full.is_file():
        return json.loads(meta.read_text())
    content = record["path"].read_bytes() if "path" in record else source_path(record).read_bytes()
    if "path" not in record and hashlib.sha256(content).hexdigest() != key:
        raise ValueError(f"Photo differs from the manifest: {record['filename']}")
    with Image.open(io.BytesIO(content)) as raw:
        rgb = ImageOps.exif_transpose(raw).convert("RGB")
    for folder in ("padded", "full", "meta"):
        (STAGE / folder).mkdir(parents=True, exist_ok=True)
    ImageOps.pad(rgb, (224, 224), method=Image.Resampling.BILINEAR, color=(127, 127, 127)).save(padded)
    copy = rgb.copy()
    copy.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    copy.save(full, format="JPEG", quality=92)
    info = {"sha256": key, "split": record["split"], "fabric": record["fabric"], "group_id": record["group_id"],
            "padded": str(padded), "full": str(full), "color": care_model.color_features(io.BytesIO(content))}
    meta.write_text(json.dumps(info))
    return info

records = rows + additions
phone_records = [{"split": "phone", "fabric": "cotton", "filename": p.name, "path": p, "group_id": "phone",
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in phone_paths]
started = time.monotonic()
with ThreadPoolExecutor(max_workers=8) as pool:
    staged = []
    for i, item in enumerate(pool.map(stage, records + phone_records), 1):
        staged.append(item)
        if i % 250 == 0:
            print(f"staged {i}/{len(records) + len(phone_records)} in {time.monotonic() - started:.0f}s", flush=True)
by_split = defaultdict(list)
for item in staged:
    by_split[item["split"]].append(item)
for split in (*SPLITS, "phone"):
    print(split, len(by_split[split]), dict(Counter(i["fabric"] for i in by_split[split])))
""")

md(r"""
## Objective 1 · CNN: train three backbones and select on validation
""")

code(r"""
# 4. CNN training helpers.
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

assert tf.config.list_physical_devices("GPU"), "Select a T4 GPU runtime before training."
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)  # leave GPU memory for the YOLO damage detector
tf.keras.utils.set_random_seed(SEED)
LABEL_IDS = list(range(len(CLASS_NAMES)))

def arrays(items):
    return [i["padded"] for i in items], np.asarray([CLASS_NAMES.index(i["fabric"]) for i in items], np.int32)

def dataset(items, training=False, batch=32):
    paths, labels = arrays(items)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    def decode(path, label):
        image = tf.cast(tf.ensure_shape(tf.io.decode_png(tf.io.read_file(path), channels=3), (224, 224, 3)), tf.float32)
        return image, tf.one_hot(label, len(CLASS_NAMES))
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)

def predict(model, items, mirror=True):
    paths, _ = arrays(items)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(
        lambda p: tf.cast(tf.io.decode_png(tf.io.read_file(p), channels=3), tf.float32)).batch(64)
    single = model.predict(ds, verbose=0)
    if not mirror:
        return single, single
    flipped = model.predict(ds.map(lambda x: tf.reverse(x, axis=[2])), verbose=0)
    both = (single + flipped) / 2
    return single, both / both.sum(axis=1, keepdims=True)

def metrics(truth, probabilities):
    guess = probabilities.argmax(axis=1)
    report = classification_report(truth, guess, labels=LABEL_IDS, target_names=CLASS_NAMES, output_dict=True,
                                   zero_division=0)
    return {"accuracy": float(accuracy_score(truth, guess)),
            "macro_f1": float(f1_score(truth, guess, labels=LABEL_IDS, average="macro", zero_division=0)),
            "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score")} for c in CLASS_NAMES},
            "confusion_matrix_actual_by_predicted": confusion_matrix(truth, guess, labels=LABEL_IDS).tolist()}

FILL = 127.0
def augmentation():
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.04, fill_mode="constant", fill_value=FILL, seed=SEED + 1),  # about ±14°
        layers.RandomZoom((-0.2, 0.15), fill_mode="constant", fill_value=FILL, seed=SEED + 2),
        layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=FILL, seed=SEED + 3),
        layers.RandomBrightness(0.15, value_range=(0, 255), seed=SEED + 4),
        layers.RandomContrast(0.2, value_range=(0, 255), seed=SEED + 5),
    ], name="retake_augmentation")

BACKBONES = {
    "mobilenetv2": ("MobileNetV2", lambda: tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)), True),
    "efficientnetv2b0": ("EfficientNetV2-B0", lambda: tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), include_preprocessing=True), False),
    "convnext_tiny": ("ConvNeXt-Tiny", lambda: tf.keras.applications.ConvNeXtTiny(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), include_preprocessing=True), False),
}

def build(key):
    name, factory, rescale = BACKBONES[key]
    backbone = factory()
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3), name="rgb_0_to_255")
    x = augmentation()(inputs)
    if rescale:
        x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="fabric_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name=f"fabricare_{key}"), backbone

class SelectOnValidation(tf.keras.callbacks.Callback):
    # Keep the epoch with the best mirror-averaged validation macro-F1.
    def __init__(self, items, path, patience, history):
        super().__init__()
        self.items, self.path, self.patience, self.history = items, path, patience, history
        self.truth = arrays(items)[1]
        self.best, self.wait = max([h["valid_macro_f1"] for h in history] or [-1.0]), 0
    def on_epoch_end(self, epoch, logs=None):
        score = metrics(self.truth, predict(self.model, self.items)[1])["macro_f1"]
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
    counts = Counter(CLASS_NAMES.index(i["fabric"]) for i in items)
    total = sum(counts.values())
    return {k: float(min(3.0, np.sqrt(total / (len(CLASS_NAMES) * n)))) for k, n in counts.items()}

def train_cnn(key, head_epochs=15, fine_epochs=30):
    folder = RUN / "cnn" / key
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "result.json"
    if done.is_file():
        print(key, "already trained:", json.loads(done.read_text())["valid"]["macro_f1"])
        return json.loads(done.read_text())
    best = folder / "best.keras"
    history = []
    model, backbone = build(key)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    train_ds, valid_ds = dataset(by_split["train"], True), dataset(by_split["valid"])
    weights = class_weights(by_split["train"])
    model.compile(tf.keras.optimizers.Adam(1e-3), loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=valid_ds, epochs=head_epochs, class_weight=weights, verbose=2,
              callbacks=[SelectOnValidation(by_split["valid"], best, 4, history)])
    model = tf.keras.models.load_model(best, compile=False)
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                    and layer.name not in {"retake_augmentation"})
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    steps = fine_epochs * len(train_ds)
    schedule = tf.keras.optimizers.schedules.CosineDecay(3e-5 if key == "convnext_tiny" else 1e-4, steps)
    model.compile(tf.keras.optimizers.AdamW(schedule, weight_decay=1e-4), loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=valid_ds, epochs=fine_epochs, class_weight=weights, verbose=2,
              callbacks=[SelectOnValidation(by_split["valid"], best, 6, history)])
    model = tf.keras.models.load_model(best, compile=False)
    truth = arrays(by_split["valid"])[1]
    single, mirror = predict(model, by_split["valid"])
    result = {"key": key, "name": BACKBONES[key][0], "valid": metrics(truth, mirror),
              "valid_single_view": metrics(truth, single), "history": history}
    done.write_text(json.dumps(result, indent=2))
    tf.keras.backend.clear_session()
    return result
""")

code(r"""
# 5. Train the three CNN backbones (each saves to Drive and is skipped if already finished).
cnn_results = {}
for key in BACKBONES:
    started = time.monotonic()
    cnn_results[key] = train_cnn(key)
    print(key, "validation:", round(cnn_results[key]["valid"]["accuracy"], 4), "accuracy,",
          round(cnn_results[key]["valid"]["macro_f1"], 4), "macro-F1,",
          f"{(time.monotonic() - started) / 60:.1f} min", flush=True)
best_cnn = max(cnn_results, key=lambda k: cnn_results[k]["valid"]["macro_f1"])
print("Selected on validation:", best_cnn)
""")

md(r"""
## Objective 1 · GLCM + Decision Tree and GLCM + SVM
""")

code(r"""
# 6. GLCM features (the API's exact 72-feature extractor) and validation-based model selection.
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

GLCM_CACHE_FILE = STAGE / "glcm72.json"
GLCM_CACHE = json.loads(GLCM_CACHE_FILE.read_text()) if GLCM_CACHE_FILE.is_file() else {}

def glcm_matrix(items):
    missing = [i for i in items if i["sha256"] not in GLCM_CACHE]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item, vector in zip(missing, pool.map(lambda i: extract_glcm72_features(i["padded"])[0].tolist(), missing)):
            GLCM_CACHE[item["sha256"]] = vector
    if missing:
        GLCM_CACHE_FILE.write_text(json.dumps(GLCM_CACHE))
    return np.asarray([GLCM_CACHE[i["sha256"]] for i in items])

glcm = {split: glcm_matrix(by_split[split]) for split in (*SPLITS, "phone")}
glcm_y = {split: arrays(by_split[split])[1] for split in (*SPLITS, "phone")}

def select(name, make, grid):
    path = RUN / "glcm" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return joblib.load(path), json.loads((path.with_suffix(".json")).read_text())
    trials = []
    for params in ParameterGrid(grid):
        model = make(**params).fit(glcm["train"], glcm_y["train"])
        score = f1_score(glcm_y["valid"], model.predict(glcm["valid"]), average="macro", zero_division=0)
        trials.append((score, params, model))
        print(name, params, round(score, 4), flush=True)
    score, params, model = max(trials, key=lambda t: t[0])
    joblib.dump(model, path)
    info = {"params": params, "valid_macro_f1": float(score)}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return model, info

glcm_dt, glcm_dt_info = select("glcm_decision_tree", lambda **p: DecisionTreeClassifier(
    class_weight="balanced", random_state=SEED, **p),
    {"max_depth": [6, 10, 14, 20, None], "min_samples_leaf": [1, 3, 5, 10], "criterion": ["gini", "entropy"]})
glcm_svm, glcm_svm_info = select("glcm_svm", lambda C, gamma: make_pipeline(StandardScaler(), SVC(
    C=C, gamma=gamma, class_weight="balanced", probability=True, random_state=SEED)),
    {"C": [1, 3, 10, 30, 100], "gamma": ["scale", 0.003, 0.01, 0.03]})
print("Decision Tree:", glcm_dt_info, "\nSVM:", glcm_svm_info)
""")

md(r"""
## Objective 1 · Final comparison on the held-out test split
The test photos are used here for the first time. The current production CNN is evaluated on the same photos as a baseline.
""")

code(r"""
# 7. Test every fabric model: clean test photos, simulated retakes, and the cotton phone photos.
import evaluate_fabric_scans as retakes

def cnn_model(key):
    return tf.keras.models.load_model(RUN / "cnn" / key / "best.keras", compile=False)

def retake_items(items, per_photo=4):
    rng = np.random.default_rng(SEED)
    folder = STAGE / "retakes"
    folder.mkdir(exist_ok=True)
    out = []
    for item in items:
        with Image.open(item["full"]) as photo:
            rgb = photo.convert("RGB")
        for k in range(per_photo):
            target = folder / f"{item['sha256']}_{k}.png"
            if not target.is_file():
                source = retakes.retake(rgb, rng)
                with Image.open(source) as shot:
                    ImageOps.pad(shot.convert("RGB"), (224, 224), method=Image.Resampling.BILINEAR,
                                 color=(127, 127, 127)).save(target)
            # A retake gets its own key so cached features never come from the original photo.
            out.append({**item, "padded": str(target), "owner": item["sha256"], "sha256": f"{item['sha256']}_r{k}"})
    return out

test_items, phone_items = by_split["test"], by_split["phone"]
retake_set = retake_items(test_items)
truth = {"test": arrays(test_items)[1], "phone": arrays(phone_items)[1], "retake": arrays(retake_set)[1]}
retake_owner = np.asarray([test_items.index(next(t for t in test_items if t["sha256"] == r["owner"]))
                           for r in retake_set])

def consistency(clean, retaken):
    same = retaken.argmax(1) == clean.argmax(1)[retake_owner]
    return {"consistency": float(same.mean()),
            "photos_never_flipping": float(np.mean([same[retake_owner == i].all() for i in range(len(test_items))]))}

def evaluate_cnn(model):
    out = {}
    for mode in ("single_view", "mirror_average"):
        index = 0 if mode == "single_view" else 1
        clean = predict(model, test_items)[index]
        retaken = predict(model, retake_set)[index]
        phone = predict(model, phone_items)[index]
        out[mode] = {"test": metrics(truth["test"], clean), "retakes": {**metrics(truth["retake"], retaken),
                     **consistency(clean, retaken)},
                     "phone_cotton": {"accuracy": float((phone.argmax(1) == truth["phone"]).mean()),
                                      "predictions": [CLASS_NAMES[i] for i in phone.argmax(1)]}}
    return out

final = {"cnn": {}}
baseline_path = REPO / "model" / "mobilenetv2_6class" / "mobilenetv2_v3_refit.keras"
final["cnn"]["production_mobilenetv2_v3_refit"] = evaluate_cnn(tf.keras.models.load_model(baseline_path, compile=False))
for key in BACKBONES:
    final["cnn"][key] = evaluate_cnn(cnn_model(key))
    tf.keras.backend.clear_session()

def evaluate_glcm(model):
    clean = model.predict_proba(glcm["test"])
    retake_x = glcm_matrix(retake_set)
    return {"test": metrics(truth["test"], clean), "retakes": {**metrics(truth["retake"], model.predict_proba(retake_x)),
            **consistency(clean, model.predict_proba(retake_x))},
            "phone_cotton": {"accuracy": float((model.predict(glcm["phone"]) == truth["phone"]).mean()),
                             "predictions": [CLASS_NAMES[i] for i in model.predict(glcm["phone"])]}}

final["glcm_decision_tree"] = evaluate_glcm(glcm_dt)
final["glcm_svm"] = evaluate_glcm(glcm_svm)

print(f"{'model':34s} {'test acc':>9s} {'macro-F1':>9s} {'retake acc':>11s} {'consistency':>12s} {'phone':>6s}")
for key, result in final["cnn"].items():
    for mode in ("single_view", "mirror_average"):
        r = result[mode]
        print(f"{key + ' · ' + mode:34s} {r['test']['accuracy']:9.2%} {r['test']['macro_f1']:9.3f} "
              f"{r['retakes']['accuracy']:11.2%} {r['retakes']['consistency']:12.2%} {r['phone_cotton']['accuracy']:6.0%}")
for key in ("glcm_decision_tree", "glcm_svm"):
    r = final[key]
    print(f"{key:34s} {r['test']['accuracy']:9.2%} {r['test']['macro_f1']:9.3f} {r['retakes']['accuracy']:11.2%} "
          f"{r['retakes']['consistency']:12.2%} {r['phone_cotton']['accuracy']:6.0%}")
(RUN / "fabric_test_results.json").write_text(json.dumps(final, indent=2))
""")

md(r"""
## Objective 2 · Care recommendation: Decision Tree vs Random Forest vs SVM
Each photo becomes one case: its fabric, the YOLOv8 damage findings (the API's own detector code and `best.pt`),
colour features and GLCM texture summary. The label is the referenced laundry mode from `care_model.reference_laundry_mode`.
Splits follow the garment-grouped V3 split. The test split is evaluated twice: with the true fabric, and end to end
with the fabric predicted by the selected CNN.
""")

code(r"""
# 8. Build care cases with the API's damage detector.
os.environ.setdefault("FABRIC_CLASSIFIER", "mobilenet")
import fabricare_api

def damage_for(item):
    cache = STAGE / "damage" / f"{item['sha256']}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    cache.parent.mkdir(exist_ok=True)
    detections = fabricare_api.detect_damage(item["full"], 0, "front")
    cache.write_text(json.dumps(detections))
    return detections

def care_rows(items, fabrics=None):
    out = []
    for n, item in enumerate(items):
        fabric = fabrics[n] if fabrics is not None else item["fabric"]
        vector = glcm_matrix([item])[0]
        out.append(care_model.feature_row(fabric, care_model.damage_features(damage_for(item)), item["color"],
                                          care_model.glcm_summary(vector)))
        if n and n % 500 == 0:
            print("care cases", n, flush=True)
    return out

care = {split: care_rows(by_split[split]) for split in SPLITS}
care_x = {s: np.vstack([care_model.feature_vector(r) for r in care[s]]) for s in SPLITS}
care_y = {s: np.asarray([care_model.CLASSES.index(care_model.reference_laundry_mode(r)[0]) for r in care[s]])
          for s in SPLITS}
for s in SPLITS:
    print(s, dict(Counter(care_model.CLASSES[i] for i in care_y[s])))
""")

code(r"""
# 9. Train, select on validation, and test the three care models.
from sklearn.ensemble import RandomForestClassifier

CARE_IDS = list(range(len(care_model.CLASSES)))
def care_metrics(truth, guess):
    report = classification_report(truth, guess, labels=CARE_IDS, target_names=list(care_model.CLASSES),
                                   output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(truth, guess)),
            "macro_f1": float(f1_score(truth, guess, labels=CARE_IDS, average="macro", zero_division=0)),
            "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                          for c in care_model.CLASSES},
            "confusion_matrix_actual_by_predicted": confusion_matrix(truth, guess, labels=CARE_IDS).tolist()}

care_specs = {
    "Decision Tree": (lambda **p: DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **p),
                      {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 3, 5]}),
    "Random Forest": (lambda **p: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                         random_state=SEED, n_jobs=-1, **p),
                      {"max_depth": [8, 16, None], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.5]}),
    "SVM": (lambda **p: make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True,
                                                            random_state=SEED, **p)),
            {"C": [1, 10, 100], "gamma": ["scale", 0.03]}),
}
# Fit on every class index so classes_ always lists all five modes in order.
def complete(x, y):
    absent = [c for c in CARE_IDS if c not in set(y)]
    return (np.vstack([x, np.zeros((len(absent), x.shape[1]))]), np.concatenate([y, absent])) if absent else (x, y)

predicted_fabric = [CLASS_NAMES[i] for i in predict(cnn_model(best_cnn), test_items)[1].argmax(1)]
care_end_to_end_x = np.vstack([care_model.feature_vector(r) for r in care_rows(test_items, predicted_fabric)])
care_results, care_models = {}, {}
for name, (make, grid) in care_specs.items():
    trials = []
    for params in ParameterGrid(grid):
        model = make(**params).fit(*complete(care_x["train"], care_y["train"]))
        trials.append((f1_score(care_y["valid"], model.predict(care_x["valid"]), average="macro", zero_division=0),
                       params, model))
    score, params, model = max(trials, key=lambda t: t[0])
    care_models[name] = model
    care_results[name] = {"params": params, "valid_macro_f1": float(score),
                          "test_true_fabric": care_metrics(care_y["test"], model.predict(care_x["test"])),
                          "test_predicted_fabric": care_metrics(care_y["test"], model.predict(care_end_to_end_x))}
    r = care_results[name]
    print(f"{name:14s} valid F1 {score:.3f} | test acc {r['test_true_fabric']['accuracy']:.2%} (true fabric), "
          f"{r['test_predicted_fabric']['accuracy']:.2%} (CNN fabric)", flush=True)
best_care = max(care_results, key=lambda k: care_results[k]["valid_macro_f1"])
print("Selected care model on validation:", best_care)
""")

md(r"""
## Export for the FabriCare API
""")

code(r"""
# 10. Save the selected models and a full report, then download a zip.
import importlib.metadata
export = RUN / "export"
if export.exists():
    shutil.rmtree(export)
(export / "fabric_cnn").mkdir(parents=True)
(export / "glcm_models").mkdir()
(export / "care").mkdir()

shutil.copy2(RUN / "cnn" / best_cnn / "best.keras", export / "fabric_cnn" / "fabric_cnn.keras")
(export / "fabric_cnn" / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2))
best_test = final["cnn"][best_cnn]["mirror_average"]["test"]
(export / "fabric_cnn" / "model_info.json").write_text(json.dumps({
    "name": f"{BACKBONES[best_cnn][0]} (V4)", "backbone": best_cnn,
    "validation_accuracy": round(cnn_results[best_cnn]["valid"]["accuracy"] * 100, 2),
    "test_accuracy": round(best_test["accuracy"] * 100, 2), "test_macro_f1": round(best_test["macro_f1"], 4),
    "inference": "mirror_average", "input": "224x224 RGB, 0-255, gray (127) padding"}, indent=2))
shutil.copy2(RUN / "glcm" / "glcm_decision_tree.joblib", export / "glcm_models" / "glcm_decision_tree.joblib")
shutil.copy2(RUN / "glcm" / "glcm_svm.joblib", export / "glcm_models" / "glcm_svm.joblib")
(export / "glcm_models" / "metrics.json").write_text(json.dumps({
    api_key: {"test_accuracy": round(final[key]["test"]["accuracy"] * 100, 2),
              "test_macro_f1": round(final[key]["test"]["macro_f1"], 4)}
    for api_key, key in (("svm", "glcm_svm"), ("glcm_dt", "glcm_decision_tree"))}, indent=2))
for name, model in care_models.items():
    bundle = {"model": model, "feature_names": care_model.FEATURE_NAMES, "classes": list(care_model.CLASSES),
              "name": name, "metrics": care_results[name]}
    joblib.dump(bundle, export / "care" / f"care_{name.lower().replace(' ', '_')}.joblib")
shutil.copy2(export / "care" / f"care_{best_care.lower().replace(' ', '_')}.joblib", export / "care" / "care_recommender.joblib")

report = {
    "created": datetime.now(timezone.utc).isoformat(), "dataset": V3_URL, "manifest_sha256": MANIFEST_SHA256,
    "counts": {s: len(by_split[s]) for s in (*SPLITS, "phone")}, "polyester_additions": len(additions),
    "selected_cnn": best_cnn, "cnn_validation": {k: {kk: v[kk] for kk in ("valid", "valid_single_view")}
                                                 for k, v in cnn_results.items()},
    "glcm_selection": {"decision_tree": glcm_dt_info, "svm": glcm_svm_info},
    "fabric_test": final, "care": care_results, "selected_care_model": best_care,
    "versions": {n: importlib.metadata.version(n) for n in ("tensorflow", "keras", "scikit-learn", "scikit-image",
                                                            "joblib", "ultralytics", "numpy")},
}
(export / "training_report.json").write_text(json.dumps(report, indent=2))
archive = shutil.make_archive(str(RUN / "fabricare_v4_export"), "zip", export)
print("Saved to Drive:", export)
print("Zip for the API:", archive)
print("Selected CNN:", best_cnn, "| test accuracy", f"{best_test['accuracy']:.2%}", "| macro-F1", round(best_test["macro_f1"], 4))
print("Selected care model:", best_care)
""")


def main():
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [{**cell, "source": cell["source"].splitlines(keepends=True)} for cell in CELLS],
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT), "with", len(CELLS), "cells")


if __name__ == "__main__":
    main()
