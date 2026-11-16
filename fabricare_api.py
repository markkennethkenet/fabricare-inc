from flask import Flask, request, jsonify, g
from flask_cors import CORS
from pathlib import Path
import base64
import io
import tempfile
import json
import os
import platform
from threading import Lock
from time import monotonic
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen
import sklearn
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO
from werkzeug.exceptions import RequestEntityTooLarge

from assessment import CONFIG, SUPPORT, assess, normalize_user_inputs
from image_processing import quality_for, framing_hint, merge_detections, aggregate_fabrics
from auth_routes import auth_bp
from database import init_schema, ping_database
from model_runtime import (
    FabricCNN, FabricDecisionTree, FabricEnsemble, FabricGLCM72, FabricMobileNet,
    FabricMobileNetV2, SIX_FABRIC_LABELS, extract_glcm72_features, load_fabric_model, read_model_info,
)
from model_analysis import model_result, validate_image_file
import care_model
from assessment import REFERENCES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CONFIG["input"]["maximum_upload_bytes"]
CORS(app)
app.register_blueprint(auth_bp)

BASE_DIR = Path(__file__).resolve().parent

FABRIC_MODEL_PATH = BASE_DIR / "model" / "fabric_glcm_decision_tree.pkl"
FABRIC_DECISION_TREE_PATH = BASE_DIR / "model" / "best_decision_tree.joblib"
FABRIC_LABELS_PATH = BASE_DIR / "model" / "fabric_labels.json"
DAMAGE_MODEL_PATH = BASE_DIR / "model" / "best.pt"
# A retrained CNN export (V4 notebook) replaces the V3 MobileNetV2 when present.
FABRIC_CNN_DIR = BASE_DIR / "model" / "fabric_cnn"
LEGACY_CNN_DIR = BASE_DIR / "model" / "mobilenetv2_6class"
CARE_MODEL_PATH = BASE_DIR / "model" / "care" / "care_recommender.joblib"


def cnn_metadata():
    """CNN name and metrics without loading TensorFlow."""
    if (FABRIC_CNN_DIR / "model_info.json").is_file():
        info = read_model_info(FABRIC_CNN_DIR)
        return SimpleNamespace(name=info["name"], validation_accuracy=info.get("validation_accuracy"),
                               test_accuracy=info.get("test_accuracy"), test_macro_f1=info.get("test_macro_f1"))
    return FabricMobileNetV2


fabric_backend = os.environ.get("FABRIC_CLASSIFIER", "mobilenet")
if fabric_backend == "decision_tree":
    fabric_model = FabricDecisionTree(FABRIC_DECISION_TREE_PATH)
    fabric_labels = fabric_model.labels
    fabric_model_name = fabric_model.name
elif fabric_backend == "mobilenet":
    fabric_model = cnn_metadata()  # Metadata is available without loading TensorFlow.
    fabric_labels = list(SIX_FABRIC_LABELS)
    fabric_model_name = fabric_model.name
elif fabric_backend == "mobilenet_legacy":
    fabric_model = FabricMobileNet(BASE_DIR / "model" / "mobilenet_20261110")
    fabric_labels = fabric_model.labels
    fabric_model_name = fabric_model.name
elif fabric_backend == "ensemble":
    fabric_model = FabricEnsemble(BASE_DIR / "model" / "validation_development")
    fabric_labels = fabric_model.labels
    fabric_model_name = fabric_model.name
elif fabric_backend == "legacy":
    fabric_model = load_fabric_model(FABRIC_MODEL_PATH)
    fabric_labels = json.loads(FABRIC_LABELS_PATH.read_text(encoding="utf-8"))
    fabric_model_name = "GLCM + Decision Tree"
else:
    raise RuntimeError("FABRIC_CLASSIFIER must be mobilenet, decision_tree, mobilenet_legacy, legacy, or ensemble.")

try:
    damage_model = YOLO(str(DAMAGE_MODEL_PATH))
except Exception:
    app.logger.exception("Could not load the damage detector")
    damage_model = None
damage_model_lock = Lock()
classifier_load_lock = Lock()
classifier_cache = {}
selectable_classifiers = ("cnn", "svm", "glcm_dt")


class SelectedClassifierError(RuntimeError):
    pass


def load_selected_classifier(model_type):
    with classifier_load_lock:
        if model_type not in classifier_cache:
            if model_type == "cnn":
                model = (FabricCNN(FABRIC_CNN_DIR) if (FABRIC_CNN_DIR / "model_info.json").is_file()
                         else FabricMobileNetV2(LEGACY_CNN_DIR))
            else:
                model = FabricGLCM72(BASE_DIR / "model" / "glcm_models", model_type)
            classifier_cache[model_type] = model
        return classifier_cache[model_type]


def selected_classifier():
    model_type = request.form.get("model_type")
    if model_type is None:
        if fabric_backend != "mobilenet":
            return None, None
        model_type = "cnn"
    if model_type not in selectable_classifiers:
        raise ValueError("Invalid model_type. Choose cnn, svm, or glcm_dt.")
    try:
        return model_type, load_selected_classifier(model_type)
    except FileNotFoundError as error:
        app.logger.warning("Missing fabric classifier asset for %s", model_type)
        raise SelectedClassifierError(str(error)) from None
    except Exception:
        app.logger.exception("Could not load fabric classifier %s", model_type)
        name = (cnn_metadata().name if model_type == "cnn"
                else FabricGLCM72.specifications[model_type][0])
        raise SelectedClassifierError(
            f"{name} could not load. Check the trained model files and API environment."
        ) from None


def classify_scan_image(image_path, model):
    name = model.name if model is not None else fabric_model_name
    try:
        return model.classify(image_path) if model is not None else classify_fabric(image_path)
    except Exception:
        app.logger.exception("Fabric inference failed with %s", name)
        raise SelectedClassifierError(
            f"{name} could not classify the photo. Try another image."
        ) from None


def scan_model_metadata(model_type, model):
    if model is None:
        return {"fabric_model": fabric_model_name, **fabric_model_metrics()}
    return {
        "model_type": model_type,
        "fabric_model": model.name,
        "fabric_validation_accuracy": getattr(model, "validation_accuracy", None),
        "fabric_test_accuracy": model.test_accuracy,
        "fabric_test_macro_f1": model.test_macro_f1,
    }


def second_fabric_opinion(records, primary_type, primary_label):
    second_type = "svm" if primary_type == "cnn" else "cnn"
    try:
        classifier = load_selected_classifier(second_type)
        predictions = [
            {**classifier.classify(record["path"]), "view": record["view"],
             "quality": record["quality"]}
            for record in records
        ]
        label = aggregate_fabrics(predictions)["fabric_type"]
    except Exception:
        app.logger.exception("Second fabric model could not assess the scan")
        return {"status": "unavailable"}
    if "uncertain" in (primary_label, label):
        status = "inconclusive"
    else:
        status = "agrees" if label == primary_label else "disagrees"
    return {"status": status, "model": classifier.name, "fabric_type": label}


care_model_lock = Lock()
care_model_cache = {}


def load_care_recommender():
    """The trained care model (Decision Tree, Random Forest or SVM), or None to use the reference rules."""
    with care_model_lock:
        if "model" not in care_model_cache:
            try:
                care_model_cache["model"] = (care_model.CareRecommender(CARE_MODEL_PATH)
                                             if CARE_MODEL_PATH.is_file() else None)
            except Exception:
                app.logger.exception("Could not load the care recommendation model")
                care_model_cache["model"] = None
        return care_model_cache["model"]


def ai_care_for(records, fabric_type, detections, user_inputs):
    """Laundry-mode suggestion and automated care alerts from the fabric, condition and image features."""
    front = next((record for record in records if record["view"] == "front"), records[0])
    try:
        row = care_model.feature_row(
            fabric_type, care_model.damage_features(detections), care_model.color_features(front["path"]),
            care_model.glcm_summary(extract_glcm72_features(front["path"])[0]))
    except Exception:
        app.logger.exception("Care features could not be extracted")
        return {"status": "unavailable"}
    reference_mode, reasons = care_model.reference_laundry_mode(row)
    recommender = load_care_recommender() if fabric_type in care_model.FABRICS else None
    if recommender is not None:
        prediction = recommender.predict(row)
        status, model_name = "model_suggestion", recommender.name
    else:
        prediction = {"laundry_mode": reference_mode, "confidence": None, "probabilities": {}}
        status, model_name = "rule_based_guidance", "Reference care rules"
    mode = prediction["laundry_mode"]
    label_entered = user_inputs["care_label"]["wash"] != "unknown"
    return {
        "status": status, "model": model_name, "laundry_mode": mode,
        "label": care_model.LAUNDRY_MODES[mode], "detail": care_model.MODE_DETAILS[mode],
        "confidence": prediction["confidence"], "probabilities": prediction["probabilities"],
        "reference_mode": reference_mode, "agrees_with_reference": mode == reference_mode,
        "reasons": [{"text": text, "reference_ids": refs} for text, refs in reasons],
        "alerts": care_model.care_alerts(row),
        "care_label_entered": label_entered,
        "note": ("Your entered care label takes priority over this suggestion." if label_entered
                 else "A suggestion from the scan. Check the garment's care label before washing."),
        "features": {name: round(value, 4) for name, value in row.items() if not name.startswith("fabric_")},
    }


def with_care_references(references, ai_care):
    """Add the sources cited by the care suggestion and alerts to the response references."""
    cited = {ref for entry in [*ai_care.get("reasons", []), *ai_care.get("alerts", [])]
             for ref in entry["reference_ids"]}
    known = {entry["id"] for entry in references}
    extra = [{"id": ref, "title": REFERENCES[ref][0], "url": REFERENCES[ref][1]}
             for ref in sorted(cited - known) if ref in REFERENCES]
    return [*references, *extra]


def extract_glcm_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Cannot read image.")

    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = (gray / 32).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )

    features = []

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    for prop in properties:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())

    features.append(np.mean(gray))
    features.append(np.std(gray))

    return np.array(features).reshape(1, -1)


def fabric_label(value):
    if isinstance(value, (str, np.str_)) and value in fabric_labels:
        return str(value)
    return fabric_labels[int(value)]


def classify_fabric(image_path):
    if fabric_backend == "mobilenet":
        return load_selected_classifier("cnn").classify(image_path)
    if fabric_backend != "legacy":
        return fabric_model.classify(image_path)
    features = extract_glcm_features(image_path)
    prediction = fabric_model.predict(features)[0]
    label = fabric_label(prediction)
    if hasattr(fabric_model, "predict_proba"):
        probabilities = fabric_model.predict_proba(features)[0]
        classes = getattr(fabric_model, "classes_", range(len(probabilities)))
        scores = {fabric_label(value): float(probability)
                  for value, probability in zip(classes, probabilities)}
    else:
        return {"fabric_type": label, "confidence": None, "probabilities": {}, "score_kind": "unavailable"}
    return {"fabric_type": label, "confidence": scores[label] * 100, "probabilities": scores}


def detect_damage(image_path, image_index=0, view="front"):
    # Ultralytics updates the shared predictor's options between passes.
    with damage_model_lock:
        return _detect_damage(image_path, image_index, view)


def _detect_damage(image_path, image_index, view):
    policy = CONFIG["detection"]
    result = damage_model.predict(source=image_path, imgsz=policy["image_size"],
                                  classes=None, conf=policy["minimum_confidence"],
                                  save=False, verbose=False)[0]
    image_height, image_width = result.orig_shape
    detections = []
    for box in result.boxes:
        detection = _damage_detection(box, (0, 0, image_width, image_height),
                                      (image_width, image_height), image_index, view,
                                      len(detections), False)
        if detection is not None:
            detections.append(detection)
    confirmed = {detection["type"] for detection in detections
                 if detection["confidence"] >= policy["detail_skip_percent"]}
    passes = []
    detail_classes = {label: index for index, label in damage_model.names.items()
                      if label in {"stain", "hole"} and label not in confirmed}
    if max(image_height, image_width) >= policy["detail_image_size"] and detail_classes:
        image = result.orig_img
        if "stain" in detail_classes:
            detail = damage_model.predict(source=image, imgsz=policy["detail_image_size"],
                                          classes=[detail_classes["stain"]],
                                          conf=policy["minimum_confidence"],
                                          save=False, verbose=False)[0]
            passes.append((detail, 0, 0, False))
        if "hole" in detail_classes:
            tile_height, tile_width = image_height // 2, image_width // 2
            if min(tile_height, tile_width) >= 2:
                offsets = [(x, y)
                           for y in (0, (image_height - tile_height) // 2, image_height - tile_height)
                           for x in (0, (image_width - tile_width) // 2, image_width - tile_width)]
                tiles = [image[y:y + tile_height, x:x + tile_width] for x, y in offsets]
                details = damage_model.predict(source=tiles, imgsz=policy["image_size"],
                                               classes=[detail_classes["hole"]],
                                               conf=policy["tile_minimum_confidence"],
                                               save=False, verbose=False)
                passes.extend((detail, x, y, True) for detail, (x, y) in zip(details, offsets))

    for detail, x, y, tiled in passes:
        height, width = detail.orig_shape
        for box in detail.boxes:
            detection = _damage_detection(box, (x, y, width, height), (image_width, image_height),
                                          image_index, view, len(detections), tiled)
            if detection is not None:
                detections.append(detection)
    return detections


def _damage_detection(box, region, image_size, image_index, view, index, tiled):
    x, y, width, height = region
    image_width, image_height = image_size
    policy = CONFIG["detection"]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    # Ignore crop proposals spanning most of a tile. These often follow folds or edges.
    if tiled and max((x2 - x1) / width, (y2 - y1) / height) > policy["tile_maximum_box_span"]:
        return None
    bbox = [(x + x1) / image_width, (y + y1) / image_height,
            (x2 - x1) / image_width, (y2 - y1) / image_height]
    area_ratio = bbox[2] * bbox[3]
    confidence = float(box.conf.item()) * 100
    label = damage_model.names[int(box.cls.item())]
    if label not in SUPPORT or area_ratio <= 0 or (
        label == "hole" and area_ratio > policy["hole_maximum_photo_area"]
    ):
        return None
    return {
        "id": f"{image_index}:{index}", "type": label, "label": label,
        "confidence": round(confidence, 2),
        "status": "Possible",
        "severity": None, "area_ratio": area_ratio,
        "bbox": bbox, "image_index": image_index, "view": view,
    }


def uploaded_views():
    """Legacy: image=<file>. New: images=<repeated files>, views=<JSON array>."""
    legacy = "image" in request.files and "images" not in request.files
    if legacy:
        return [("front", request.files["image"])], True
    files = request.files.getlist("images")
    try:
        views = json.loads(request.form.get("views", "[]"))
    except (ValueError, TypeError):
        raise ValueError("views must be a JSON array")
    if not isinstance(views, list) or len(files) != len(views) or not files:
        raise ValueError("Provide images and a matching views array")
    if len(files) > CONFIG["input"]["maximum_images"]:
        raise ValueError("At most three images are supported")
    if any(not isinstance(view, str) or view not in CONFIG["input"]["views"] for view in views):
        raise ValueError("Unsupported image view")
    if len(set(views)) != len(views) or "front" not in views:
        raise ValueError("Front is required and each view must be unique")
    return list(zip(views, files)), False


def prepare_images(uploads, directory, *, validate_format=False, guide_framing=False):
    records = []
    for index, (view, upload) in enumerate(uploads):
        path = str(Path(directory) / f"image_{index}.jpg")
        upload.save(path)
        if validate_format:
            validate_image_file(path)
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot read {view} image. Upload a valid photo.")
        height, width = image.shape[:2]
        quality = quality_for(image)
        if guide_framing:
            quality["framing_hint"] = framing_hint(image, view)
        records.append({"image_index": index, "view": view, "width": width,
                        "height": height, "quality": quality, "path": path})
    return records


def public_image(record):
    return {key: value for key, value in record.items() if key != "path"}


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "Images exceed the upload limit. Use smaller photos."}), 413


@app.before_request
def start_scan_timer():
    if request.method == "POST" and request.path in {"/quality", "/predict/hybrid", "/analyze"}:
        g.scan_started = monotonic()


@app.after_request
def log_scan_response(response):
    if hasattr(g, "scan_started"):
        log = app.logger.warning if response.status_code >= 400 else app.logger.info
        log(
            "Scan POST %s status=%d bytes=%s elapsed_ms=%d",
            request.path, response.status_code, request.content_length,
            (monotonic() - g.scan_started) * 1000,
        )
    return response


@app.route("/quality", methods=["POST"])
def check_quality():
    try:
        uploads, _ = uploaded_views()
        with tempfile.TemporaryDirectory() as directory:
            records = prepare_images(uploads, directory, validate_format=True, guide_framing=True)
            return jsonify({"per_image": [public_image(r) for r in records]})
    except RequestEntityTooLarge:
        raise
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Quality check failed")
        return jsonify({"error": "Could not check image quality. Try again."}), 500


def damage_labels():
    return list(damage_model.names.values()) if damage_model is not None else []


def analyze_upload(*, hybrid):
    try:
        uploads, legacy = uploaded_views()
        try:
            user_inputs = normalize_user_inputs(json.loads(request.form.get("user_inputs", "{}")))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid garment history: {error}") from None
        with tempfile.TemporaryDirectory() as directory:
            records = prepare_images(uploads, directory, validate_format=True)
            failed_quality = any(not r["quality"]["passed"] for r in records)
            if failed_quality and (hybrid or not legacy) and request.form.get("allow_low_quality") != "true":
                return jsonify({"error": "Review image quality and retake or continue anyway.",
                                "quality_requires_confirmation": True,
                                "per_image": [public_image(r) for r in records]}), 422
            model_type, model = selected_classifier()
            detections = []
            for record in records:
                record.update(classify_scan_image(record["path"], model))
                raw = []
                record["damage_available"] = damage_model is not None
                if record["damage_available"]:
                    try:
                        raw = detect_damage(record["path"], record["image_index"], record["view"])
                    except Exception:
                        app.logger.exception("Damage inference failed")
                        record["damage_available"] = False
                record["raw_detections"] = raw
                detections.extend(raw)
            fabric = aggregate_fabrics(records)
            opinion = None
            if hybrid and model_type and request.form.get("ai_assist") == "true":
                opinion = second_fabric_opinion(records, model_type, fabric["fabric_type"])
            needs_model_review = opinion and opinion["status"] in {"disagrees", "inconclusive"}
            merged = merge_detections(detections)
            detector_available = any(record["damage_available"] for record in records)
            assessment = assess(
                fabric["fabric_type"], merged, user_inputs,
                fabric_confidence=None if needs_model_review else fabric["fabric_confidence"],
                detector_available=detector_available, quality_limited=failed_quality,
                coverage_limited=not all(record["damage_available"] for record in records),
            )
            ai_care = ai_care_for(records, fabric["fabric_type"], merged, user_inputs)
            assessment["references"] = with_care_references(assessment["references"], ai_care)
            metadata = scan_model_metadata(model_type, model)
            analysis = model_result(
                {"fabric_type": fabric["fabric_type"], "confidence": fabric["fabric_confidence"],
                 "score_kind": records[0].get("score_kind", "model_score")},
                merged, model.labels if model is not None else fabric_labels,
                detector_available, damage_labels(),
            )
            return jsonify({
                "schema_version": 3, **fabric, "damages": merged, **metadata, **analysis, **assessment,
                **({"fabric_second_opinion": opinion} if opinion is not None else {}),
                "ai_care": ai_care,
                "per_image": [public_image(record) for record in records], "quality_override": failed_quality,
                "model_versions": {"fabric": metadata["fabric_model"],
                                   "damage": DAMAGE_MODEL_PATH.name if detector_available else None},
            })
    except RequestEntityTooLarge:
        raise
    except SelectedClassifierError as error:
        return jsonify({"error": str(error)}), 503
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Scan analysis failed")
        return jsonify({"error": "Analysis failed. Please try another image."}), 500


@app.post("/analyze")
def analyze():
    return analyze_upload(hybrid=False)


@app.post("/predict/hybrid")
def predict_hybrid():
    return analyze_upload(hybrid=True)


@app.post("/vision-opinion")
def vision_opinion():
    try:
        uploads, _ = uploaded_views()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    if len(uploads) != 1 or uploads[0][0] != "front":
        return jsonify({"error": "Provide one front photo for Qwen."}), 400
    try:
        with Image.open(uploads[0][1].stream) as source:
            if source.format not in {"JPEG", "PNG"} or source.width * source.height > 24_000_000:
                raise ValueError("Unsupported photo.")
            photo = ImageOps.exif_transpose(source).convert("RGB")
        photo.thumbnail((960, 960), Image.Resampling.LANCZOS)
        with io.BytesIO() as encoded:
            photo.save(encoded, format="JPEG", quality=92)
            image = base64.b64encode(encoded.getvalue()).decode("ascii")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return jsonify({"error": "Use a readable JPG or PNG front photo."}), 400
    labels = [*SIX_FABRIC_LABELS, "uncertain"]
    payload = {
        "model": "qwen3-vl:2b", "stream": False,
        "messages": [{
            "role": "user", "images": [image],
            "content": (
                "Identify the fabric of the main garment. Choose cotton, denim, linen, polyester, "
                "silk, wool, or uncertain. Choose uncertain when visual clues are insufficient. "
                'Keep reasoning brief and return a JSON object such as {"fabric":"uncertain"}.'
            ),
        }],
        "format": {
            "type": "object",
            "properties": {"fabric": {"type": "string", "enum": labels}},
            "required": ["fabric"],
        },
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1536},
    }
    query = Request("http://127.0.0.1:11434/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
    try:
        with urlopen(query, timeout=90) as response:
            answer = json.load(response)
    except (OSError, URLError, ValueError):
        return jsonify({"error": "Qwen is unavailable. Check that Ollama is running on the PC."}), 503
    try:
        content = answer["message"]["content"]
        label = json.loads(content)["fabric"] if content.strip() else None
    except (KeyError, ValueError, AttributeError, TypeError):
        label = None
    return jsonify({
        "model": "Qwen3-VL 2B",
        "fabric_type": label if label in labels else None,
    })


@app.post("/review-damage")
def review_damage():
    """Recalculate the assessment from the findings the user can see."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Provide damage review details."}), 400
    fabric_type = data.get("fabric_type")
    findings = data.get("findings")
    if not isinstance(fabric_type, str) or not fabric_type.strip() or not isinstance(findings, list):
        return jsonify({"error": "Provide the fabric type and reviewed findings."}), 400
    observations = []
    reviewed = False
    for finding in findings:
        if (not isinstance(finding, dict) or "confirmed" not in finding
                or finding["confirmed"] is not None and type(finding["confirmed"]) is not bool):
            return jsonify({"error": "Mark each finding as confirmed, dismissed, or pending."}), 400
        decision = finding["confirmed"]
        reviewed |= decision is not None
        kind, area = finding.get("type"), finding.get("area_ratio")
        if not isinstance(kind, str) or not kind.strip() or type(area) not in (int, float) or not 0 <= area <= 1:
            return jsonify({"error": "Invalid damage finding."}), 400
        if decision is False:
            continue
        finding_id, view = finding.get("id"), finding.get("view")
        if finding_id is not None and (not isinstance(finding_id, str) or len(finding_id) > 64):
            return jsonify({"error": "Invalid finding ID."}), 400
        if view is not None and view not in CONFIG["input"]["views"]:
            return jsonify({"error": "Invalid photo view."}), 400
        box = finding.get("bbox")
        if box is not None and (not isinstance(box, list) or len(box) != 4
                                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in box)):
            return jsonify({"error": "Invalid finding location."}), 400
        observations.append({"type": kind, "area_ratio": area, "id": finding_id,
                             "view": view, "bbox": box, "confirmed": decision is True})
    if not reviewed:
        return jsonify({"error": "Confirm or dismiss at least one finding."}), 400
    try:
        confidence = data.get("fabric_confidence")
        if confidence is not None and (type(confidence) not in (int, float) or not 0 <= confidence <= 100):
            raise ValueError("Invalid fabric model score.")
        quality_limited = data.get("quality_limited", False)
        coverage_limited = data.get("coverage_limited", False)
        detector_available = data.get("detector_available", True)
        if any(type(flag) is not bool for flag in (quality_limited, coverage_limited, detector_available)):
            raise ValueError("Invalid image quality flag.")
        return jsonify(assess(fabric_type, observations, data.get("user_inputs"), reviewed=True,
                              fabric_confidence=confidence, quality_limited=quality_limited,
                              detector_available=detector_available, coverage_limited=coverage_limited))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


def fabric_model_metrics():
    metrics = {"fabric_validation_accuracy": getattr(fabric_model, "validation_accuracy", None)
               if fabric_backend in {"mobilenet", "decision_tree"} else None}
    if fabric_backend == "mobilenet":
        metrics.update(fabric_test_accuracy=fabric_model.test_accuracy,
                       fabric_test_macro_f1=fabric_model.test_macro_f1)
    return metrics


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "FabricCare API is running.",
        "fabric_model": fabric_model_name,
        **fabric_model_metrics(),
        "damage_model": "YOLOv8" if damage_model is not None else None,
        "fabric_labels": fabric_labels,
        "damage_labels": damage_labels(),
        "runtime": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "database": "MySQL",
        # Keep the discovery endpoint fast; API clients need model metadata even
        # when MySQL is unavailable. Use /health/ready for the database probe.
        "database_connected": None,
        "database_error": "Not checked; use /health/ready.",
    })


@app.route("/health/ready", methods=["GET"])
def readiness():
    try:
        ping_database()
    except Exception:
        return jsonify({"ready": False, "database_connected": False}), 503
    return jsonify({"ready": True, "database_connected": True})


if __name__ == "__main__":
    try:
        init_schema()
        print("MySQL schema ready.")
    except Exception as e:
        print(f"MySQL setup failed: {e}")
        print("Check .env (copy from .env.example) and that MySQL is running.")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
