"""Fabric care recommendation: shared features, reference labels, alerts and the trained model.

The training notebook (docs/fabricare_v4_training_colab.ipynb) and the API both import this
module, so a scan is described by exactly the same features in training and in use.

Inputs follow objective 2: the identified fabric type, the detected fabric condition (damage
findings) and extracted image features (colour and GLCM texture). The reference laundry mode
comes from the textile-care sources the app already cites. A model trained on these labels
learns that reference; its accuracy measures agreement with the reference rules, not
laboratory-verified care outcomes. A garment's own care label always takes precedence.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

FABRICS = ("cotton", "denim", "linen", "polyester", "silk", "wool")
DAMAGE_TYPES = ("hole", "needlemark", "pilling", "stain")
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
# Detector score (percent) a finding needs before it changes care, health or lifespan on its own.
# Chosen from the V4 run on the mostly undamaged V3 photos: at these levels the YOLOv8 detector flags
# holes on 3.4% of photos, stains on 4.2%, pilling on 5.0% and needle marks on under 0.1%, against 35.1%,
# 15.7%, 9.2% and 3.0% at a 40% score. Weaker findings remain visible for the user to review, and a
# finding the user confirms always counts.
DAMAGE_MIN_CONFIDENCE = {"hole": 90.0, "stain": 70.0, "pilling": 70.0, "needlemark": 60.0}


def finding_counts(kind, confidence, confirmed=False):
    """True when a damage finding should drive care, health or lifespan automatically."""
    if confirmed:
        return True
    threshold = DAMAGE_MIN_CONFIDENCE.get(kind)
    return threshold is not None and isinstance(confidence, (int, float)) and confidence >= threshold
# Colour thresholds on 0–255 luminance and 0–1 saturation.
DARK_LUMINANCE = 90.0
DARK_PIXEL_LUMINANCE = 70.0
DARK_FRACTION = 0.5
STRONG_SATURATION = 0.45

FEATURE_NAMES = (
    [f"fabric_{name}" for name in FABRICS]
    + [f"{kind}_{stat}" for kind in DAMAGE_TYPES for stat in ("count", "max_area")]
    + ["luminance_mean", "luminance_std", "saturation_mean", "dark_fraction"]
    + [f"glcm_{prop}" for prop in GLCM_PROPERTIES]
)

LAUNDRY_MODES = {
    "machine_normal_warm": "Machine wash · normal cycle · warm",
    "machine_gentle_cold": "Machine wash · gentle cycle · cold",
    "hand_wash_cold": "Hand wash · cold water",
    "professional_clean": "Professional cleaning recommended",
    "repair_first": "Repair before washing",
}
MODE_DETAILS = {
    "machine_normal_warm": "Sturdy, light-coloured cotton and linen tolerate a normal warm wash.",
    "machine_gentle_cold": "A cold, gentle cycle limits colour loss, dye transfer, heat damage and friction.",
    "hand_wash_cold": "Cold water and gentle handling protect delicate protein fibres.",
    "professional_clean": "A professional cleaner can treat the mark without risking water spots on silk.",
    "repair_first": "Mend the opening first; washing and pulling can enlarge it.",
}
CLASSES = tuple(LAUNDRY_MODES)


def _detections_by_type(detections):
    grouped = {kind: [] for kind in DAMAGE_TYPES}
    for detection in detections or []:
        kind = detection.get("type")
        if kind in grouped and finding_counts(kind, detection.get("confidence"), detection.get("confirmed") is True):
            area = detection.get("area_ratio")
            grouped[kind].append(float(area) if isinstance(area, (int, float)) else 0.0)
    return grouped


def damage_features(detections):
    """Count and largest box share of each damage type that passes DAMAGE_MIN_CONFIDENCE."""
    grouped = _detections_by_type(detections)
    features = {}
    for kind in DAMAGE_TYPES:
        areas = grouped[kind]
        features[f"{kind}_count"] = float(len(areas))
        features[f"{kind}_max_area"] = float(max(areas)) if areas else 0.0
    return features


def color_features(image_source):
    """Luminance and saturation of the central 80% of an EXIF-corrected photo."""
    with Image.open(image_source) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    rgb.thumbnail((256, 256), Image.Resampling.BILINEAR)
    width, height = rgb.size
    left, top = round(width * 0.1), round(height * 0.1)
    crop = rgb.crop((left, top, max(left + 1, width - left), max(top + 1, height - top)))
    pixels = np.asarray(crop, dtype=np.float32) / 255.0
    luminance = (0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]) * 255
    high, low = pixels.max(axis=2), pixels.min(axis=2)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)
    return {
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "saturation_mean": float(saturation.mean()),
        "dark_fraction": float((luminance < DARK_PIXEL_LUMINANCE).mean()),
    }


def glcm_summary(glcm72):
    """Mean of each GLCM property over the scanner's 3 distances x 4 angles."""
    values = np.asarray(glcm72, dtype=np.float64).reshape(len(GLCM_PROPERTIES), 12)
    return {f"glcm_{prop}": float(values[i].mean()) for i, prop in enumerate(GLCM_PROPERTIES)}


def fabric_features(fabric_type):
    fabric = str(fabric_type).casefold()
    return {f"fabric_{name}": 1.0 if fabric == name else 0.0 for name in FABRICS}


def feature_row(fabric_type, damage, color, glcm):
    """Combine feature groups into one dictionary covering FEATURE_NAMES."""
    row = {**fabric_features(fabric_type), **damage, **color, **glcm}
    missing = [name for name in FEATURE_NAMES if name not in row]
    if missing:
        raise ValueError(f"Missing care features: {', '.join(missing)}")
    return row


def feature_vector(row):
    return np.asarray([[row[name] for name in FEATURE_NAMES]], dtype=np.float64)


def is_dark_or_strong(row):
    return (row["luminance_mean"] < DARK_LUMINANCE or row["dark_fraction"] > DARK_FRACTION
            or row["saturation_mean"] > STRONG_SATURATION)


def reference_laundry_mode(row):
    """Return (mode, reasons) from the referenced care rules.

    Each reason is (text, reference ids). Reference ids match assessment.REFERENCES.
    """
    fabric = next((name for name in FABRICS if row[f"fabric_{name}"] >= 0.5), None)
    has = {kind: row[f"{kind}_count"] > 0 for kind in DAMAGE_TYPES}
    if has["hole"]:
        return "repair_first", [("An opening was detected. Pulling and agitation can enlarge it.", ["I", "L"])]
    if fabric == "silk":
        if has["stain"]:
            return "professional_clean", [
                ("Silk with a possible stain: wet spot treatments can leave water marks on silk.", ["M", "R", "N"])]
        return "hand_wash_cold", [("Silk is delicate when wet and can water-mark.", ["M", "R"])]
    if fabric == "wool":
        return "hand_wash_cold", [("Wool can felt and shrink with heat and agitation.", ["H", "J"])]
    if fabric == "denim":
        return "machine_gentle_cold", [("Cold, inside-out washing limits fading and dye transfer from denim.", ["K", "P"])]
    if fabric == "polyester":
        return "machine_gentle_cold", [("Polyester is heat-sensitive; a cool wash protects it.", ["M"])]
    reasons = []
    if is_dark_or_strong(row):
        reasons.append(("Dark or strongly coloured fabric: cooler, shorter cycles reduce colour loss and dye transfer.",
                        ["Q", "G"]))
    if has["pilling"]:
        reasons.append(("Pilling was detected: a gentle cycle reduces friction.", ["B", "J"]))
    if reasons:
        return "machine_gentle_cold", reasons
    if fabric in {"cotton", "linen"}:
        return "machine_normal_warm", [(f"Light-coloured {fabric} tolerates a normal warm wash.", ["M", "G"])]
    return "hand_wash_cold", [("The fabric is unidentified, so the gentlest method is suggested.", ["E", "G"])]


def care_alerts(row):
    """Automated care-risk alerts from the scan, each with a level and references."""
    fabric = next((name for name in FABRICS if row[f"fabric_{name}"] >= 0.5), None)
    alerts = []

    def alert(level, title, action, refs):
        alerts.append({"level": level, "title": title, "action": action, "reference_ids": refs})

    if row["hole_count"] > 0:
        alert("high", "Possible opening", "Mend the opening before washing or wearing; pulling can enlarge it.", ["I", "L"])
    if row["stain_count"] > 0:
        alert("medium", "Possible stain", "Identify the stain and test any treatment on a hidden area first. Avoid hot water until then.",
              ["N", "G"])
    if row["pilling_count"] > 0:
        alert("medium", "Pilling", "Wash inside out on a gentle cycle and reduce rubbing on the area.", ["B", "J"])
    if row["needlemark_count"] > 0:
        alert("low", "Needle-like mark", "Check the area for loose threads before cleaning.", ["A", "L"])
    if is_dark_or_strong(row):
        alert("medium", "Colour may bleed", "Wash dark or bright items separately in cool water.", ["Q", "G"])
    if fabric == "wool":
        alert("medium", "Felting risk", "Avoid hot water, wringing and tumble drying; dry flat.", ["H"])
    if fabric == "silk":
        alert("medium", "Water-mark risk", "Avoid wringing and wet spot treatment; dry away from direct sun.", ["M", "R"])
    if fabric == "polyester":
        alert("low", "Heat-sensitive", "Avoid high heat when drying or ironing.", ["M"])
    if fabric == "denim":
        alert("low", "Dye transfer", "Keep dark denim away from pale items when washing.", ["K", "P"])
    return alerts


class CareRecommender:
    """A trained Decision Tree, Random Forest or SVM saved by the training notebook."""

    def __init__(self, path):
        import joblib

        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or not {"model", "feature_names", "classes", "name"}.issubset(bundle):
            raise RuntimeError("Care model bundle must contain model, feature_names, classes and name.")
        if list(bundle["feature_names"]) != FEATURE_NAMES:
            raise RuntimeError("Care model features do not match care_model.FEATURE_NAMES.")
        if [str(c) for c in bundle["classes"]] != list(CLASSES):
            raise RuntimeError("Care model classes do not match care_model.CLASSES.")
        model = bundle["model"]
        if list(getattr(model, "classes_", [])) != list(range(len(CLASSES))):
            raise RuntimeError("Care model must predict class indexes in care_model.CLASSES order.")
        self.model = model
        self.name = str(bundle["name"])
        self.metrics = bundle.get("metrics", {})

    def predict(self, row):
        features = feature_vector(row)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            if hasattr(self.model, "predict_proba"):
                probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=np.float64)
            else:
                probabilities = np.eye(len(CLASSES))[int(self.model.predict(features)[0])]
        index = int(probabilities.argmax())
        return {
            "laundry_mode": CLASSES[index],
            "confidence": round(float(probabilities[index]) * 100, 2),
            "probabilities": {mode: round(float(p), 4) for mode, p in zip(CLASSES, probabilities)},
        }
