"""V6 fabric-condition classifier and the condition-aware care model.

The ConvNeXt-Tiny condition classifier (model/damage_classifier) labels a whole photo as hole, needle mark,
normal, pilling, stain or wrinkle. The V6 care model (model/care_v6) recommends a laundry mode from the
fabric, that condition, and the colour and GLCM features (damage_condition.FEATURE_NAMES). Both are optional:
when their files are missing the API keeps the YOLOv8 detector and the V5 care model.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import care_model
import damage_condition
from model_runtime import prepare_fabric_rgb


class ConditionClassifier:
    architecture = "damage_cnn_architecture.json"
    half_weights = "damage_cnn_weights_fp16.npz"

    def __init__(self, directory):
        directory = Path(directory)
        info = json.loads((directory / "model_info.json").read_text(encoding="utf-8"))
        labels = json.loads((directory / "class_names.json").read_text(encoding="utf-8"))
        if list(labels) != list(damage_condition.DAMAGE_CLASSES):
            raise RuntimeError("Condition classifier classes do not match damage_condition.DAMAGE_CLASSES.")
        import tensorflow as tf

        model = tf.keras.models.model_from_json((directory / self.architecture).read_text(encoding="utf-8"))
        with np.load(directory / self.half_weights) as stored:
            model.set_weights([stored[f"arr_{i}"].astype(np.float32) for i in range(len(stored.files))])
        if tuple(model.output_shape[1:]) != (len(labels),):
            raise RuntimeError("Condition classifier output does not match its class list.")
        self.model, self.labels = model, list(labels)
        self.name = str(info.get("name", "ConvNeXt-Tiny damage condition"))
        self.test_accuracy = info.get("test_accuracy")

    def classify(self, image_path):
        pixels = np.asarray(prepare_fabric_rgb(image_path), dtype=np.float32)
        batch = np.stack([pixels, pixels[:, ::-1, :]])  # mirror-averaged, as in training and testing
        output = np.asarray(self.model(batch, training=False), dtype=np.float64).mean(axis=0)
        output = output / output.sum()
        index = int(output.argmax())
        return {"condition": self.labels[index], "confidence": round(float(output[index]) * 100, 2),
                "probabilities": {label: round(float(p), 4) for label, p in zip(self.labels, output)},
                "model": self.name}


class ConditionCareRecommender:
    """A V6 Decision Tree, Random Forest or SVM bundle saved by the V6 notebook."""

    def __init__(self, path):
        import joblib

        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or not {"model", "feature_names", "classes", "name"}.issubset(bundle):
            raise RuntimeError("V6 care bundle must contain model, feature_names, classes and name.")
        if list(bundle["feature_names"]) != list(damage_condition.FEATURE_NAMES):
            raise RuntimeError("V6 care model features do not match damage_condition.FEATURE_NAMES.")
        if [str(c) for c in bundle["classes"]] != list(care_model.CLASSES):
            raise RuntimeError("V6 care model classes do not match care_model.CLASSES.")
        if list(getattr(bundle["model"], "classes_", [])) != list(range(len(care_model.CLASSES))):
            raise RuntimeError("V6 care model must predict class indexes in care_model.CLASSES order.")
        self.model, self.name = bundle["model"], str(bundle["name"])

    def predict(self, row):
        features = damage_condition.feature_vector(row)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            if hasattr(self.model, "predict_proba"):
                probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=np.float64)
            else:
                probabilities = np.eye(len(care_model.CLASSES))[int(self.model.predict(features)[0])]
        index = int(probabilities.argmax())
        return {"laundry_mode": care_model.CLASSES[index],
                "confidence": round(float(probabilities[index]) * 100, 2),
                "probabilities": {mode: round(float(p), 4) for mode, p in zip(care_model.CLASSES, probabilities)}}


def with_condition(row, condition):
    """A care_model row whose damage counts also include the classifier's condition as one finding."""
    finding = damage_condition.CONDITION_FINDINGS.get(condition)
    if not finding:
        return dict(row)
    return {**row, f"{finding}_count": max(1.0, float(row.get(f"{finding}_count", 0.0)))}


def condition_row(row, fabric_type, condition):
    """The V6 care-model input built from a care_model row (its colour and GLCM values) and the condition."""
    color = {key: row[key] for key in ("luminance_mean", "luminance_std", "saturation_mean", "dark_fraction")}
    glcm = {key: row[key] for key in row if key.startswith("glcm_")}
    return damage_condition.feature_row(fabric_type, condition, color, glcm)
