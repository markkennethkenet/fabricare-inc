"""Load the FabriCare fabric classifiers."""
import json
import math
import os
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.feature import graycomatrix, graycoprops
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DECISION_TREE_FEATURES = [
    "contrast_mean", "contrast_std",
    "dissimilarity_mean", "dissimilarity_std",
    "homogeneity_mean", "homogeneity_std",
    "energy_mean", "energy_std",
    "correlation_mean", "correlation_std",
    "ASM_mean", "ASM_std",
]
DECISION_TREE_LABELS = ["cotton", "denim", "polyester", "silk", "wool"]
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def extract_decision_tree_features(image_source):
    """Match the supplied Colab pipeline and return features in artifact order."""
    try:
        source = Image.open(image_source)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Cannot read image. Upload a valid JPG or PNG photo.") from error

    with source:
        if source.format not in {"JPEG", "PNG"}:
            raise ValueError("Only JPG and PNG images are supported.")
        try:
            gray = ImageOps.grayscale(source).resize((128, 128))
            quantized = np.asarray(gray, dtype=np.uint8) // 8
        except (OSError, ValueError) as error:
            raise ValueError("Cannot read image. Upload a valid JPG or PNG photo.") from error

    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    values = {}
    for prop in GLCM_PROPERTIES:
        result = graycoprops(glcm, prop)
        values[f"{prop}_mean"] = float(result.mean())
        values[f"{prop}_std"] = float(result.std())

    return np.asarray([values[name] for name in DECISION_TREE_FEATURES], dtype=np.float64).reshape(1, 12)


class FabricDecisionTree:
    """The trusted five-class 12-feature model supplied for FabriCare."""

    name = "GLCM (12 features) + Decision Tree"
    labels = DECISION_TREE_LABELS
    validation_accuracy = 47.53

    def __init__(self, model_path):
        artifact = load_fabric_model(Path(model_path))
        if not isinstance(artifact, dict) or not {"model", "features"}.issubset(artifact):
            raise RuntimeError("Decision Tree artifact must contain model and features metadata.")
        if artifact["features"] != DECISION_TREE_FEATURES:
            raise RuntimeError("Decision Tree feature metadata does not match the supported 12-feature pipeline.")

        self.model = artifact["model"]
        if getattr(self.model, "n_features_in_", None) != 12:
            raise RuntimeError("Decision Tree must accept exactly 12 GLCM features.")
        model_features = getattr(self.model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != DECISION_TREE_FEATURES:
            raise RuntimeError("Decision Tree feature ordering does not match its metadata.")
        classes = [str(label) for label in getattr(self.model, "classes_", [])]
        if len(classes) != 5 or set(classes) != set(DECISION_TREE_LABELS):
            raise RuntimeError("Decision Tree classes must be Cotton, Denim, Polyester, Silk, and Wool.")

    def classify(self, image_path):
        features = extract_decision_tree_features(image_path)
        # The artifact records feature names; the vector is built in that verified
        # stored order, so only sklearn's array-vs-DataFrame notice is suppressed.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names",
                category=UserWarning,
            )
            prediction = str(self.model.predict(features)[0])
        if prediction not in self.labels:
            raise RuntimeError("Decision Tree returned an unsupported fabric class.")
        if hasattr(self.model, "predict_proba"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names",
                    category=UserWarning,
                )
                probabilities = self.model.predict_proba(features)[0]
            scores = {str(label): float(score)
                      for label, score in zip(self.model.classes_, probabilities)}
        else:
            scores = {prediction: 1.0}
        return {"fabric_type": prediction,
                "confidence": scores[prediction] * 100,
                "probabilities": scores,
                "score_kind": "uncalibrated_decision_tree_vote"}


class FabricMobileNet:
    name = "MobileNetV2 (six fabrics)"

    def __init__(self, directory):
        import tensorflow as tf

        self.labels = json.loads((directory / "fabric_labels.json").read_text(encoding="utf-8"))
        if (not isinstance(self.labels, list) or len(self.labels) != 6
                or any(not isinstance(label, str) or not label for label in self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise RuntimeError("Invalid MobileNetV2 fabric labels.")
        self.model = tf.keras.models.load_model(directory / "fabric_mobilenetv2.keras", compile=False)
        if (tuple(self.model.input_shape[1:]) != (160, 160, 3)
                or tuple(self.model.output_shape[1:]) != (len(self.labels),)):
            raise RuntimeError("MobileNetV2 model dimensions do not match the export.")
        self.resize = tf.image.resize

    def classify(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot read image.")
        # Match training: RGB, antialiased bilinear resize, and 0–255 input.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = self.resize(rgb, (160, 160), antialias=True)[None]
        probabilities = np.asarray(self.model(batch, training=False), dtype=np.float32)[0]
        return fabric_prediction(self.labels, probabilities, self.name)


SIX_FABRIC_LABELS = ("cotton", "denim", "linen", "polyester", "silk", "wool")
MOBILENET_V2_CLASSES = set(SIX_FABRIC_LABELS)


def prepare_fabric_rgb(image_source):
    """Prepare the shared 224 x 224 RGB image with centered gray padding."""
    try:
        with Image.open(image_source) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPG and PNG images are supported.")
            if source.width * source.height > 24_000_000:
                raise ValueError("Image dimensions are too large to analyze.")
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            return ImageOps.pad(rgb, (224, 224), method=Image.Resampling.BILINEAR,
                                color=(127, 127, 127))
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Cannot read image. Upload a valid JPG or PNG photo.") from error


def prepare_mobilenet_v2_image(image_source):
    """Return a batch of raw RGB pixels in the saved CNN's input format."""
    return np.asarray(prepare_fabric_rgb(image_source), dtype=np.float32)[None]


def fabric_tta_enabled():
    """Mirror averaging is on unless FABRIC_TTA is set to off, false, 0, or none."""
    return os.environ.get("FABRIC_TTA", "flip").strip().lower() not in {"0", "off", "false", "none"}


def prepare_mobilenet_v2_tta_batch(image_source):
    """Return the photo and its horizontal mirror as one batch.

    The V3 CNN trained with RandomFlip("horizontal"), so the mirror is an
    in-distribution view. Averaging both reduces score swings from small
    framing changes without adding crops or zooms the model never saw.
    """
    image = prepare_mobilenet_v2_image(image_source)
    return np.concatenate([image, image[:, :, ::-1, :]])


class FabricMobileNetV2:
    name = "MobileNetV2 (V3 refit)"
    filename = "mobilenetv2_v3_refit.keras"
    validation_accuracy = 72.85
    test_accuracy = 81.87
    test_macro_f1 = 0.8221

    def __init__(self, directory):
        labels_path = directory / "class_names.json"
        for path in (labels_path, *self.model_files(directory)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing MobileNetV2 file: {path.name}.")
        try:
            self.labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot read MobileNetV2 labels from {labels_path}.") from error
        if (not isinstance(self.labels, list) or len(self.labels) != 6
                or any(not isinstance(label, str) for label in self.labels)
                or set(self.labels) != MOBILENET_V2_CLASSES):
            raise RuntimeError("class_names.json must list each of the six supported fabrics once, in model output order.")
        try:
            self.model = self.load_network(directory)
        except Exception as error:
            raise RuntimeError("Cannot load the CNN export. Check the model files and install the "
                               "pinned API runtime with scripts/setup_api.ps1.") from error
        if (tuple(self.model.input_shape[1:]) != (224, 224, 3)
                or tuple(self.model.output_shape[1:]) != (len(self.labels),)):
            raise RuntimeError("MobileNetV2 must accept 224 x 224 RGB and return six scores.")
        self.tta = fabric_tta_enabled()

    def model_files(self, directory):
        return [directory / self.filename]

    def load_network(self, directory):
        import tensorflow as tf

        return tf.keras.models.load_model(directory / self.filename, compile=False)

    def classify(self, image_path):
        batch = (prepare_mobilenet_v2_tta_batch(image_path) if self.tta
                 else prepare_mobilenet_v2_image(image_path))
        try:
            scores = np.asarray(self.model(batch, training=False), dtype=np.float32)
        except Exception as error:
            raise RuntimeError("MobileNetV2 inference failed.") from error
        if scores.shape != (len(batch), len(self.labels)):
            raise RuntimeError("MobileNetV2 returned an unexpected output shape.")
        probabilities = scores.mean(axis=0)
        total = float(probabilities.sum())
        if total > 0:
            probabilities = probabilities / total
        return {
            **fabric_prediction(self.labels, probabilities, self.name),
            "score_kind": "uncalibrated_softmax",
            "inference_views": len(batch),
        }


class FabricCNN(FabricMobileNetV2):
    """A retrained CNN export described by its own model_info.json (V4 training notebook).

    The export is either a full Keras file, or an architecture file with 16-bit weights. The 16-bit form
    keeps large backbones under GitHub's 100 MB file limit; weights are cast back to 32-bit when loaded.
    """

    filename = "fabric_cnn.keras"
    architecture = "fabric_cnn_architecture.json"
    half_weights = "fabric_cnn_weights_fp16.npz"
    # The V4 notebook trains with label smoothing 0.1, which caps the top score of a six-class
    # prediction near 92% (0.9 + 0.1 / 6). The API's general 85% review threshold would flag
    # almost every confident scan, so V4 exports use their own threshold (model_info.json).
    default_review_threshold = 60.0

    def __init__(self, directory):
        info = read_model_info(directory)
        self.name = info["name"]
        self.validation_accuracy = info.get("validation_accuracy")
        self.test_accuracy = info.get("test_accuracy")
        self.test_macro_f1 = info.get("test_macro_f1")
        threshold = info.get("review_threshold_percent", self.default_review_threshold)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
            raise RuntimeError("model_info.json review_threshold_percent must be between 0 and 100.")
        self.review_threshold = float(threshold)
        super().__init__(directory)

    def model_files(self, directory):
        if (directory / self.filename).is_file():
            return [directory / self.filename]
        return [directory / self.architecture, directory / self.half_weights]

    def load_network(self, directory):
        import tensorflow as tf

        if (directory / self.filename).is_file():
            return tf.keras.models.load_model(directory / self.filename, compile=False)
        model = tf.keras.models.model_from_json((directory / self.architecture).read_text(encoding="utf-8"))
        with np.load(directory / self.half_weights) as stored:
            weights = [stored[f"arr_{i}"].astype(np.float32) for i in range(len(stored.files))]
        model.set_weights(weights)
        return model


def read_model_info(directory):
    """Name and test metrics of a CNN export, readable without loading TensorFlow."""
    try:
        info = json.loads((Path(directory) / "model_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read model_info.json in {directory}.") from error
    if not isinstance(info, dict) or not isinstance(info.get("name"), str) or not info["name"]:
        raise RuntimeError("model_info.json must name the CNN.")
    return info


def extract_glcm72_features(image_source):
    rgb = prepare_fabric_rgb(image_source)
    gray = ImageOps.grayscale(rgb).resize((128, 128), Image.Resampling.BICUBIC)
    glcm = graycomatrix(
        np.asarray(gray, dtype=np.uint8) // 8,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True,
    )
    features = np.concatenate([graycoprops(glcm, prop).ravel() for prop in GLCM_PROPERTIES])
    return np.nan_to_num(features, nan=0, posinf=0, neginf=0).reshape(1, 72)


class FabricGLCM72:
    # The supplied six-class exports use integer IDs in alphabetical fabric order.
    labels = SIX_FABRIC_LABELS
    specifications = {
        "svm": ("GLCM + SVM", "glcm_svm.joblib", 58.92, 0.5781),
        "glcm_dt": ("GLCM + Decision Tree", "glcm_decision_tree.joblib", 50.42, 0.5109),
    }

    def __init__(self, directory, model_type):
        self.name, filename, self.test_accuracy, self.test_macro_f1 = self.specifications[model_type]
        self.model_type = model_type
        # A retrained export ships metrics.json with its own held-out results.
        metrics_path = directory / "metrics.json"
        if metrics_path.is_file():
            try:
                measured = json.loads(metrics_path.read_text(encoding="utf-8")).get(model_type, {})
            except (OSError, ValueError, AttributeError):
                measured = {}
            self.test_accuracy = measured.get("test_accuracy", self.test_accuracy)
            self.test_macro_f1 = measured.get("test_macro_f1", self.test_macro_f1)
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {self.name} model file: {filename}.")
        self.model = load_fabric_model(path)
        if model_type == "svm":
            valid = (isinstance(self.model, Pipeline)
                     and [type(step) for _, step in self.model.steps] == [StandardScaler, SVC])
        else:
            valid = isinstance(self.model, DecisionTreeClassifier)
        if (not valid or getattr(self.model, "n_features_in_", None) != 72
                or list(getattr(self.model, "classes_", [])) != list(range(len(self.labels)))):
            raise RuntimeError(f"{self.name} does not match the expected 72 features and six class IDs.")

    def classify(self, image_path):
        features = extract_glcm72_features(image_path)
        index = self.model.predict(features)[0]
        if index not in range(len(self.labels)):
            raise RuntimeError(f"{self.name} returned an unsupported fabric class.")
        label = self.labels[index]
        if not hasattr(self.model, "predict_proba"):
            return {"fabric_type": label, "confidence": None,
                    "probabilities": {}, "score_kind": "unavailable"}
        probabilities = self.model.predict_proba(features)[0]
        result = fabric_prediction(self.labels, probabilities, self.name)
        return {**result, "fabric_type": label,
                "confidence": result["probabilities"][label] * 100,
                "score_kind": "svm_platt_probability" if self.model_type == "svm"
                else "uncalibrated_decision_tree_vote"}


def fabric_prediction(labels, probabilities, model_name):
    if (probabilities.shape != (len(labels),)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1, atol=1e-4)):
        raise RuntimeError(f"{model_name} returned invalid probabilities.")
    scores = dict(zip(labels, map(float, probabilities)))
    label = max(scores, key=scores.get)
    return {"fabric_type": label, "confidence": scores[label] * 100,
            "probabilities": scores}


class FabricEnsemble:
    name = "EfficientNetB0 + MobileNetV2"

    def __init__(self, directory):
        config = json.loads((directory / "inference_config.json").read_text(encoding="utf-8"))
        self.labels = config["classes"]
        if (not isinstance(self.labels, list) or not self.labels
                or any(not isinstance(label, str) or not label for label in self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise RuntimeError("Invalid fabric ensemble labels.")
        members = [(config[f"{key}_weight"], config[f"{key}_model"])
                   for key in ("efficientnet", "mobilenet")]
        if (any(not isinstance(weight, (int, float)) or not 0 <= weight <= 1
                for weight, _ in members)
                or not math.isclose(sum(weight for weight, _ in members), 1)):
            raise RuntimeError("Invalid fabric ensemble weights.")

        try:
            from tensorflow.keras.models import load_model
        except ModuleNotFoundError as error:
            raise RuntimeError("Install requirements-ensemble.txt to run the fabric ensemble.") from error

        self.members = [(weight, load_model(directory / filename, compile=False))
                        for weight, filename in members if weight > 0]
        for _, model in self.members:
            if (tuple(model.input_shape[1:]) != (224, 224, 3)
                    or tuple(model.output_shape[1:]) != (len(self.labels),)):
                raise RuntimeError("Fabric ensemble model dimensions do not match the export.")

    def classify(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot read image.")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        batch = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None].astype(np.float32)
        # Both models expect RGB in [0, 255] and embed their own preprocessing.
        probabilities = sum(
            weight * np.asarray(model(batch, training=False), dtype=np.float32)[0]
            for weight, model in self.members
        )
        return fabric_prediction(self.labels, probabilities, self.name)


def load_fabric_model(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            return joblib.load(path)
    except InconsistentVersionWarning as error:
        raise RuntimeError(
            f"Fabric model needs scikit-learn {error.original_sklearn_version}; "
            f"this interpreter has {error.current_sklearn_version}. "
            "Run scripts/setup_api.ps1 and start with .venv-api/Scripts/python.exe. "
            "Do not re-save or retrain the model to fix the environment."
        ) from error
