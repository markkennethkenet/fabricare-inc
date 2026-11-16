import unittest
import warnings
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning
from model_runtime import (
    DECISION_TREE_FEATURES,
    DECISION_TREE_LABELS,
    SIX_FABRIC_LABELS,
    FabricDecisionTree,
    FabricMobileNetV2,
    extract_decision_tree_features,
    fabric_tta_enabled,
    load_fabric_model,
    prepare_mobilenet_v2_image,
    prepare_mobilenet_v2_tta_batch,
)


class ModelRuntimeTests(unittest.TestCase):
    def test_mobilenet_v2_corrects_orientation_and_pads_raw_rgb_pixels(self):
        source = Image.new("RGB", (80, 40), (250, 20, 30))
        exif = source.getexif()
        exif[274] = 6
        encoded = BytesIO()
        source.save(encoded, format="PNG", exif=exif)
        encoded.seek(0)

        batch = prepare_mobilenet_v2_image(encoded)
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_array_equal(batch[0, 0, 0], [127, 127, 127])
        np.testing.assert_array_equal(batch[0, 0, 112], [250, 20, 30])
        np.testing.assert_array_equal(batch[0, 223, 223], [127, 127, 127])

    def test_matching_runtime_loads_original_file(self):
        model = object()
        with patch("model_runtime.joblib.load", return_value=model) as load:
            self.assertIs(load_fabric_model("unchanged.pkl"), model)
        load.assert_called_once_with("unchanged.pkl")

    def test_mismatched_runtime_fails_instead_of_hiding_warning(self):
        def incompatible(_path):
            warnings.warn(InconsistentVersionWarning(
                estimator_name="DecisionTreeClassifier",
                current_sklearn_version="1.9.0",
                original_sklearn_version="1.6.1",
            ))
        with patch("model_runtime.joblib.load", side_effect=incompatible):
            with self.assertRaisesRegex(RuntimeError, r"needs scikit-learn 1\.6\.1"):
                load_fabric_model("unchanged.pkl")

    def test_supplied_decision_tree_loads_and_uses_twelve_ordered_features(self):
        model_path = __import__("pathlib").Path(__file__).parent / "model" / "best_decision_tree.joblib"
        classifier = FabricDecisionTree(model_path)
        self.assertEqual(classifier.labels, DECISION_TREE_LABELS)

        pixels = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
        image = Image.fromarray(pixels, mode="L")
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        encoded.seek(0)

        features = extract_decision_tree_features(encoded)
        self.assertEqual(features.shape, (1, 12))
        self.assertEqual(len(DECISION_TREE_FEATURES), 12)
        result = classifier.classify(BytesIO(encoded.getvalue()))
        self.assertIn(result["fabric_type"], DECISION_TREE_LABELS)
        self.assertEqual(result["score_kind"], "uncalibrated_decision_tree_vote")

    def test_decision_tree_rejects_invalid_or_unsupported_image(self):
        model_path = __import__("pathlib").Path(__file__).parent / "model" / "best_decision_tree.joblib"
        classifier = FabricDecisionTree(model_path)
        with self.assertRaisesRegex(ValueError, "valid JPG or PNG"):
            classifier.classify(BytesIO(b"not an image"))

        image = Image.new("RGB", (32, 32), "white")
        encoded = BytesIO()
        image.save(encoded, format="BMP")
        encoded.seek(0)
        with self.assertRaisesRegex(ValueError, "Only JPG and PNG"):
            classifier.classify(encoded)


def _png(image):
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    encoded.seek(0)
    return encoded


def _cnn(scores_by_view, tta):
    """A FabricMobileNetV2 with a stub network, skipping the TensorFlow load."""
    classifier = FabricMobileNetV2.__new__(FabricMobileNetV2)
    classifier.labels = list(SIX_FABRIC_LABELS)
    classifier.tta = tta
    classifier.seen = []

    def network(batch, training):
        classifier.seen.append(np.asarray(batch))
        return np.asarray(scores_by_view[:len(batch)], dtype=np.float32)

    classifier.model = network
    return classifier


class HalfSizeCnnTests(unittest.TestCase):
    def test_sixteen_bit_export_loads_and_matches_the_full_model(self):
        import json
        import tempfile
        from pathlib import Path

        import tensorflow as tf
        from model_runtime import FabricCNN

        tf.keras.utils.set_random_seed(3)
        inputs = tf.keras.Input((224, 224, 3))
        x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
        x = tf.keras.layers.Conv2D(4, 3, strides=4, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(6, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        directory = Path(tempfile.mkdtemp())
        (directory / FabricCNN.architecture).write_text(model.to_json(), encoding="utf-8")
        np.savez_compressed(directory / FabricCNN.half_weights, *[w.astype(np.float16) for w in model.get_weights()])
        (directory / "class_names.json").write_text(json.dumps(list(SIX_FABRIC_LABELS)))
        (directory / "model_info.json").write_text(json.dumps({"name": "Test CNN (V4)", "test_accuracy": 86.4}))

        loaded = FabricCNN(directory)
        self.assertEqual(loaded.name, "Test CNN (V4)")
        self.assertEqual(loaded.test_accuracy, 86.4)
        photo = _png(Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 90, 3), dtype=np.uint8)))
        batch = prepare_mobilenet_v2_image(photo)
        np.testing.assert_allclose(np.asarray(loaded.model(batch)), np.asarray(model(batch)), atol=2e-3)
        photo.seek(0)
        self.assertIn(loaded.classify(photo)["fabric_type"], SIX_FABRIC_LABELS)


class MirrorAveragingTests(unittest.TestCase):
    def test_tta_batch_adds_the_horizontal_mirror(self):
        pixels = np.zeros((224, 224, 3), dtype=np.uint8)
        pixels[:, :112] = (200, 10, 10)
        batch = prepare_mobilenet_v2_tta_batch(_png(Image.fromarray(pixels)))
        self.assertEqual(batch.shape, (2, 224, 224, 3))
        np.testing.assert_array_equal(batch[0, 50, 10], [200, 10, 10])
        np.testing.assert_array_equal(batch[1, 50, 10], [0, 0, 0])
        np.testing.assert_array_equal(batch[1], batch[0, :, ::-1, :])

    def test_cnn_averages_photo_and_mirror_scores(self):
        original = [0.50, 0.40, 0.02, 0.04, 0.02, 0.02]
        mirrored = [0.30, 0.60, 0.02, 0.04, 0.02, 0.02]
        classifier = _cnn([original, mirrored], tta=True)
        result = classifier.classify(_png(Image.new("RGB", (64, 48), "navy")))
        self.assertEqual(classifier.seen[0].shape[0], 2)
        self.assertEqual(result["inference_views"], 2)
        self.assertEqual(result["fabric_type"], "denim")
        self.assertAlmostEqual(result["probabilities"]["cotton"], 0.40, places=5)
        self.assertAlmostEqual(result["probabilities"]["denim"], 0.50, places=5)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=5)
        self.assertEqual(result["score_kind"], "uncalibrated_softmax")

    def test_cnn_uses_one_view_when_mirroring_is_off(self):
        classifier = _cnn([[0.50, 0.40, 0.02, 0.04, 0.02, 0.02]], tta=False)
        result = classifier.classify(_png(Image.new("RGB", (64, 48), "navy")))
        self.assertEqual(classifier.seen[0].shape[0], 1)
        self.assertEqual(result["inference_views"], 1)
        self.assertEqual(result["fabric_type"], "cotton")

    def test_environment_switch_turns_mirroring_off(self):
        for value, expected in (("", True), ("flip", True), ("off", False),
                                ("0", False), ("FALSE", False), ("none", False)):
            with patch.dict("os.environ", {"FABRIC_TTA": value}):
                self.assertIs(fabric_tta_enabled(), expected, value)
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(fabric_tta_enabled())


if __name__ == "__main__":
    unittest.main()
