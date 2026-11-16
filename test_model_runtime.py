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
    FabricDecisionTree,
    extract_decision_tree_features,
    load_fabric_model,
    prepare_mobilenet_v2_image,
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


if __name__ == "__main__":
    unittest.main()
