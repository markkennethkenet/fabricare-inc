import os
import unittest
import importlib
from unittest.mock import patch

import care_model
import condition_runtime

COLOR = {"luminance_mean": 180.0, "luminance_std": 20.0, "saturation_mean": 0.1, "dark_fraction": 0.05}
GLCM = {f"glcm_{p}": 0.5 for p in care_model.GLCM_PROPERTIES}
ROW = care_model.feature_row("cotton", care_model.damage_features([]), COLOR, GLCM)
INPUTS = {"care_label": {"wash": "unknown"}}


class ConditionRowsTest(unittest.TestCase):
    def test_condition_becomes_one_finding_for_the_rules(self):
        self.assertEqual(condition_runtime.with_condition(ROW, "hole")["hole_count"], 1.0)
        self.assertEqual(care_model.reference_laundry_mode(condition_runtime.with_condition(ROW, "hole"))[0],
                         "repair_first")
        unchanged = condition_runtime.with_condition(ROW, "wrinkle")
        self.assertEqual({k: unchanged[k] for k in ROW}, ROW)

    def test_condition_row_matches_the_v6_features(self):
        row = condition_runtime.condition_row(ROW, "cotton", "stain")
        self.assertEqual(row["condition_stain"], 1.0)
        self.assertEqual(row["glcm_contrast"], 0.5)


class FakeV6:
    name = "Decision Tree"

    def predict(self, row):
        assert row["condition_pilling"] == 1.0
        return {"laundry_mode": "machine_gentle_cold", "confidence": 100.0, "probabilities": {}}


class ApiConditionCareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, FABRIC_CLASSIFIER="legacy"), patch("joblib.load"), patch("ultralytics.YOLO"):
            cls.api = importlib.import_module("fabricare_api")

    def setUp(self):
        gemini_off = patch.dict(os.environ, {"GEMINI_API_KEY": ""})
        gemini_off.start()
        self.addCleanup(gemini_off.stop)

    def test_v6_model_is_used_when_a_condition_is_detected(self):
        with patch.object(self.api, "load_condition_care", return_value=FakeV6()):
            care = self.api.ai_care_from_row(ROW, "cotton", INPUTS, condition={"condition": "pilling"})
        self.assertEqual((care["model"], care["laundry_mode"], care["condition"]),
                         ("Decision Tree (V6)", "machine_gentle_cold", "pilling"))
        self.assertTrue(any(alert["title"] == "Pilling" for alert in care["alerts"]))

    def test_without_a_condition_the_v5_path_is_unchanged(self):
        with patch.object(self.api, "load_condition_care") as v6, \
                patch.object(self.api, "load_care_recommender", return_value=None):
            care = self.api.ai_care_from_row(ROW, "cotton", INPUTS)
        v6.assert_not_called()
        self.assertEqual(care["status"], "rule_based_guidance")
        self.assertIsNone(care["condition"])

    def test_missing_classifier_returns_no_condition(self):
        with patch.object(self.api, "load_condition_classifier", return_value=None):
            self.assertIsNone(self.api.detect_condition([{"view": "front", "path": "x.jpg"}]))


if __name__ == "__main__":
    unittest.main()
