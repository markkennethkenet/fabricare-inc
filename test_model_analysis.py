import unittest

from model_analysis import model_result


class ModelAnalysisTests(unittest.TestCase):
    def test_high_model_score_can_return_result(self):
        result = model_result(
            {"fabric_type": "cotton", "confidence": 91}, [], ["cotton"], True, ["hole"]
        )
        self.assertEqual(result["fabric"]["verification_status"], "model_result")
        self.assertEqual(result["fabric"]["final_prediction"], "cotton")
        self.assertFalse(result["fabric"]["review_required"])
        self.assertEqual(result["damage"]["verification_status"], "no_visible_damage")

    def test_low_model_score_requires_review(self):
        result = model_result(
            {"fabric_type": "cotton", "confidence": 58}, [], ["cotton"], True, ["hole"]
        )
        self.assertEqual(result["fabric"]["verification_status"], "low_confidence_review")
        self.assertTrue(result["fabric"]["review_required"])

    def test_unsupported_model_class_requires_review(self):
        result = model_result(
            {"fabric_type": "linen", "confidence": 94}, [], ["cotton", "denim"], True, ["hole"]
        )
        self.assertEqual(result["fabric"]["verification_status"], "unsupported_model_class")
        self.assertEqual(result["fabric"]["final_prediction"], "uncertain")

    def test_damage_findings_are_presented_for_manual_review(self):
        result = model_result(
            {"fabric_type": "cotton", "confidence": 91}, [{"type": "hole"}], ["cotton"], True, ["hole"]
        )
        self.assertEqual(result["damage"]["final_predictions"], ["hole"])
        self.assertTrue(result["damage"]["review_required"])


if __name__ == "__main__":
    unittest.main()
