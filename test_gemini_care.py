"""Gemini care wording: off without a key, never fails a scan, and keeps the referenced tips."""
import io
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

import gemini_care

ROOT = Path(__file__).parent
FACTS = {"fabric": "denim", "laundry_mode_from_care_model": "Machine wash · gentle cycle · cold"}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def gemini_answer(care, laundry):
    text = json.dumps({"care_tips": care, "laundry_tips": laundry})
    return FakeResponse(json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode())


class GeminiCareTests(unittest.TestCase):
    def setUp(self):
        self.photo = ROOT / "test_gemini_photo.png"
        Image.new("RGB", (1200, 900), (30, 40, 90)).save(self.photo)
        self.addCleanup(self.photo.unlink)

    @patch.dict("os.environ", {"GEMINI_API_KEY": ""})
    def test_off_without_a_key(self):
        with patch("gemini_care.urlopen") as call:
            self.assertFalse(gemini_care.enabled())
            self.assertIsNone(gemini_care.write_care_advice(FACTS, str(self.photo)))
        call.assert_not_called()

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": ""})
    def test_writes_up_to_three_sentences_from_results_and_photo(self):
        answer = gemini_answer(["Wash it inside out to keep the indigo", "b.", "c.", "a fourth is dropped."],
                               ["Machine wash cold on a gentle cycle."])
        with patch("gemini_care.urlopen", return_value=answer) as call:
            written = gemini_care.write_care_advice(FACTS, str(self.photo))
        self.assertEqual(written["care_tips"][0], "Wash it inside out to keep the indigo.")
        self.assertEqual(len(written["care_tips"]), 3)
        self.assertEqual(written["laundry_tips"], ["Machine wash cold on a gentle cycle."])
        request = call.call_args.args[0]
        self.assertIn("/models/gemini-3.5-flash-lite:generateContent", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        self.assertNotIn("test-key", request.full_url)  # the key never goes in the URL
        body = json.loads(request.data)
        parts = body["contents"][0]["parts"]
        self.assertIn('"fabric": "denim"', parts[0]["text"])
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/jpeg")
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertIn("never go beyond the care label limits", body["systemInstruction"]["parts"][0]["text"])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_SEND_PHOTO": "false"})
    def test_photo_can_be_left_out(self):
        with patch("gemini_care.urlopen", return_value=gemini_answer(["a."], ["b."])) as call:
            gemini_care.write_care_advice(FACTS, str(self.photo))
        parts = json.loads(call.call_args.args[0].data)["contents"][0]["parts"]
        self.assertEqual(len(parts), 1)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_quota_errors_and_bad_answers_keep_the_rule_tips(self):
        quota = HTTPError("https://example.test", 429, "Too Many Requests", {}, None)
        cases = [quota, TimeoutError(), gemini_answer([], ["b."]), gemini_answer(["x" * 300], ["b."]),
                 FakeResponse(b'{"candidates": []}')]
        for case in cases:
            with self.subTest(case=type(case).__name__), patch(
                    "gemini_care.urlopen", **({"side_effect": case} if isinstance(case, Exception)
                                              else {"return_value": case})):
                self.assertIsNone(gemini_care.write_care_advice(FACTS))


class ApiIntegrationTests(unittest.TestCase):
    def test_api_uses_gemini_wording_and_keeps_the_sources(self):
        import care_model
        import fabricare_api as api
        from assessment import normalize_user_inputs

        row = care_model.feature_row("denim", care_model.damage_features([]),
                                     {"luminance_mean": 40.0, "luminance_std": 10.0, "saturation_mean": 0.3,
                                      "dark_fraction": 0.7}, care_model.glcm_summary(list(range(72))))
        inputs = normalize_user_inputs({})
        written = {"care_tips": ["Wash your dark jeans inside out."], "laundry_tips": ["Cold, gentle cycle."]}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
                patch("gemini_care.write_care_advice", return_value=written) as write:
            care = api.ai_care_from_row(row, "denim", inputs, fabric_model="ConvNeXt-Tiny (V4)")
        facts = write.call_args.args[0]
        self.assertEqual(facts["colour"], "dark or strongly coloured")
        self.assertTrue(facts["verified_guidance"])
        self.assertEqual(care["care_tips"][0]["text"], "Wash your dark jeans inside out.")
        self.assertEqual(care["writer"], "gemini-3.5-flash-lite")
        self.assertTrue(care["provided_by"].endswith("· worded by Gemini"))
        self.assertTrue(care["reference_tips"]["care"])  # the referenced tips stay with the scan
        cited = {ref["id"] for ref in api.with_care_references([], care)}
        self.assertTrue({"K", "P"} <= cited)
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            plain = api.ai_care_from_row(row, "denim", inputs)
        self.assertIsNone(plain["writer"])
        self.assertEqual(plain["care_tips"], plain["reference_tips"]["care"])


class DeploymentFilesTests(unittest.TestCase):
    def test_every_local_module_the_api_imports_is_shipped(self):
        modules = {path.stem for path in ROOT.glob("*.py") if not path.stem.startswith("test_")}
        imported = set()
        for name in ("fabricare_api", "serve_api", "auth_routes", "assessment", "care_model", "gemini_care",
                     "model_runtime", "model_analysis", "image_processing", "database", "db_config"):
            source = (ROOT / f"{name}.py").read_text(encoding="utf-8")
            imported |= {m for m in re.findall(r"^\s*(?:from|import)\s+(\w+)", source, re.M) if m in modules}
        for dockerfile in ("Dockerfile", "Dockerfile.vercel"):
            copied = set(re.findall(r"(\w+)\.py", (ROOT / dockerfile).read_text(encoding="utf-8")))
            self.assertEqual(imported - copied, set(), dockerfile)
        for ignore in (".dockerignore", ".vercelignore"):
            allowed = set(re.findall(r"^!(\w+)\.py", (ROOT / ignore).read_text(encoding="utf-8"), re.M))
            if allowed:
                self.assertEqual(imported - allowed, set(), ignore)


if __name__ == "__main__":
    unittest.main()
