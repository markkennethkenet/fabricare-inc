import unittest
from datetime import date

from assessment import LIFESPAN_MESSAGE, assess, normalize_user_inputs


class AssessmentTests(unittest.TestCase):
    def test_clear_photo_does_not_imply_health_or_service_life(self):
        result = assess("cotton", [], fabric_confidence=93)
        self.assertIn("No supported findings flagged", result["condition_rating"])
        self.assertIsNone(result["health_score"])
        self.assertIsNone(result["estimated_remaining_months"])
        self.assertEqual(result["estimated_lifespan"], LIFESPAN_MESSAGE)
        self.assertEqual(result["lifespan"]["status"], "insufficient_evidence")
        self.assertEqual(result["six_month_outlook"]["provenance"], "qualitative_scenario")
        self.assertIsNone(result["projected_condition_after_6_months"])

    def test_only_supported_candidates_and_reviewed_observations(self):
        findings = [{"type": name, "area_ratio": .12, "confidence": 95}
                    for name in ("stain", "pilling", "hole", "wrinkle", "fading", "fuzzing", "tear")]
        initial = assess("linen", findings)
        checked = assess("linen", findings, reviewed=True, quality_limited=True)
        for result in (initial, checked):
            observations = result["condition_assessment"]["observations"]
            self.assertEqual([entry["type"] for entry in observations], ["stain", "pilling", "hole"])
            self.assertTrue(all(entry["severity"] is None for entry in observations))
            self.assertIn("effect on the fibers are unknown", observations[0]["explanation"])
        self.assertEqual(initial["condition_assessment"]["provenance"], "model_predicted")
        self.assertFalse(any("mend" in r["action"] for r in initial["risk_alerts"]))
        self.assertEqual(checked["condition_assessment"]["provenance"], "user_reported")
        self.assertTrue(any("mend" in r["action"] for r in checked["risk_alerts"]))
        self.assertIn("limits the assessment", checked["condition_assessment"]["explanation"])
        self.assertTrue(any("pilling" in text for text in checked["six_month_outlook"]["concerns"]))

    def test_missing_detector(self):
        condition = assess("denim", [], detector_available=False)["condition_assessment"]
        self.assertEqual(condition["status"], "insufficient_evidence")
        self.assertEqual(condition["category"], "Damage assessment unavailable")

    def test_history_qualifies_scenario_without_predicting(self):
        result = assess("cotton", [], {"washing_frequency": "multiple_per_week", "previous_repairs": "Hem"})
        self.assertIn("wash frequency is multiple per week", result["six_month_outlook"]["assumptions"])
        self.assertEqual(result["user_inputs"]["previous_repairs"], "Hem")
        self.assertIsNone(result["estimated_remaining_months"])

    def test_professional_only_overrides_fabric_and_ancillary_settings(self):
        for fabric in ("cotton", "denim", "linen", "polyester", "silk", "wool"):
            with self.subTest(fabric=fabric):
                result = assess(fabric, [], {"composition": "100% wool", "care_label": {
                    "wash": "dry_clean_only", "dry": "tumble_low", "iron": "do_not_iron",
                    "bleach": "do_not_bleach",
                }}, fabric_confidence=99)
                laundry = result["laundry_settings"]
                self.assertEqual(laundry["laundry_mode"], "Dry clean only")
                self.assertEqual(laundry["dry_method"], "Check complete label.")
                self.assertEqual(laundry["iron_level"], "Do not iron.")
                self.assertEqual(laundry["bleach_warning"], "Do not bleach.")
                self.assertEqual(laundry["detergent"], "Unavailable")
                self.assertIsNone(laundry["max_temperature_c"])

    def test_wool_and_silk_need_label_and_wool_detergent_needs_composition(self):
        for fabric in ("silk", "wool"):
            unknown = assess(fabric, [], fabric_confidence=99)["laundry_settings"]
            hand = assess(fabric, [], {"care_label": {"wash": "hand"}}, fabric_confidence=99)["laundry_settings"]
            self.assertEqual(unknown["laundry_mode"], "Unknown")
            self.assertEqual(hand["laundry_mode"], "Hand wash only")
            self.assertIsNone(hand["max_temperature_c"])
            self.assertNotEqual(hand["detergent"], "Mild wool detergent")
        known = assess("wool", [], {"composition": "100% wool", "care_label": {
            "wash": "machine_very_mild", "max_temperature_c": 30, "dry": "flat",
        }})["laundry_settings"]
        self.assertEqual(known["max_temperature_c"], 30)
        self.assertEqual(known["laundry_mode"], "Very mild machine process")
        self.assertEqual(known["detergent"], "Mild wool detergent")
        self.assertEqual(known["dry_method"], "Dry flat as directed on the label.")

    def test_uncertain_class_and_conflicting_label(self):
        result = assess("cotton", [], {"care_label": {"wash": "conflicting"}}, fabric_confidence=5)
        self.assertTrue(result["care_guidance"]["recommendations"])
        self.assertEqual(result["laundry_settings"]["laundry_mode"], "Verify the care label")
        self.assertIsNone(result["laundry_settings"]["max_temperature_c"])
        self.assertTrue(any("conflicting" in r["action"] for r in result["risk_alerts"]))
        self.assertTrue(all(row["url"].startswith("https://") for row in result["references"]))
        contradictory = assess("silk", [], {"care_label": {
            "wash": "dry_clean_only", "professional": "do_not_dry_clean",
        }})["laundry_settings"]
        self.assertEqual(contradictory["laundry_mode"], "Verify the care label")
        self.assertEqual(contradictory["professional_care"], "Do not dry clean.")

    def test_scenario_a_conditional_cotton_and_pending_stain(self):
        result = assess("cotton", [{
            "type": "stain", "confidence": 80, "view": "front", "bbox": [.12, .2, .1, .1],
        }], fabric_confidence=62.5)
        self.assertTrue(result["fabric_needs_confirmation"])
        observation, = result["condition_assessment"]["observations"]
        self.assertFalse(observation["confirmed"])
        self.assertEqual(observation["location"], "Front photo · upper left")
        self.assertEqual(observation["model_score"], 80)
        advice = result["care_guidance"]["recommendations"]
        self.assertTrue(any(row["action"].startswith("Avoid strong stain") for row in advice))
        self.assertTrue(any(row["action"].startswith("If this garment is cotton") for row in advice))
        self.assertFalse(result["laundry_settings"]["confirmed_settings"])
        self.assertIsNone(result["health_score"])
        self.assertIsNone(result["estimated_remaining_months"])

    def test_scenario_b_reviewed_pilling_and_wool_label_limits(self):
        result = assess("wool", [{"type": "pilling", "view": "back"}], {
            "care_label": {"wash": "machine_mild", "max_temperature_c": 30},
            "use_frequency": "daily", "laundering_practice": "rubbing or wringing",
        }, reviewed=True, fabric_confidence=97)
        self.assertEqual(result["condition_assessment"]["category"], "Surface pilling confirmed")
        self.assertEqual(result["laundry_settings"]["confirmed_settings"]["Wash"], "Mild machine process")
        self.assertEqual(result["laundry_settings"]["max_temperature_c"], 30)
        self.assertTrue(any("mild wool detergent" in row["action"]
                            for row in result["laundry_settings"]["general_guidance"]))
        self.assertTrue(any("pilled area" in row["action"] for row in result["risk_alerts"]))
        self.assertIn("pilling", result["lifespan"]["summary"])
        self.assertTrue(any("frequent wear" in line for line in result["six_month_outlook"]["concerns"]))
        self.assertIn("rubbing or wringing", result["six_month_outlook"]["assumptions"])

    def test_scenario_c_unknown_material_still_has_next_steps(self):
        result = assess("uncertain", [])
        self.assertTrue(result["fabric_needs_confirmation"])
        self.assertIn("check seams", result["lifespan"]["summary"].lower())
        self.assertTrue(result["laundry_settings"]["general_guidance"])
        self.assertIsNone(result["health_score"])
        self.assertIsNone(result["laundry_settings"]["max_temperature_c"])
        self.assertIsNone(result["estimated_remaining_months"])

    def test_missing_label_offers_conditional_cotton_guidance(self):
        result = assess("cotton", [], {"care_label": {"wash": "unknown"}}, fabric_confidence=95)
        laundry = result["laundry_settings"]
        advice = laundry["general_guidance"]
        evidence, = [item for item in advice if "Q" in item["reference_ids"]]
        self.assertIn("you establish that it can be machine washed", evidence["action"])
        self.assertEqual(laundry["laundry_mode"], "Unknown")
        self.assertIsNone(laundry["max_temperature_c"])
        self.assertEqual(laundry["confirmed_settings"], {})
        self.assertIn("Q", {reference["id"] for reference in result["references"]})

    def test_missing_label_with_stain_and_opening_prioritizes_precautions(self):
        for reviewed in (True, False):
            with self.subTest(reviewed=reviewed):
                result = assess("cotton", [{"type": "hole"}, {"type": "stain"}],
                                reviewed=reviewed, fabric_confidence=95)
                directions = result["care_recommendations"]
                self.assertTrue(any("stain treatments and hot water" in action for action in directions))
                self.assertFalse(any("cooler, shorter cycle" in action for action in directions))
                self.assertEqual(result["laundry_settings"]["laundry_mode"], "Unknown")
                self.assertIsNone(result["laundry_settings"]["max_temperature_c"])
                self.assertIn("cleaning method has not been established", result["six_month_outlook"]["assumptions"])
                self.assertEqual(any("mend" in item["action"] for item in result["risk_alerts"]), reviewed)

    def test_scenario_d_dry_clean_only_blocks_wool_wash_advice(self):
        result = assess("wool", [{"type": "stain"}], {
            "composition": "100% wool", "care_label": {"wash": "dry_clean_only"},
        }, fabric_confidence=97)
        self.assertEqual(result["laundry_settings"]["confirmed_settings"], {"Wash": "Dry clean only"})
        self.assertIsNone(result["laundry_settings"]["max_temperature_c"])
        actions = " ".join(result["care_recommendations"])
        self.assertNotIn("detergent", actions)
        self.assertIn("leave home spot treatment on hold", actions)

    def test_input_validation(self):
        today = date(2026, 10, 20)
        for inputs in [[], {"condition": "invalid"}, {"use_frequency": []},
                       {"purchase_date_known": True}, {"purchase_date": "2027-01-01"},
                       {"purchase_date": "bad-date"}, {"purchase_date_known": "yes"},
                       {"care_label": {"wash": "hand", "max_temperature_c": True}},
                       {"care_label": {"wash": "professional_only", "max_temperature_c": 40}},
                       {"care_label": {"wash": []}}, {"composition": 100}]:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                normalize_user_inputs(inputs, today=today)
        unknown = normalize_user_inputs({"condition": "previously_used", "purchase_date_known": False}, today=today)
        self.assertIsNone(unknown["purchase_date"])


if __name__ == "__main__":
    unittest.main()
