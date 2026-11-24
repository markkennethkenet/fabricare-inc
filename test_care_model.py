import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier

import care_model
from care_model import (CLASSES, FEATURE_NAMES, CareRecommender, care_alerts, color_features,
                        damage_features, feature_row, glcm_summary, reference_laundry_mode)


def png(color, size=(120, 90)):
    data = BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    data.seek(0)
    return data


def row(fabric, detections=(), color=(230, 225, 215)):
    return feature_row(fabric, damage_features(list(detections)), color_features(png(color)),
                       glcm_summary(np.arange(72, dtype=float)))


def finding(kind, confidence=95, area=0.02):
    return {"type": kind, "confidence": confidence, "area_ratio": area}


class CareFeatureTests(unittest.TestCase):
    def test_feature_row_covers_every_feature_in_order(self):
        values = row("cotton")
        self.assertEqual(list(values)[:len(FEATURE_NAMES)], FEATURE_NAMES)
        self.assertEqual(list(values)[len(FEATURE_NAMES):], ["discoloration_count", "discoloration_max_area"])
        self.assertEqual(care_model.feature_vector(values).shape, (1, len(FEATURE_NAMES)))

    def test_damage_counts_only_confident_or_confirmed_findings(self):
        features = damage_features([finding("hole", 92, 0.03), finding("hole", 55, 0.5), finding("stain", 70, 0.1),
                                    finding("pilling", 65), {**finding("pilling", 30), "confirmed": True}])
        self.assertEqual(features["hole_count"], 1)  # 55% is below the 60% hole threshold
        self.assertAlmostEqual(features["hole_max_area"], 0.03)
        self.assertEqual(features["stain_count"], 1)
        self.assertEqual(features["pilling_count"], 1)  # only the confirmed one
        self.assertEqual(features["needlemark_count"], 0)

    def test_thresholds_match_the_detector_study(self):
        self.assertEqual(care_model.DAMAGE_MIN_CONFIDENCE,
                         {"hole": 60.0, "stain": 70.0, "pilling": 70.0, "needlemark": 60.0,
                          "discoloration": 60.0})
        self.assertTrue(care_model.finding_counts("hole", 65))   # a real slit on a phone photo
        self.assertFalse(care_model.finding_counts("hole", 59))
        self.assertTrue(care_model.finding_counts("hole", 20, confirmed=True))
        self.assertFalse(care_model.finding_counts("wrinkle", 99))
        self.assertFalse(care_model.finding_counts("stain", None))

    def test_discoloration_drives_the_rules_but_not_the_model_features(self):
        clean = row("cotton", color=(235, 230, 220))
        faded = row("cotton", [finding("discoloration", 72, 0.08)], (235, 230, 220))
        self.assertEqual(faded["discoloration_count"], 1)
        self.assertNotIn("discoloration_count", FEATURE_NAMES)
        self.assertTrue((care_model.feature_vector(faded) == care_model.feature_vector(clean)).all())
        self.assertEqual(reference_laundry_mode(clean)[0], "machine_normal_warm")
        self.assertEqual(reference_laundry_mode(faded)[0], "machine_gentle_cold")
        self.assertIn("Colour change", [a["title"] for a in care_model.care_alerts(faded)])
        self.assertTrue(any("bleach" in tip["text"] for tip in care_model.care_tips(faded)))
        weak = row("cotton", [finding("discoloration", 45, 0.08)], (235, 230, 220))
        self.assertEqual(weak["discoloration_count"], 0)   # below 60% it stays a finding to review
        old_row = {k: v for k, v in clean.items() if not k.startswith("discoloration")}
        self.assertEqual(care_model.care_alerts(old_row), care_model.care_alerts(clean))   # rows from before V7.1

    def test_colour_features_separate_dark_and_light(self):
        dark, light = color_features(png((20, 20, 30))), color_features(png((235, 230, 220)))
        self.assertLess(dark["luminance_mean"], 40)
        self.assertGreater(dark["dark_fraction"], 0.9)
        self.assertGreater(light["luminance_mean"], 200)
        self.assertLess(light["saturation_mean"], 0.1)

    def test_glcm_summary_averages_each_property(self):
        summary = glcm_summary(np.arange(72, dtype=float))
        self.assertAlmostEqual(summary["glcm_contrast"], 5.5)
        self.assertAlmostEqual(summary["glcm_ASM"], 65.5)


class ReferenceRuleTests(unittest.TestCase):
    def test_rules_follow_fabric_condition_and_colour(self):
        cases = [
            (row("cotton"), "machine_normal_warm"),
            (row("linen"), "machine_normal_warm"),
            (row("cotton", color=(25, 25, 40)), "machine_gentle_cold"),
            (row("cotton", [finding("pilling")]), "machine_gentle_cold"),
            (row("denim"), "machine_gentle_cold"),
            (row("polyester"), "machine_gentle_cold"),
            (row("wool"), "hand_wash_cold"),
            (row("silk"), "hand_wash_cold"),
            (row("silk", [finding("stain")]), "professional_clean"),
            (row("wool", [finding("hole")]), "repair_first"),
            (row("uncertain"), "hand_wash_cold"),
            # An unidentified fabric gets the gentlest method even when dark or pilled.
            (row("uncertain", [finding("pilling")], color=(25, 25, 40)), "hand_wash_cold"),
            (row("uncertain", [finding("hole")]), "repair_first"),
        ]
        for values, expected in cases:
            mode, reasons = reference_laundry_mode(values)
            self.assertEqual(mode, expected)
            self.assertIn(mode, CLASSES)
            self.assertTrue(all(refs for _, refs in reasons))

    def test_alerts_cover_findings_and_fabric_risks(self):
        alerts = care_alerts(row("wool", [finding("hole"), finding("stain")], color=(30, 30, 30)))
        titles = {alert["title"] for alert in alerts}
        self.assertTrue({"Possible opening", "Possible stain", "Colour may bleed", "Felting risk"} <= titles)
        self.assertEqual(alerts[0]["level"], "high")
        self.assertTrue(all(alert["reference_ids"] for alert in alerts))
        self.assertEqual(care_alerts(row("cotton")), [])


class CareAdviceTests(unittest.TestCase):
    def label(self, **care_label):
        return {"care_label": {"wash": "unknown", "dry": "unknown", "iron": "unknown", "bleach": "unknown",
                               "professional": "unknown", "max_temperature_c": None, **care_label}}

    def test_never_more_than_three_sentences(self):
        busy = row("denim", [finding("hole"), finding("stain", 80), finding("pilling", 80), finding("needlemark", 70)],
                   color=(20, 25, 60))
        inputs = {"use_frequency": "daily", "washing_frequency": "multiple_per_week",
                  "laundering_practice": "rubbing or wringing", **self.label()}
        tips = care_model.care_tips(busy, inputs)
        _, laundry = care_model.laundry_tips(busy, "repair_first", inputs)
        self.assertEqual(len(tips), 3)
        self.assertLessEqual(len(laundry), 3)
        for tip in tips + laundry:
            self.assertEqual(tip["text"].count(". "), 0, tip["text"])  # one sentence each
            self.assertTrue(tip["reference_ids"])

    def test_damage_comes_before_fabric_advice(self):
        tips = care_model.care_tips(row("wool", [finding("hole")]))
        self.assertIn("Mend the opening", tips[0]["text"])
        self.assertTrue(any("wool" in tip["text"] for tip in tips))

    def test_advice_follows_fabric_and_colour(self):
        denim = care_model.care_tips(row("denim", color=(20, 25, 60)))
        self.assertTrue(any("transfer dye" in tip["text"] for tip in denim))
        self.assertFalse(any("dark colours" in tip["text"] for tip in denim))  # no repeated dye advice
        dark_cotton = care_model.care_tips(row("cotton", color=(20, 25, 30)))
        self.assertTrue(any("dark colours" in tip["text"] for tip in dark_cotton))
        silk = care_model.care_tips(row("silk", [finding("stain", 80)]))
        self.assertIn("professional cleaner", silk[0]["text"])
        headline, laundry = care_model.laundry_tips(row("wool"), "hand_wash_cold")
        self.assertEqual(headline, care_model.LAUNDRY_MODES["hand_wash_cold"])
        self.assertIn("wool-safe detergent", laundry[1]["text"])
        self.assertIn("flat", laundry[2]["text"])

    def test_dark_garments_are_washed_inside_out(self):
        _, laundry = care_model.laundry_tips(row("cotton", color=(25, 25, 30)), "machine_gentle_cold")
        self.assertTrue(laundry[0]["text"].startswith("Turn it inside out, then machine wash"))

    def test_entered_care_label_sets_the_limits(self):
        inputs = self.label(wash="machine_mild", max_temperature_c=40, dry="do_not_tumble", iron="low")
        headline, laundry = care_model.laundry_tips(row("cotton"), "machine_gentle_cold", inputs)
        self.assertEqual(headline, "Machine wash · gentle cycle · per your care label")
        self.assertIn("at up to 30 °C", laundry[0]["text"])  # a cold suggestion stays below the label limit
        self.assertIn("not in a tumble dryer", laundry[2]["text"])
        self.assertIn("iron on low heat", laundry[2]["text"])
        warm = care_model.laundry_tips(row("cotton"), "machine_normal_warm",
                                       self.label(wash="machine_normal", max_temperature_c=60))[1]
        self.assertIn("at up to 60 °C", warm[0]["text"])
        dry_clean = care_model.laundry_tips(row("cotton"), "machine_normal_warm", self.label(wash="dry_clean_only"))
        self.assertEqual(dry_clean[0], "Dry clean · per your care label")
        self.assertEqual(len(dry_clean[1]), 2)
        self.assertIn("professionally cleaned", dry_clean[1][0]["text"])

    def test_unconfirmed_fabric_gets_gentle_advice(self):
        tips = care_model.care_tips(row("polyester"), fabric_confirmed=False)
        self.assertIn("fibre label", tips[0]["text"])
        self.assertFalse(any("polyester" in tip["text"] for tip in tips))
        headline, laundry = care_model.laundry_tips(row("polyester"), "machine_normal_warm", fabric_confirmed=False)
        self.assertEqual(headline, care_model.LAUNDRY_MODES["hand_wash_cold"])
        self.assertTrue(laundry[0]["text"].startswith("Until the fabric is confirmed, hand wash"))


class CareRecommenderTests(unittest.TestCase):
    def bundle(self, **changes):
        rows = [row(f) for f in ("cotton", "wool", "denim")] + [row("cotton", [finding("hole")])]
        x = np.vstack([care_model.feature_vector(r) for r in rows])
        y = [CLASSES.index(reference_laundry_mode(r)[0]) for r in rows]
        y_all = list(range(len(CLASSES)))
        x_all = np.vstack([x, np.zeros((len(CLASSES), x.shape[1]))])
        model = DecisionTreeClassifier(random_state=0).fit(x_all, y + y_all)
        return {"model": model, "feature_names": FEATURE_NAMES, "classes": list(CLASSES),
                "name": "Random Forest", **changes}

    def save(self, bundle):
        path = Path(tempfile.mkdtemp()) / "care.joblib"
        joblib.dump(bundle, path)
        return path

    def test_loads_and_predicts_a_laundry_mode_with_confidence(self):
        recommender = CareRecommender(self.save(self.bundle()))
        result = recommender.predict(row("wool"))
        self.assertEqual(result["laundry_mode"], "hand_wash_cold")
        self.assertEqual(result["confidence"], 100.0)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=3)
        self.assertEqual(recommender.name, "Random Forest")

    def test_rejects_bundles_with_other_features_or_classes(self):
        with self.assertRaisesRegex(RuntimeError, "features"):
            CareRecommender(self.save(self.bundle(feature_names=FEATURE_NAMES[:-1])))
        with self.assertRaisesRegex(RuntimeError, "classes"):
            CareRecommender(self.save(self.bundle(classes=list(reversed(CLASSES)))))


if __name__ == "__main__":
    unittest.main()
