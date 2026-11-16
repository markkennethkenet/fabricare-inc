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


def finding(kind, confidence=80, area=0.02):
    return {"type": kind, "confidence": confidence, "area_ratio": area}


class CareFeatureTests(unittest.TestCase):
    def test_feature_row_covers_every_feature_in_order(self):
        values = row("cotton")
        self.assertEqual(list(values), FEATURE_NAMES)
        self.assertEqual(care_model.feature_vector(values).shape, (1, len(FEATURE_NAMES)))

    def test_damage_counts_only_confident_findings(self):
        features = damage_features([finding("hole", 90, 0.03), finding("hole", 20, 0.5), finding("stain", 45, 0.1)])
        self.assertEqual(features["hole_count"], 1)
        self.assertAlmostEqual(features["hole_max_area"], 0.03)
        self.assertEqual(features["stain_count"], 1)
        self.assertEqual(features["pilling_count"], 0)

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
