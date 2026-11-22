import hashlib
import unittest
from collections import Counter

import damage_condition
from scripts import damage_split

COLOR = {"luminance_mean": 180.0, "luminance_std": 20.0, "saturation_mean": 0.1, "dark_fraction": 0.05}
GLCM = {f"glcm_{p}": 0.5 for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")}


def photo(source, folder, filename, garment=None, key=None):
    item = {"source": source, "folder": folder, "filename": filename,
            "path": f"{source}/{garment or ''}/{folder}/{filename}",
            "sha256": key or hashlib.sha256(f"{source}{garment}{folder}{filename}".encode()).hexdigest()}
    if garment:
        item["garment"] = garment
    return item


class FolderNamesTest(unittest.TestCase):
    def test_drive_folder_names_map_to_classes(self):
        cases = {"Holes ": "hole", " Stains": "stain", "Needle_Marks": "needlemark", "Normal_Fabric": "normal",
                 "Normal": "normal", "Pilling": "pilling", "Wrinkles": "wrinkle", "Multiple_Damage": "multiple"}
        for folder, label in cases.items():
            self.assertEqual(damage_split.folder_class(folder), label, folder)
        self.assertIsNone(damage_split.folder_class("Discoloration"))


class DamageSplitTest(unittest.TestCase):
    def test_own_photos_split_by_tshirt_and_online_near_70_20_10(self):
        own = [photo("own", "Holes ", f"h{i}.jpg", garment=f"Tshirt_00{g}") for g in range(1, 7) for i in range(3)]
        online = [photo("online", "Stains", f"stain_{i}.jpg") for i in range(100)]
        records, _ = damage_split.build_split(own + online)
        for r in records:
            if r["source"] == "own":
                self.assertEqual(r["split"], damage_split.OWN_GARMENT_SPLIT[r["garment"].casefold()])
        counts = Counter(r["split"] for r in records if r["source"] == "online")
        self.assertEqual(counts, {"train": 70, "valid": 20, "test": 10})

    def test_multiple_damage_is_a_separate_check_set(self):
        records, _ = damage_split.build_split([photo("own", "Multiple_Damage", "m.jpg", garment="Tshirt_001")])
        self.assertEqual(records[0]["split"], "multiple")

    def test_conflicting_labels_and_copies(self):
        a = photo("online", "Holes", "x.jpg", key="a" * 64)
        b = photo("online", "Stains", "x.jpg", key="a" * 64)
        c = photo("online", "Holes", "y.jpg", key="c" * 64)
        d = photo("online", "Holes", "y (1).jpg", key="c" * 64)
        unknown = photo("online", "Discoloration", "z.jpg")
        records, report = damage_split.build_split([a, b, c, d, unknown])
        self.assertEqual(len(records), 1)
        self.assertEqual(report["excluded_conflicting_labels"], 2)
        self.assertEqual(report["duplicate_copy_ignored"], 1)
        self.assertEqual(report["unknown_folder:Discoloration"], 1)

    def test_roboflow_versions_share_a_group(self):
        names = ["IMG_1_jpg.rf.abc123.jpg", "IMG_1_jpg.rf.def456.jpg"]
        self.assertEqual(len({damage_split.online_group("hole", n) for n in names}), 1)


class CareFeaturesTest(unittest.TestCase):
    def test_condition_drives_the_referenced_rules(self):
        modes = {condition: damage_condition.reference_laundry_mode(
                     damage_condition.feature_row("cotton", condition, COLOR, GLCM))[0]
                 for condition in damage_condition.DAMAGE_CLASSES}
        self.assertEqual(modes["hole"], "repair_first")
        self.assertEqual(modes["pilling"], "machine_gentle_cold")
        self.assertEqual(modes["normal"], "machine_normal_warm")
        self.assertEqual(modes["wrinkle"], "machine_normal_warm")
        silk_stain = damage_condition.feature_row("silk", "stain", COLOR, GLCM)
        self.assertEqual(damage_condition.reference_laundry_mode(silk_stain)[0], "professional_clean")

    def test_feature_vector_shape_and_unknown_condition(self):
        row = damage_condition.feature_row("wool", "normal", COLOR, GLCM)
        self.assertEqual(damage_condition.feature_vector(row).shape, (1, len(damage_condition.FEATURE_NAMES)))
        with self.assertRaises(ValueError):
            damage_condition.feature_row("wool", "torn", COLOR, GLCM)


if __name__ == "__main__":
    unittest.main()
