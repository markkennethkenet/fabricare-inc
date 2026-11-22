import unittest
from collections import Counter

from scripts import v7_detector_data as v7


class V7DetectorDataTest(unittest.TestCase):
    def test_garment_prefix_survives_roboflow_renaming(self):
        name = v7.prelabel_name("Tshirt_003", "Holes ", "IMG_20261122_1.jpg")
        self.assertEqual(name, "tshirt_003__holes__img_20261122_1.jpg")
        exported = "tshirt_003__holes__img_20261122_1_jpg.rf.0a1b2c3d4e.jpg"
        self.assertEqual(v7.garment_of(exported), "tshirt_003")
        self.assertTrue(v7.garment_of("random.jpg").startswith("photo:"))

    def test_split_keeps_garments_together_and_is_close_to_70_15_15(self):
        names = [f"g{g:02d}__front__p{i}_jpg.rf.{g}{i}.jpg" for g in range(40) for i in range(5)]
        split = v7.split_by_garment(names)
        for g in range(40):
            self.assertEqual(len({split[n] for n in names if n.startswith(f"g{g:02d}__")}), 1)
        counts = Counter(split.values())
        self.assertEqual(counts, {"train": 140, "valid": 30, "test": 30})
        self.assertEqual(split, v7.split_by_garment(list(reversed(names))))

    def test_public_classes_map_to_the_app_order(self):
        text = "0 .5 .5 .1 .1\n4 .2 .2 .1 .1\n5 .3 .3 .1 .1\n3 .4 .4 .2 .2\nbad line\n"
        self.assertEqual(v7.remap_label_lines(text, v7.PUBLIC_TO_APP), "0 .5 .5 .1 .1\n3 .4 .4 .2 .2\n")
        self.assertEqual(v7.remap_label_lines("5 .1 .1 .1 .1\n", v7.PUBLIC_TO_APP), "")

    def test_roboflow_class_names_in_any_order(self):
        mapping = v7.roboflow_mapping(["Stain", "hole", "needle_mark", "wrinkle", "pilling"])
        self.assertEqual(mapping, {0: 3, 1: 0, 2: 1, 4: 2})


if __name__ == "__main__":
    unittest.main()
