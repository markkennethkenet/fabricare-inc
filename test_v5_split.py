import csv
import hashlib
import unittest
from collections import Counter
from pathlib import Path

from scripts import v5_split

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "research" / "v3_split" / "split_manifest_70_20_10.csv"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def photo(fabric, filename, key=None):
    return {"fabric": fabric, "filename": filename, "sha256": key or sha(fabric + filename),
            "path": f"raw/{fabric}/{filename}"}


class GroupIdTest(unittest.TestCase):
    def test_reproduces_every_v3_group(self):
        rows = list(csv.DictReader(MANIFEST.read_text(encoding="utf-8-sig").splitlines()))
        self.assertEqual(len(rows), 3462)
        for row in rows:
            self.assertEqual(v5_split.group_id(row["fabric"], row["filename"]), row["group_id"], row["filename"])

    def test_burst_frames_and_copies_share_a_group(self):
        names = ["IMG_20261120_101010.jpg", "IMG_20261120_101010_Burst02.jpg", "IMG_20261120_101010 (1).jpg"]
        self.assertEqual({v5_split.group_id("silk", n) for n in names}, {"silk:img_20261120_101010"})
        self.assertEqual(v5_split.group_id("wool", "wool_g0042_007.jpg"), "wool:garment_0042")


class BuildSplitTest(unittest.TestCase):
    def setUp(self):
        self.v3 = [{**photo("cotton", "IMG_20260101_000000.jpg"), "split": "test",
                    "group_id": "cotton:img_20260101_000000"},
                   {**photo("silk", "silk_g0001_001.jpg"), "split": "train", "group_id": "silk:garment_0001"}]

    def test_v3_photos_keep_their_split_and_new_photos_are_added(self):
        new = [photo("linen", f"linen_g{g:04d}_{i:03d}.jpg") for g in range(1, 21) for i in range(5)]
        records, report = v5_split.build_split(self.v3, self.v3 + new)
        by_sha = {r["sha256"]: r for r in records}
        self.assertEqual(by_sha[self.v3[0]["sha256"]]["split"], "test")
        self.assertEqual(report["new_photos"], 100)
        counts = Counter(r["split"] for r in records if r["source"] == "new")
        self.assertEqual(counts, {"train": 70, "valid": 20, "test": 10})

    def test_new_burst_frame_follows_its_v3_group(self):
        frame = photo("cotton", "IMG_20260101_000000_Burst03.jpg")
        records, _ = v5_split.build_split(self.v3, self.v3 + [frame])
        self.assertEqual(next(r for r in records if r["sha256"] == frame["sha256"])["split"], "test")

    def test_audit_exclusions_cross_class_copies_and_duplicates(self):
        quarantined = photo("linen", "Silk and Linen Scrim.jpg")
        twin = photo("wool", "same.jpg", key=sha("shared"))
        twin_elsewhere = photo("silk", "same.jpg", key=sha("shared"))
        copy = {**self.v3[1], "filename": "silk_g0001_001 (1).jpg"}
        records, report = v5_split.build_split(
            self.v3, self.v3 + [quarantined, twin, twin_elsewhere, copy], excluded_sha256=[quarantined["sha256"]])
        self.assertEqual(len(records), 2)
        self.assertEqual(report["excluded_by_v3_audit"], 1)
        self.assertEqual(report["excluded_cross_class_copy"], 2)
        self.assertEqual(report["duplicate_copy_ignored"], 1)

    def test_missing_v3_photo_is_reported_not_fatal(self):
        records, report = v5_split.build_split(self.v3, self.v3[1:])
        self.assertEqual(len(records), 1)
        self.assertEqual(report["v3_photo_missing_from_drive"], 1)

    def test_polyester_additions_stay_in_training(self):
        extra = photo("polyester", "poly-web-01.jpg")
        records, _ = v5_split.build_split(self.v3, self.v3 + [extra], polyester_additions=[extra["sha256"]])
        added = next(r for r in records if r["sha256"] == extra["sha256"])
        self.assertEqual((added["split"], added["source"]), ("train", "polyester_addition"))

    def test_assignment_ignores_upload_order(self):
        new = [photo("denim", f"denim photo {i}.jpg") for i in range(30)]
        first, _ = v5_split.build_split(self.v3, self.v3 + new)
        second, _ = v5_split.build_split(self.v3, list(reversed(self.v3 + new)))
        self.assertEqual({r["sha256"]: r["split"] for r in first}, {r["sha256"]: r["split"] for r in second})


if __name__ == "__main__":
    unittest.main()
