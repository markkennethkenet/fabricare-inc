import unittest

import numpy as np

import garment_mask


def photo():
    """A yellow garment with a floor-coloured hole, lying on a pale tiled floor."""
    rng = np.random.default_rng(3)
    image = np.full((400, 300, 3), 225, np.uint8) + rng.integers(0, 12, (400, 300, 3), dtype=np.uint8)
    image[60:340, 70:230] = (20, 190, 230)          # garment (BGR yellow)
    image[180:200, 140:150] = (225, 225, 225)        # a hole showing the floor
    return image


class GarmentMaskTest(unittest.TestCase):
    def test_boxes_on_the_garment_stay_and_boxes_on_the_floor_go(self):
        outline = garment_mask.silhouette(photo())
        self.assertIsNotNone(outline)
        on_garment = [140 / 300, 175 / 400, 20 / 300, 30 / 400]   # around the hole
        on_floor = [5 / 300, 5 / 400, 50 / 300, 120 / 400]         # beside the sleeve
        self.assertGreater(garment_mask.share_on_garment(outline, on_garment), 0.9)
        self.assertLess(garment_mask.share_on_garment(outline, on_floor), 0.1)

    def test_photo_filled_by_fabric_is_not_segmented(self):
        close_up = np.full((300, 300, 3), (20, 190, 230), np.uint8)
        self.assertIsNone(garment_mask.silhouette(close_up))


if __name__ == "__main__":
    unittest.main()
