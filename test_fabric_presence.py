import os
import unittest
from unittest.mock import patch

import numpy as np

import fabric_presence


class FabricPresenceTest(unittest.TestCase):
    def test_textile_score_sums_only_clothing_and_fabric_classes(self):
        probabilities = np.zeros(1000)
        probabilities[411] = 0.3   # handkerchief (textile)
        probabilities[799] = 0.7   # sliding door (not textile)
        self.assertAlmostEqual(fabric_presence.textile_score(probabilities), 0.3)
        self.assertEqual(len(set(fabric_presence.TEXTILE_CLASS_IDS)), 51)

    def test_switched_off_or_unavailable_never_blocks(self):
        with patch.dict(os.environ, {"FABRIC_PRESENCE": "off"}):
            self.assertEqual(fabric_presence.check("any.jpg")["fabric_detected"], True)
        with patch.dict(os.environ, {"FABRIC_PRESENCE": "on"}), patch.object(fabric_presence, "_load", return_value=None):
            result = fabric_presence.check("any.jpg")
        self.assertEqual((result["checked"], result["fabric_detected"]), (False, True))

    def test_threshold_can_be_tuned(self):
        with patch.dict(os.environ, {"FABRIC_PRESENCE_MIN": "0.1"}):
            self.assertEqual(fabric_presence.min_score(), 0.1)
        with patch.dict(os.environ, {"FABRIC_PRESENCE_MIN": "oops"}):
            self.assertEqual(fabric_presence.min_score(), fabric_presence.DEFAULT_MIN_TEXTILE_SCORE)


if __name__ == "__main__":
    unittest.main()
