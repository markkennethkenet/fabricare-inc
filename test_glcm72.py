import io
import unittest

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

from model_runtime import extract_glcm72_features


class Glcm72Tests(unittest.TestCase):
    def test_feature_order_and_preprocessing_match_training_recipe(self):
        pixels = np.random.default_rng(5).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        photo = Image.fromarray(pixels)
        encoded = io.BytesIO()
        photo.save(encoded, format="PNG")
        encoded.seek(0)
        features = extract_glcm72_features(encoded)

        gray = np.asarray(photo.convert("L").resize((128, 128))) // 8
        matrix = graycomatrix(gray, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                              levels=32, symmetric=True, normed=True)
        expected = np.concatenate([
            graycoprops(matrix, name).flatten() for name in
            ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
        ])
        self.assertEqual(features.shape, (1, 72))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[0], expected)
