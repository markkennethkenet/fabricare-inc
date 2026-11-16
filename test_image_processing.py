import unittest
import numpy as np
from image_processing import quality_for, merge_detections, aggregate_fabrics

class ImageProcessingTests(unittest.TestCase):
    def test_quality_dark_bright_blurry_and_clear(self):
        self.assertIn('Too dark', quality_for(np.zeros((50, 50, 3), dtype=np.uint8))['reason'])
        self.assertIn('Too bright', quality_for(np.full((50, 50, 3), 255, dtype=np.uint8))['reason'])
        self.assertIn('blurry', quality_for(np.full((50, 50, 3), 120, dtype=np.uint8))['reason'])
        image = np.random.default_rng(7).integers(50, 200, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(quality_for(image)['passed'])

    def test_merge_same_image_only(self):
        def item(id, kind='future_class', image=0, bbox=None, confidence=90):
            return {'id': id, 'type': kind, 'image_index': image, 'view': 'front',
                    'bbox': bbox or [.1, .1, .2, .2], 'confidence': confidence}
        detections = [item('0:0'), item('0:1', confidence=80), item('1:0', image=1),
                      item('0:2', kind='other'), item('0:3', bbox=[.7, .7, .1, .1])]
        merged = merge_detections(detections)
        self.assertEqual({d['id'] for d in merged}, {'0:0', '1:0', '0:2', '0:3'})
        self.assertEqual(merge_detections([]), [])

    def test_probability_aggregation_and_vote_fallback(self):
        result = aggregate_fabrics([{'probabilities': {'a': .8, 'b': .2}},
                                   {'probabilities': {'a': .4, 'b': .6}}])
        self.assertEqual(result['fabric_type'], 'a')
        self.assertEqual(result['fabric_confidence'], 60)
        voted = aggregate_fabrics([{'probabilities': {'new_class': 1}},
                                  {'probabilities': {'new_class': 1}}, {'probabilities': {'other': 1}}])
        self.assertEqual(voted['fabric_type'], 'new_class')
        self.assertAlmostEqual(voted['fabric_confidence'], 66.67)

    def test_conflicting_close_up_cannot_override_agreeing_front_and_back(self):
        predictions = [
            {'view': 'front', 'fabric_type': 'polyester',
             'probabilities': {'polyester': .55, 'denim': .45}},
            {'view': 'back', 'fabric_type': 'polyester',
             'probabilities': {'polyester': .52, 'denim': .48}},
            {'view': 'close_up', 'fabric_type': 'denim',
             'probabilities': {'polyester': .01, 'denim': .99}},
        ]
        result = aggregate_fabrics(predictions)
        self.assertEqual(result['fabric_type'], 'polyester')
        self.assertEqual(result['fabric_confidence'], 53.5)
        self.assertTrue(result['fabric_close_up_excluded'])
        predictions[1]['quality'] = {'passed': False}
        for views in (predictions, [predictions[0], predictions[2]]):
            result = aggregate_fabrics(views)
            self.assertTrue(result['fabric_close_up_excluded'])
            self.assertEqual(result['fabric_confidence'], 55)

    def test_blurry_optional_photo_does_not_override_clear_front(self):
        predictions = [
            {'view': 'front', 'fabric_type': 'cotton',
             'probabilities': {'cotton': .62, 'linen': .38}, 'quality': {'passed': True}},
            {'view': 'back', 'fabric_type': 'linen',
             'probabilities': {'cotton': .01, 'linen': .99}, 'quality': {'passed': False}},
        ]
        result = aggregate_fabrics(predictions)
        self.assertEqual(result['fabric_type'], 'cotton')
        self.assertEqual(result['fabric_confidence'], 62)

    def test_disagreeing_front_and_back_are_uncertain_even_with_a_majority(self):
        predictions = [
            {'view': 'front', 'fabric_type': 'polyester', 'probabilities': {'polyester': 1}},
            {'view': 'back', 'fabric_type': 'denim', 'probabilities': {'denim': 1}},
            {'view': 'close_up', 'fabric_type': 'polyester', 'probabilities': {'polyester': 1}},
        ]
        result = aggregate_fabrics(predictions)
        self.assertEqual(result['fabric_type'], 'uncertain')
        self.assertEqual(result['fabric_uncertainty_reason'], 'front_back_disagree')
        self.assertGreater(result['fabric_probabilities']['polyester'], .5)
        self.assertEqual(aggregate_fabrics(predictions[:1])['fabric_type'], 'polyester')

if __name__ == '__main__':
    unittest.main()
