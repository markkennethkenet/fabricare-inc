"""API contract tests use fake inference; production model files are never loaded."""
import io
import json
import importlib
import os
from pathlib import Path
from PIL import Image
import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import care_model
from model_runtime import FabricDecisionTree

class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch inference construction before importing the Flask application.
        with patch.dict(os.environ, FABRIC_CLASSIFIER='legacy'), patch('joblib.load'), patch('ultralytics.YOLO'):
            cls.api = importlib.import_module('fabricare_api')
        cls.client = cls.api.app.test_client()

    def setUp(self):
        # db_config loads .env, which may hold a real GEMINI_API_KEY: tests never call Gemini.
        gemini_off = patch.dict(os.environ, {'GEMINI_API_KEY': ''})
        gemini_off.start()
        self.addCleanup(gemini_off.stop)
        self.fabric = patch.object(self.api, 'classify_fabric', return_value={
            'fabric_type': 'future_fabric', 'confidence': 85,
            'probabilities': {'future_fabric': .85, 'other': .15}})
        self.damage = patch.object(self.api, 'detect_damage', return_value=[])
        self.fabric_mock = self.fabric.start()
        self.damage.start()
        self.addCleanup(self.fabric.stop)
        self.addCleanup(self.damage.stop)

    def photo(self, dark=False):
        image = np.zeros((100, 100, 3), dtype=np.uint8) if dark else np.random.default_rng(7).integers(50, 200, (100, 100, 3), dtype=np.uint8)
        return io.BytesIO(cv2.imencode('.png', image)[1].tobytes()), 'photo.png'

    def test_legacy_single_image_including_low_quality(self):
        response = self.client.post('/analyze', data={'image': self.photo(dark=True)})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['fabric_type'], 'future_fabric')
        self.assertIn('estimated_lifespan', body)
        self.assertIn('care_recommendation', body)
        self.assertEqual(body['per_image'][0]['view'], 'front')
        self.assertTrue(body['quality_override'])

    def test_scan_endpoint_uses_trained_models_without_external_ai(self):
        with patch.object(self.api, 'classify_fabric', return_value={
            'fabric_type': 'cotton', 'confidence': 91, 'probabilities': {'cotton': .91}
        }):
            response = self.client.post('/predict/hybrid', data={'image': self.photo()})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['fabric']['verification_status'], 'model_result')
        self.assertEqual(body['damage']['verification_status'], 'no_visible_damage')
        self.assertIn('damages', body)

    def test_selected_models_run_real_inference_on_a_fabric_photo(self):
        photo = Path(__file__).parent / 'model' / '2b2a6f82-68e1-4d0f-ad22-a011dd8df367.jpg'
        expected = {
            'cnn': 'ConvNeXt-Tiny (V4)',
            'svm': 'GLCM + SVM',
            'glcm_dt': 'GLCM + Decision Tree',
        }
        for model_type, name in expected.items():
            with self.subTest(model_type=model_type):
                response = self.client.post('/predict/hybrid', data={
                    'image': (io.BytesIO(photo.read_bytes()), 'fabric.jpg'),
                    'model_type': model_type,
                    'allow_low_quality': 'true',
                    'ai_assist': 'true',
                })
                self.assertEqual(response.status_code, 200, response.get_json())
                body = response.get_json()
                self.assertEqual(body['model_type'], model_type)
                self.assertEqual(body['fabric_model'], name)
                opinion = body['fabric_second_opinion']
                self.assertIn(opinion['status'], {'agrees', 'disagrees'})
                self.assertNotEqual(opinion['model'], name)
                if opinion['status'] == 'disagrees':
                    self.assertTrue(body['fabric_needs_confirmation'])
                self.assertEqual(list(self.api.classifier_cache[model_type].labels),
                                 ['cotton', 'denim', 'linen', 'polyester', 'silk', 'wool'])
                self.assertIn(body['fabric_type'], self.api.classifier_cache[model_type].labels)
                self.assertTrue(0 <= body['health_score'] <= 100)
                self.assertIn(body['health_grade'], {'Excellent', 'Good', 'Fair', 'Poor'})
                self.assertIsInstance(body['estimated_remaining_months'], int)
                self.assertEqual(body['lifespan']['status'], 'experimental_estimate')
                care = body['ai_care']
                self.assertEqual(care['status'], 'model_suggestion')
                self.assertEqual(care['model'], 'Decision Tree')
                self.assertTrue(0 < care['confidence'] <= 100)
                self.assertIn(care['laundry_mode'], care_model.CLASSES)
                self.assertTrue(care['label'] and care['reasons'])
                self.assertIsInstance(care['alerts'], list)
                self.assertTrue(1 <= len(care['care_tips']) <= 3 and 1 <= len(care['laundry_tips']) <= 3)
                self.assertTrue(care['headline'])
                self.assertEqual(care['provided_by'], f'Decision Tree care model · {name} fabric result')
                cited = {ref for entry in care['reasons'] + care['alerts'] + care['care_tips'] + care['laundry_tips']
                         for ref in entry['reference_ids']}
                self.assertTrue(cited <= {ref['id'] for ref in body['references']})
                self.assertEqual(body['schema_version'], 3)
                self.assertIn('care_recommendation', body)
                # Objective 3: every fabric model reports a confidence score.
                self.assertIsInstance(body['fabric_confidence'], (int, float))
                self.assertAlmostEqual(sum(body['fabric_probabilities'].values()), 1, places=3)
                if model_type == 'svm':
                    self.assertEqual(body['per_image'][0]['score_kind'], 'svm_platt_probability')
                if model_type == 'cnn':
                    self.assertEqual(body['per_image'][0]['inference_views'], 2)
                    # Label smoothing caps the V4 CNN's scores near 92%, so it has its own threshold.
                    self.assertEqual(body['fabric']['threshold_percent'], 60)
        self.fabric_mock.assert_not_called()

    def test_invalid_model_type_is_rejected(self):
        with patch.object(self.api, 'load_selected_classifier') as load:
            response = self.client.post('/predict/hybrid', data={
                'image': self.photo(), 'model_type': 'unknown',
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid model_type', response.get_json()['error'])
        load.assert_not_called()

    def test_unavailable_model_has_no_fallback(self):
        with patch.object(self.api, 'BASE_DIR', Path(__file__).parent / 'missing_models'), \
                patch.dict(self.api.classifier_cache, clear=True):
            response = self.client.post('/predict/hybrid', data={
                'image': self.photo(), 'model_type': 'svm',
            })
        self.assertEqual(response.status_code, 503)
        self.assertIn('Missing GLCM + SVM model file: glcm_svm.joblib',
                      response.get_json()['error'])
        self.fabric_mock.assert_not_called()

    def test_hybrid_endpoint_rejects_unsupported_image_format_before_inference(self):
        self.fabric_mock.reset_mock()
        for endpoint in ('/quality', '/predict/hybrid'):
            with self.subTest(endpoint=endpoint):
                encoded = io.BytesIO()
                Image.new('RGB', (100, 100), 'white').save(encoded, format='BMP')
                encoded.seek(0)
                response = self.client.post(endpoint, data={'image': (encoded, 'photo.bmp')})
                self.assertEqual(response.status_code, 400)
                self.assertIn('JPG and PNG', response.get_json()['error'])
        self.fabric_mock.assert_not_called()

    def test_multiple_images_and_per_image_predictions(self):
        response = self.client.post('/analyze', data={'images': [self.photo(), self.photo()], 'views': '["front", "back"]'})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body['per_image']), 2)
        self.assertEqual(body['per_image'][1]['view'], 'back')
        self.assertEqual(body['per_image'][1]['fabric_type'], 'future_fabric')
        self.assertNotIn('path', body['per_image'][0])
        self.assertEqual(self.fabric_mock.call_count, 2)

    def test_quality_preflight_and_explicit_override(self):
        result = self.client.post('/quality', data={'images': [self.photo(True)], 'views': '["front"]'})
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.get_json()['per_image'][0]['quality']['passed'])
        self.fabric_mock.assert_not_called()
        response = self.client.post('/analyze', data={'images': [self.photo(True)], 'views': '["front"]'})
        self.assertEqual(response.status_code, 422)
        self.fabric_mock.assert_not_called()
        override = self.client.post('/analyze', data={'images': [self.photo(True)], 'views': '["front"]', 'allow_low_quality': 'true'})
        self.assertEqual(override.status_code, 200)

    def test_bad_inputs(self):
        self.assertEqual(self.client.post('/analyze').status_code, 400)
        self.assertEqual(self.client.post('/analyze', data={'image': (io.BytesIO(b'not an image'), 'bad.jpg')}).status_code, 400)
        self.assertEqual(self.client.post('/analyze', data={'images': [self.photo()], 'views': '["back"]'}).status_code, 400)
        self.assertEqual(self.client.post('/analyze', data={'images': [self.photo()], 'views': 'oops'}).status_code, 400)
        self.assertEqual(self.client.post('/analyze', data={'images': [self.photo(), self.photo()], 'views': '["front", "front"]'}).status_code, 400)

    def test_uploaded_fabric_image_uses_supplied_tree_and_returns_supported_json(self):
        previous = {
            name: getattr(self.api, name)
            for name in ('fabric_backend', 'fabric_model', 'fabric_labels', 'fabric_model_name')
        }
        self.addCleanup(lambda: [setattr(self.api, name, value) for name, value in previous.items()])
        self.api.fabric_backend = 'decision_tree'
        self.api.fabric_model = FabricDecisionTree(
            __import__('pathlib').Path(__file__).parent / 'model' / 'best_decision_tree.joblib')
        self.api.fabric_labels = self.api.fabric_model.labels
        self.api.fabric_model_name = self.api.fabric_model.name

        self.fabric.stop()
        response = self.client.post('/analyze', data={
            'image': self.photo(),
        })
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['fabric_model'], 'GLCM (12 features) + Decision Tree')
        self.assertEqual(body['fabric_validation_accuracy'], 47.53)
        self.assertIn(body['fabric_type'], ['cotton', 'denim', 'polyester', 'silk', 'wool'])
        self.assertEqual(body['per_image'][0]['score_kind'], 'uncalibrated_decision_tree_vote')

    def test_upload_limit_returns_json_413(self):
        with patch.dict(self.api.app.config, MAX_CONTENT_LENGTH=10):
            response = self.client.post('/analyze', data={'image': self.photo()})
        self.assertEqual(response.status_code, 413)
        self.assertIn('error', response.get_json())

    def test_damage_review_recalculates_assessment_from_confirmed_findings(self):
        data = {'fabric_type': 'polyester', 'findings': [
            {'type': 'hole', 'area_ratio': .2, 'confirmed': False},
            {'type': 'stain', 'area_ratio': .01, 'confirmed': True},
        ]}
        response = self.client.post('/review-damage', json=data)
        self.assertEqual(response.status_code, 200)
        condition = response.get_json()["condition_assessment"]
        self.assertEqual(condition["provenance"], "user_reported")
        self.assertEqual([item["type"] for item in condition["observations"]], ["stain"])
        data['findings'][1]['confirmed'] = False
        response = self.client.post('/review-damage', json=data)
        self.assertEqual(response.get_json()["condition_assessment"]["observations"], [])
        self.assertEqual(response.get_json()["health_score"], 100)
        data['findings'][1]['confirmed'] = 'yes'
        self.assertEqual(self.client.post('/review-damage', json=data).status_code, 400)

    def test_damage_review_refreshes_the_care_advice(self):
        features = {name: 0.5 for name in self.api.IMAGE_CARE_FEATURES}
        features.update(luminance_mean=200.0, saturation_mean=0.05, dark_fraction=0.0)
        data = {'fabric_type': 'cotton', 'fabric_confidence': 75, 'fabric_review_threshold': 60,
                'fabric_model': 'ConvNeXt-Tiny (V4)', 'care_features': features,
                'findings': [{'type': 'hole', 'area_ratio': .02, 'confirmed': True}]}
        result = self.client.post('/review-damage', json=data).get_json()
        self.assertFalse(result['fabric_needs_confirmation'])  # 75% passes the model's 60% threshold
        self.assertTrue(result['ai_care']['care_tips'][0]['text'].startswith('Mend the opening'))
        self.assertIn('ConvNeXt-Tiny (V4)', result['ai_care']['provided_by'])
        data['findings'][0]['confirmed'] = False
        result = self.client.post('/review-damage', json=data).get_json()
        self.assertFalse(any('opening' in tip['text'] for tip in result['ai_care']['care_tips']))
        # A pending finding counts again only when its detector score passes the damage threshold.
        data['findings'] = [{'type': 'hole', 'area_ratio': .02, 'confirmed': None, 'confidence': 95},
                            {'type': 'stain', 'area_ratio': .01, 'confirmed': False}]
        result = self.client.post('/review-damage', json=data).get_json()
        self.assertTrue(result['ai_care']['care_tips'][0]['text'].startswith('Mend the opening'))
        for bad in ({'care_features': {'luminance_mean': 'dark'}}, {'fabric_review_threshold': 140},
                    {'fabric_model': 'x' * 90}):
            with self.subTest(bad=bad):
                self.assertEqual(self.client.post('/review-damage', json={**data, **bad}).status_code, 400)
        without = {key: value for key, value in data.items() if key != 'care_features'}
        self.assertNotIn('ai_care', self.client.post('/review-damage', json=without).get_json())

    def test_partial_review_keeps_candidates_and_replaces_care(self):
        data = {"fabric_type": "cotton", "fabric_confidence": 62.5, "findings": [
            {"id": "0:0", "type": "hole", "view": "front", "bbox": [.7, .1, .1, .1],
             "area_ratio": .01, "confirmed": True},
            {"id": "1:0", "type": "stain", "view": "back", "area_ratio": .02, "confirmed": None},
        ]}
        result = self.client.post("/review-damage", json=data).get_json()
        self.assertEqual(result["condition_assessment"]["provenance"], "mixed")
        hole, candidate = result["condition_assessment"]["observations"]
        self.assertEqual(hole["location"], "Front photo · upper right")
        self.assertTrue(hole["confirmed"])
        self.assertFalse(candidate["confirmed"])
        self.assertTrue(any("mend" in row["action"] for row in result["risk_alerts"]))
        self.assertTrue(any("stain treatments" in action for action in result["care_recommendations"]))
        data["findings"][0]["confirmed"] = False
        result = self.client.post("/review-damage", json=data).get_json()
        self.assertEqual(result["condition_assessment"]["category"], "Possible stain · awaiting review")
        self.assertFalse(any("mend" in row["action"] for row in result["risk_alerts"]))
        data["findings"][1]["confirmed"] = False
        result = self.client.post("/review-damage", json=data).get_json()
        self.assertFalse(any("stain treatments" in action for action in result["care_recommendations"]))

    def test_readiness_requires_database_and_hides_connection_details(self):
        with patch.object(self.api, 'ping_database', return_value=True):
            response = self.client.get('/health/ready')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()['ready'])
        with patch.object(self.api, 'ping_database', side_effect=RuntimeError('private details')):
            response = self.client.get('/health/ready')
            health = self.client.get('/')
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()['ready'])
        self.assertNotIn('private', health.get_data(as_text=True))

    def test_api_discovery_does_not_block_on_database(self):
        with patch.object(self.api, 'ping_database', side_effect=AssertionError('must not probe')) as ping:
            response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['database_connected'], None)
        self.assertEqual(response.get_json()['fabric_labels'], self.api.fabric_labels)
        ping.assert_not_called()

    def test_browser_auth_preflight_allows_local_flutter_origin(self):
        response = self.client.options('/auth/register', headers={
            'Origin': 'http://localhost:50552',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'http://localhost:50552')
        self.assertIn('POST', response.headers['Access-Control-Allow-Methods'])

    def test_garment_inputs_are_validated_and_returned(self):
        inputs = {'condition': 'previously_used', 'use_frequency': 'daily',
                  'washing_frequency': 'weekly', 'purchase_date_known': False}
        response = self.client.post('/analyze', data={'image': self.photo(), 'user_inputs': json.dumps(inputs)})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['user_inputs']['use_frequency'], 'daily')
        self.assertIn('projected_condition_after_6_months', body)
        self.assertIn('daily', body['usage_status'])
        bad = self.client.post('/analyze', data={'image': self.photo(), 'user_inputs': '{bad'})
        self.assertEqual(bad.status_code, 400)
        self.fabric_mock.reset_mock()
        missing_date = self.client.post('/analyze', data={'image': self.photo(), 'user_inputs':
            json.dumps({'condition': 'previously_used', 'purchase_date_known': True})})
        self.assertEqual(missing_date.status_code, 400)
        self.fabric_mock.assert_not_called()

    def test_detection_geometry_and_severity_independent_of_confidence(self):
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [10, 10, 20, 20]
        box.cls.item.return_value = 0
        box.conf.item.return_value = .99
        result = MagicMock(orig_shape=(100, 100), boxes=[box])
        with patch.object(self.api, 'damage_model') as model:
            model.predict.return_value = [result]
            model.names = {0: 'hole'}
            # detect_damage itself is not stubbed for this test.
            self.damage.stop()
            output = self.api.detect_damage('ignored', 1, 'back')[0]
        self.assertIsNone(output['severity'])
        self.assertAlmostEqual(output['area_ratio'], .01)
        self.assertEqual(output['bbox'], [.1, .1, .1, .1])
        self.assertEqual(output['view'], 'back')

    def test_missing_detector_keeps_fabric_scan_available(self):
        with patch.object(self.api, "damage_model", None):
            response = self.client.post("/predict/hybrid", data={"image": self.photo()})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["fabric_type"], "future_fabric")
        self.assertEqual(result["condition_assessment"]["category"], "Damage assessment unavailable")
        self.assertEqual(result["damages"], [])
        self.assertIsNone(result["health_score"])  # no detector, no visible-condition score
        self.assertIsInstance(result["estimated_remaining_months"], int)
        self.assertEqual(result["ai_care"]["status"], "rule_based_guidance")  # unknown fabric uses the rules

    def test_label_restrictions_survive_upload_and_review(self):
        label = {"wash": "professional_only", "dry": "do_not_tumble"}
        photo = self.client.post("/predict/hybrid", data={
            "image": self.photo(), "user_inputs": json.dumps({"care_label": label}),
        }).get_json()
        review = self.client.post("/review-damage", json={
            "fabric_type": photo["fabric_type"], "user_inputs": photo["user_inputs"],
            "findings": [{"type": "fading", "area_ratio": .1, "confirmed": True}],
        }).get_json()
        for result in (photo, review):
            self.assertEqual(result["laundry_settings"]["laundry_mode"], "Professional care only")
            self.assertIsNone(result["laundry_settings"]["max_temperature_c"])
            self.assertEqual(result["condition_assessment"]["observations"], [])
        self.assertEqual(photo["laundry_settings"], review["laundry_settings"])

    def test_tile_boxes_are_mapped_to_the_photo_and_oversized_boxes_are_ignored(self):
        box = MagicMock()
        box.cls.item.return_value = 0
        box.conf.item.return_value = .64
        box.xyxy[0].tolist.return_value = [20, 10, 40, 30]
        with patch.object(self.api.damage_model, 'names', {0: 'hole'}):
            detection = self.api._damage_detection(
                box, (50, 100, 100, 100), (200, 200), 0, 'front', 0, True)
            self.assertEqual(detection['bbox'], [.35, .55, .1, .1])
            self.assertEqual(detection['status'], 'Possible')
            box.xyxy[0].tolist.return_value = [0, 10, 95, 30]
            self.assertIsNone(self.api._damage_detection(
                box, (50, 100, 100, 100), (200, 200), 0, 'front', 0, True))

    def test_full_photo_hole_area_filter(self):
        box = MagicMock()
        box.cls.item.return_value = 0
        box.conf.item.return_value = .9
        with patch.object(self.api.damage_model, 'names', {0: 'hole'}):
            for view, coordinates, kept in (
                ('front', [0, 0, 75, 77], False),
                ('back', [0, 0, 98, 98], False),
                ('close_up', [20, 20, 54, 60], True),
            ):
                with self.subTest(view=view):
                    box.xyxy[0].tolist.return_value = coordinates
                    result = self.api._damage_detection(
                        box, (0, 0, 100, 100), (100, 100), 0, view, 0, False)
                    self.assertEqual(result is not None, kept)

if __name__ == '__main__':
    unittest.main()
