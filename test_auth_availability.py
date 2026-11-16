"""Account routes must report database outages without leaking credentials."""
import unittest
from unittest.mock import patch

from flask import Flask
from mysql.connector import Error

from auth_routes import auth_bp


class AuthAvailabilityTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(auth_bp)
        self.client = app.test_client()

    def test_database_outage_returns_safe_json_for_login_and_register(self):
        for route in ('login', 'register'):
            with self.subTest(route=route), patch(
                'auth_routes.get_connection',
                side_effect=Error('private database connection details', errno=2003),
            ):
                response = self.client.post('/auth/' + route, json={
                    'full_name': 'Test User', 'email': 'test@example.com',
                    'password': 'test-password',
                })
                self.assertEqual(response.status_code, 503)
                self.assertIn('Account service', response.get_json()['error'])
                self.assertNotIn('private', response.get_data(as_text=True))

    def test_missing_fields_still_return_validation_error(self):
        for route in ('login', 'register', 'google'):
            response = self.client.post('/auth/' + route, json={})
            self.assertEqual(response.status_code, 400)


CLIENT = 'test-client.apps.googleusercontent.com'
CLAIMS = {'iss': 'https://accounts.google.com', 'aud': CLIENT, 'email': 'Pat@Example.com',
          'email_verified': True, 'name': 'Pat Doe', 'picture': 'https://example.com/pat.png'}
USER = {'id': 7, 'full_name': 'Pat Doe', 'email': 'pat@example.com', 'phone': '', 'location': '',
        'created_at': 1}


@patch.dict('os.environ', {'GOOGLE_CLIENT_IDS': CLIENT})
class GoogleSignInTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(auth_bp)
        self.client = app.test_client()

    def sign_in(self, claims=CLAIMS, existing=None):
        with patch('auth_routes.verify_google_token', return_value=claims) as verify, \
                patch('auth_routes._fetch_user_by_email', return_value=existing) as lookup, \
                patch('auth_routes._fetch_user_by_id', return_value=USER), \
                patch('auth_routes.insert_user', return_value=7) as insert, \
                patch('auth_routes.get_connection'), \
                patch('auth_routes._create_token', return_value='session-token'):
            response = self.client.post('/auth/google', json={'id_token': 'google-id-token'})
        return response, verify, lookup, insert

    def test_new_google_account_is_created_with_the_google_name(self):
        response, verify, lookup, insert = self.sign_in()
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body['token'], 'session-token')
        self.assertTrue(body['created'])
        self.assertEqual(body['picture'], 'https://example.com/pat.png')
        verify.assert_called_once_with('google-id-token', [CLIENT])
        lookup.assert_called_once_with('pat@example.com')
        self.assertEqual(insert.call_args.args[1:3], ('Pat Doe', 'pat@example.com'))

    def test_existing_email_signs_in_to_the_same_account(self):
        response, _, _, insert = self.sign_in(existing=USER)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()['created'])
        self.assertEqual(response.get_json()['user']['id'], 7)
        insert.assert_not_called()

    def test_unverified_email_is_refused(self):
        response, _, _, insert = self.sign_in(claims={**CLAIMS, 'email_verified': False})
        self.assertEqual(response.status_code, 401)
        insert.assert_not_called()

    def test_invalid_token_is_refused(self):
        with patch('auth_routes.verify_google_token', side_effect=ValueError('Token expired')):
            response = self.client.post('/auth/google', json={'id_token': 'expired'})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn('expired', response.get_json()['error'])

    def test_built_in_client_ids_apply_when_unset(self):
        from auth_routes import DEFAULT_GOOGLE_CLIENT_IDS, google_client_ids

        with patch.dict('os.environ', clear=True):
            self.assertEqual(google_client_ids(), list(DEFAULT_GOOGLE_CLIENT_IDS))
        self.assertTrue(all(value.endswith('.apps.googleusercontent.com') for value in DEFAULT_GOOGLE_CLIENT_IDS))

    def test_unconfigured_server_says_so(self):
        with patch.dict('os.environ', {'GOOGLE_CLIENT_IDS': ''}):
            response = self.client.post('/auth/google', json={'id_token': 'token'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('not set up', response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
