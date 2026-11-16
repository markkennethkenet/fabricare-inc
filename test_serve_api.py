import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import serve_api


class HostedStartupTests(unittest.TestCase):
    def test_initializes_schema_and_uses_host_assigned_port(self):
        database = SimpleNamespace(init_schema=MagicMock())
        app = SimpleNamespace(config={'MAX_CONTENT_LENGTH': 12345})
        waitress = SimpleNamespace(serve=MagicMock())
        with patch.dict(sys.modules, {
            'database': database, 'fabricare_api': SimpleNamespace(app=app),
            'waitress': waitress,
        }), patch.dict(os.environ, PORT='8080'):
            serve_api.main()
        database.init_schema.assert_called_once()
        waitress.serve.assert_called_once_with(
            app, host='0.0.0.0', port=8080, threads=1, max_request_body_size=12345)

    def test_database_failure_keeps_scanning_up_without_exposing_credentials(self):
        database = SimpleNamespace(init_schema=MagicMock(side_effect=RuntimeError('secret')))
        waitress = SimpleNamespace(serve=MagicMock())
        with patch.dict(sys.modules, {
            'database': database, 'fabricare_api': SimpleNamespace(app=MagicMock()),
            'waitress': waitress,
        }), self.assertLogs(level='ERROR') as captured:
            serve_api.main()
        waitress.serve.assert_called_once()  # scans keep working without accounts
        self.assertNotIn('secret', ''.join(captured.output))
        self.assertIn('Scanning stays available', ''.join(captured.output))


if __name__ == '__main__':
    unittest.main()
