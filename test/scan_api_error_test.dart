import 'dart:io';

import 'package:fabricare/models/scan_result.dart';
import 'package:fabricare/services/fabricare_api_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test('scan explains HTML route and size errors and invalid JSON responses',
      () async {
    final directory = await Directory.systemTemp.createTemp('scan-api-errors');
    addTearDown(() => directory.delete(recursive: true));
    final photo = await File('${directory.path}/front.jpg').writeAsBytes([1]);
    final images = [ScanImage(view: 'front', path: photo.path)];
    for (final (status, body, expected) in [
      (404, '<html>Not found</html>', 'HTTP 404'),
      (413, '<html>Payload too large</html>', 'upload limit'),
      (422, '{"error":"Review image quality"}', 'Review image quality'),
      (200, '<html>Unexpected page</html>', 'unexpected response'),
      (200, '{}', 'incomplete scan result'),
    ]) {
      await http.runWithClient(
        () => expectLater(
          FabricCareApiService.instance.analyzeImages(images),
          throwsA(isA<ScanApiException>()
              .having((error) => error.message, 'message', contains(expected))),
        ),
        () => MockClient((request) async {
          expect(request.url.path, endsWith('/predict/hybrid'));
          expect(request.headers['content-type'], startsWith('multipart/form-data'));
          return http.Response(body, status);
        }),
      );
    }
  });
}
