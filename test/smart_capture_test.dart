import 'package:camera/camera.dart';
import 'package:fabricare/screens/scan/smart_capture_screen.dart';
import 'package:fabricare/services/capture_guidance.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('guideCropRect', () {
    test('crops a photo with the preview shape to the box plus a margin', () {
      const aspect = 9 / 16;
      final box = guideBoxFor('front').fractions(aspect);
      final rect = guideCropRect(1080, 1920, box.width, box.height, aspect)!;
      expect(rect.width, closeTo(1080 * (box.width + 0.12), 1));
      expect(rect.height, closeTo(1920 * (box.height + 0.12), 1));
      expect(rect.x + rect.width / 2, closeTo(540, 1));
      expect(rect.y + rect.height / 2, closeTo(960, 1));
    });

    test('maps the box through the visible part of a wider photo', () {
      // A 3:4 photo behind a 9:16 preview: the preview shows the middle strip.
      const aspect = 9 / 16;
      final box = guideBoxFor('close_up').fractions(aspect);
      final rect = guideCropRect(3024, 4032, box.width, box.height, aspect)!;
      const visibleWidth = 4032 * aspect;
      expect(rect.width, closeTo(visibleWidth * (box.width + 0.12), 1));
      expect(rect.x + rect.width / 2, closeTo(1512, 1));
      // The close-up box is square on screen, so the crop is close to square.
      expect(rect.width / rect.height, closeTo(1, 0.15));
    });

    test('leaves landscape photos uncropped', () {
      expect(guideCropRect(1920, 1080, 0.8, 0.5, 9 / 16), isNull);
      expect(guideCropRect(0, 1080, 0.8, 0.5, 9 / 16), isNull);
    });

    test('keeps the crop inside the photo', () {
      final rect = guideCropRect(100, 120, 0.99, 0.99, 100 / 120)!;
      expect(rect.x, greaterThanOrEqualTo(0));
      expect(rect.y, greaterThanOrEqualTo(0));
      expect(rect.x + rect.width, lessThanOrEqualTo(100));
      expect(rect.y + rect.height, lessThanOrEqualTo(120));
    });
  });

  testWidgets('offers the gallery when camera access is denied',
      (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: SmartCaptureScreen(
        view: 'front',
        loadCameras: () async =>
            throw CameraException('CameraAccessDenied', 'Denied in test'),
      ),
    ));
    await tester.pumpAndSettle();
    expect(find.textContaining('Camera access is off'), findsOneWidget);
    expect(find.text('Choose from gallery'), findsOneWidget);
    expect(find.text('Use the phone camera app'), findsOneWidget);
    expect(find.text('Try again'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('explains a device without a camera', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: SmartCaptureScreen(view: 'close_up', loadCameras: () async => []),
    ));
    await tester.pumpAndSettle();
    expect(find.textContaining('No camera was found'), findsOneWidget);
  });

  test('every view has capture instructions and a distance', () {
    for (final view in ['front', 'back', 'close_up']) {
      final info = CaptureViewInfo.of(view);
      expect(info.title, isNotEmpty);
      expect(info.distance, contains('cm'));
    }
    expect(CaptureViewInfo.of('close_up').distance, contains('10–15'));
  });
}
