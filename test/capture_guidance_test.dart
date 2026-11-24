import 'dart:math';
import 'dart:typed_data';

import 'package:fabricare/services/capture_guidance.dart';
import 'package:flutter_test/flutter_test.dart';

const _width = 240, _height = 320, _aspect = _width / _height;

/// Builds a portrait luma frame. [fabric] marks the textured garment area as
/// (left, top, right, bottom) fractions; everything else is plain background.
LumaFrame frame({
  int background = 70,
  (double, double, double, double)? fabric,
  int fabricLevel = 160,
  int seed = 3,
  bool smooth = false,
}) {
  final random = Random(seed);
  final bytes = Uint8List(_width * _height);
  for (var y = 0; y < _height; y++) {
    for (var x = 0; x < _width; x++) {
      var value = background;
      if (fabric != null) {
        final (l, t, r, b) = fabric;
        final u = x / _width, v = y / _height;
        if (u >= l && u <= r && v >= t && v <= b) {
          value = smooth
              ? fabricLevel + (x ~/ 40)
              : fabricLevel + random.nextInt(61) - 30;
        }
      }
      bytes[y * _width + x] = value.clamp(0, 255);
    }
  }
  return LumaFrame(
      bytes: bytes, width: _width, height: _height, rowStride: _width);
}

CaptureGuidance analyze(LumaFrame frame, String view,
        {List<double>? previous}) =>
    analyzeFrame(frame,
        view: view, previewAspect: _aspect, previousGrid: previous);

void main() {
  test('dark and glaring frames ask for better light first', () {
    expect(
        analyze(frame(background: 20, fabric: (0, 0, 1, 1), fabricLevel: 25),
                'front')
            .issue,
        GuidanceIssue.tooDark);
    expect(
        analyze(frame(background: 245, fabric: (0, 0, 1, 1), fabricLevel: 238),
                'front')
            .issue,
        GuidanceIssue.tooBright);
  });

  test('the shutter locks in the dark and unlocks only in clearly better light',
      () {
    CaptureGuidance at(double brightness, GuidanceIssue issue) =>
        CaptureGuidance(
            issue: issue,
            message: '',
            grid: const [],
            brightness: brightness,
            sharpness: 50);
    const limit = GuidanceThresholds.minBrightness;
    expect(
        shutterLockedByDarkness(false, at(30, GuidanceIssue.tooDark)), isTrue);
    // Just above the limit: still locked once locked, open if never locked.
    expect(shutterLockedByDarkness(true, at(limit + 2, GuidanceIssue.none)),
        isTrue);
    expect(shutterLockedByDarkness(false, at(limit + 2, GuidanceIssue.none)),
        isFalse);
    expect(shutterLockedByDarkness(true, at(limit + 10, GuidanceIssue.moving)),
        isFalse);
    // A real dark frame from the analyser locks it.
    final dark = analyze(
        frame(background: 20, fabric: (0, 0, 1, 1), fabricLevel: 25), 'front');
    expect(shutterLockedByDarkness(false, dark), isTrue);
  });

  test('a small garment on a plain background asks the user to move closer',
      () {
    final result = analyze(frame(fabric: (0.4, 0.4, 0.6, 0.6)), 'front');
    expect(result.issue, GuidanceIssue.tooFar);
    expect(result.message, contains('Move closer'));
    expect(result.fill, lessThan(GuidanceThresholds.minFill));
  });

  test('a garment that fills the guide box is ready', () {
    final result = analyze(frame(fabric: (0.1, 0.13, 0.9, 0.87)), 'front');
    expect(result.issue, GuidanceIssue.none, reason: result.message);
    expect(result.ready, isTrue);
  });

  test('a garment spilling past the box asks the user to move back', () {
    final result = analyze(frame(fabric: (0.03, 0.04, 0.97, 0.96)), 'back');
    expect(result.issue, GuidanceIssue.tooClose, reason: result.message);
    expect(result.message, contains('Move back'));
  });

  test('fabric covering the whole frame in a front photo means too close', () {
    final result = analyze(frame(fabric: (0, 0, 1, 1)), 'front');
    expect(result.issue, GuidanceIssue.tooClose, reason: result.message);
  });

  test('a moving camera is told to hold steady', () {
    final first = analyze(frame(fabric: (0.1, 0.13, 0.9, 0.87)), 'front');
    final moved = analyze(
        frame(background: 130, fabric: (0.1, 0.13, 0.9, 0.87), fabricLevel: 60),
        'front',
        previous: first.grid);
    expect(moved.issue, GuidanceIssue.moving);
  });

  test('a sharp fabric close-up is ready', () {
    final result = analyze(frame(fabric: (0, 0, 1, 1)), 'close_up');
    expect(result.issue, GuidanceIssue.none, reason: result.message);
  });

  test('a smooth, detail-free close-up means the camera cannot focus', () {
    final result =
        analyze(frame(fabric: (0, 0, 1, 1), smooth: true), 'close_up');
    expect(result.issue, GuidanceIssue.tooCloseToFocus);
    expect(result.message, contains('move back'));
  });

  test(
      'a close-up that still shows background asks to fill the box with fabric',
      () {
    final result = analyze(
        frame(background: 30, fabric: (0.5, 0, 1, 1), fabricLevel: 200),
        'close_up');
    expect(result.issue, GuidanceIssue.tooFar, reason: result.message);
  });

  test('rotated sensor frames swap the box axes', () {
    // A landscape sensor frame for a portrait preview.
    const w = 320, h = 240;
    final random = Random(1);
    final bytes = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final u = x / w, v = y / h;
        final inBox = u > 0.13 && u < 0.87 && v > 0.1 && v < 0.9;
        bytes[y * w + x] = inBox ? 150 + random.nextInt(61) - 30 : 70;
      }
    }
    final result = analyzeFrame(
        LumaFrame(bytes: bytes, width: w, height: h, rowStride: w),
        view: 'front',
        previewAspect: h / w,
        rotated: true);
    expect(result.issue, GuidanceIssue.none, reason: result.message);
  });

  test('guide boxes keep the requested proportions inside the preview', () {
    final front = guideBoxFor('front').fractions(9 / 16);
    expect(front.width, closeTo(0.84, 1e-9));
    expect(front.height, closeTo(0.84 * 1.22 * 9 / 16, 1e-9));
    final tall = guideBoxFor('front').fractions(1.0);
    expect(tall.height, closeTo(0.78, 1e-9));
    final closeUp = guideBoxFor('close_up').fractions(9 / 16);
    // A square box: equal pixel width and height in a 9:16 preview.
    expect(closeUp.height, closeTo(closeUp.width * 9 / 16, 1e-9));
  });
}
