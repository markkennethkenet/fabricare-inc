import 'dart:math' as math;
import 'dart:typed_data';

/// Live framing advice for the smart camera.
///
/// Works on the camera's brightness (luma) plane only, so it is cheap enough
/// to run several times a second on a phone. The checks are heuristics: they
/// steer the user toward the sharp, evenly lit, well-filled photos the fabric
/// model handles best. The server still runs its own quality check.

/// The photo views a scan collects, matching the API's view names.
const captureViewFront = 'front';
const captureViewBack = 'back';
const captureViewCloseUp = 'close_up';

/// What is wrong with the current frame, in priority order.
enum GuidanceIssue {
  none,
  tooDark,
  tooBright,
  moving,
  blurry,
  tooCloseToFocus,
  tooFar,
  tooClose,
}

/// The guide box drawn over the preview, in portrait preview space.
///
/// [widthFraction] is the box width as a share of the preview width and
/// [aspect] is box height ÷ box width in pixels.
class GuideBox {
  final double widthFraction;
  final double aspect;
  const GuideBox(this.widthFraction, this.aspect);

  /// Box height as a share of a preview with the given width ÷ height ratio.
  double heightFraction(double previewAspect) =>
      math.min(0.78, widthFraction * aspect * previewAspect);

  /// Box size as (width, height) shares of the preview.
  ({double width, double height}) fractions(double previewAspect) {
    final height = heightFraction(previewAspect);
    return (width: height / (aspect * previewAspect), height: height);
  }
}

/// Front and back photos frame the whole garment; a close-up frames fabric.
GuideBox guideBoxFor(String view) => view == captureViewCloseUp
    ? const GuideBox(0.74, 1.0)
    : const GuideBox(0.84, 1.22);

/// A camera frame's brightness values.
///
/// Android YUV frames pass plane 0 (Y) with [pixelStride] 1. iOS BGRA frames
/// pass the single plane with [pixelStride] 4 and [bgra] true.
class LumaFrame {
  final Uint8List bytes;
  final int width;
  final int height;
  final int rowStride;
  final int pixelStride;
  final bool bgra;

  const LumaFrame({
    required this.bytes,
    required this.width,
    required this.height,
    required this.rowStride,
    this.pixelStride = 1,
    this.bgra = false,
  });

  int at(int x, int y) {
    final i = y * rowStride + x * pixelStride;
    if (!bgra) return bytes[i];
    // BGRA: weight blue, green, red as in ITU-R BT.601.
    return (bytes[i] * 29 + bytes[i + 1] * 150 + bytes[i + 2] * 77) >> 8;
  }
}

class CaptureGuidance {
  final GuidanceIssue issue;
  final String message;

  /// Optional, non-blocking advice shown under the main message.
  final String? tip;

  /// Coarse brightness grid kept for motion detection on the next frame.
  final List<double> grid;
  final double brightness;
  final double sharpness;

  /// Share of the guide box that differs from the background (front/back).
  final double? fill;

  const CaptureGuidance({
    required this.issue,
    required this.message,
    required this.grid,
    required this.brightness,
    required this.sharpness,
    this.fill,
    this.tip,
  });

  bool get ready => issue == GuidanceIssue.none;

  /// Light problems need the user to change something before a good photo.
  bool get isProblem =>
      issue == GuidanceIssue.tooDark || issue == GuidanceIssue.tooBright;
}

/// Thresholds for preview-resolution frames. Kept a little stricter than the
/// server's photo check (brightness 45–220) so accepted photos pass it.
class GuidanceThresholds {
  static const minBrightness = 52.0;
  static const maxBrightness = 214.0;
  static const motion = 7.0;
  static const sharpness = 22.0;
  static const plainBackgroundSpread = 10.0;
  static const minFill = 0.38;
  static const maxSpill = 0.45;
  static const closeUpMaxSpread = 44.0;

  /// Extra brightness needed before a dark-locked shutter unlocks again.
  static const darkUnlockMargin = 6.0;
}

/// Whether the shutter stays locked because the scene is too dark.
///
/// Locks as soon as a frame is too dark, and unlocks only once the scene is a
/// little brighter than the limit, so the button does not flicker when the
/// light sits right at the threshold.
bool shutterLockedByDarkness(bool wasLocked, CaptureGuidance guidance) {
  if (guidance.issue == GuidanceIssue.tooDark) return true;
  if (!wasLocked) return false;
  return guidance.brightness <
      GuidanceThresholds.minBrightness + GuidanceThresholds.darkUnlockMargin;
}

const _gridWidth = 40;
const _gridHeight = 30;

/// Analyse one frame.
///
/// [rotated] is true when the frame is landscape relative to the portrait
/// preview (sensor orientation 90 or 270), so the box axes are swapped.
/// [previewAspect] is the portrait preview's width ÷ height.
CaptureGuidance analyzeFrame(
  LumaFrame frame, {
  required String view,
  required double previewAspect,
  bool rotated = false,
  List<double>? previousGrid,
}) {
  final box = guideBoxFor(view).fractions(previewAspect);
  final boxW = rotated ? box.height : box.width;
  final boxH = rotated ? box.width : box.height;
  final left = (1 - boxW) / 2, top = (1 - boxH) / 2;
  final right = left + boxW, bottom = top + boxH;

  final grid = _grid(frame);
  final inside = <double>[], border = <double>[], ring = <double>[];
  for (var j = 0; j < _gridHeight; j++) {
    final cy = (j + 0.5) / _gridHeight;
    for (var i = 0; i < _gridWidth; i++) {
      final cx = (i + 0.5) / _gridWidth;
      final value = grid[j * _gridWidth + i];
      final isBorder =
          i < 2 || j < 2 || i >= _gridWidth - 2 || j >= _gridHeight - 2;
      if (cx >= left && cx <= right && cy >= top && cy <= bottom) {
        inside.add(value);
      } else if (isBorder) {
        border.add(value);
      } else {
        ring.add(value);
      }
    }
  }
  final brightness = _mean(inside);

  // Sharpness: Laplacian variance in five full-resolution patches in the box.
  final pxLeft = left * frame.width, pxTop = top * frame.height;
  final pxW = boxW * frame.width, pxH = boxH * frame.height;
  final patch = math.max(8, math.min(48, math.min(pxW, pxH) ~/ 4));
  double boxPatch(double u, double v) => _laplacianVariance(
      frame, (pxLeft + u * pxW).round(), (pxTop + v * pxH).round(), patch);
  final centerTexture = boxPatch(0.5, 0.5);
  final sharpness = [
    centerTexture,
    boxPatch(0.3, 0.3),
    boxPatch(0.7, 0.3),
    boxPatch(0.3, 0.7),
    boxPatch(0.7, 0.7),
  ].reduce(math.max);

  CaptureGuidance result(GuidanceIssue issue, String message,
          {String? tip, double? fill}) =>
      CaptureGuidance(
          issue: issue,
          message: message,
          tip: tip,
          grid: grid,
          brightness: brightness,
          sharpness: sharpness,
          fill: fill);

  if (brightness < GuidanceThresholds.minBrightness) {
    return result(GuidanceIssue.tooDark, 'Too dark — move to brighter light');
  }
  if (brightness > GuidanceThresholds.maxBrightness) {
    return result(
        GuidanceIssue.tooBright, 'Too bright — avoid direct light and glare');
  }
  if (previousGrid != null && previousGrid.length == grid.length) {
    var change = 0.0;
    for (var k = 0; k < grid.length; k++) {
      change += (grid[k] - previousGrid[k]).abs();
    }
    if (change / grid.length > GuidanceThresholds.motion) {
      return result(GuidanceIssue.moving, 'Hold steady');
    }
  }

  if (view == captureViewCloseUp) {
    if (sharpness < GuidanceThresholds.sharpness) {
      return result(GuidanceIssue.tooCloseToFocus,
          'Too close to focus — move back a little',
          tip: 'Hold the phone about 10–15 cm from the fabric.');
    }
    if (_spread(inside) > GuidanceThresholds.closeUpMaxSpread) {
      return result(
          GuidanceIssue.tooFar, 'Move closer — fill the box with fabric only',
          tip: 'Keep labels, seams and the background out of the box.');
    }
    return result(GuidanceIssue.none, 'Sharp texture — hold still');
  }

  // If the frame edges match the middle in brightness and fine texture, the
  // garment runs past the edges of the photo: the user is too close.
  final cornerTexture = [
        for (final (u, v) in const [
          (0.06, 0.06),
          (0.94, 0.06),
          (0.06, 0.94),
          (0.94, 0.94)
        ])
          _laplacianVariance(frame, (u * frame.width).round(),
              (v * frame.height).round(), math.max(8, patch ~/ 2)),
      ].reduce((a, b) => a + b) /
      4;
  final sameAsEdges = (_mean(inside) - _mean(border)).abs() < 14 &&
      centerTexture > GuidanceThresholds.sharpness &&
      cornerTexture > 0.5 * centerTexture;
  if (sameAsEdges) {
    return result(GuidanceIssue.tooClose,
        'Move back a little — fit the whole garment in the box');
  }

  final backgroundLevel = _median(border);
  final backgroundSpread = _spread(border);
  if (backgroundSpread < GuidanceThresholds.plainBackgroundSpread) {
    final threshold = math.max(14.0, 2.5 * backgroundSpread);
    bool foreground(double v) => (v - backgroundLevel).abs() > threshold;
    final fill = inside.where(foreground).length / inside.length;
    final spill =
        ring.isEmpty ? 0.0 : ring.where(foreground).length / ring.length;
    if (fill < GuidanceThresholds.minFill) {
      return result(
          GuidanceIssue.tooFar, 'Move closer — fill the box with the garment',
          tip: 'Start about 30–50 cm away.', fill: fill);
    }
    if (spill > GuidanceThresholds.maxSpill) {
      return result(GuidanceIssue.tooClose,
          'Move back a little — fit the garment inside the box',
          fill: fill);
    }
    if (sharpness < GuidanceThresholds.sharpness) {
      return result(
          GuidanceIssue.blurry, 'Hold steady and tap the fabric to focus',
          fill: fill);
    }
    return result(GuidanceIssue.none, 'Looks good — hold still', fill: fill);
  }

  // A busy background: the garment may still be well framed.
  if (sharpness < GuidanceThresholds.sharpness) {
    return result(
        GuidanceIssue.blurry, 'Hold steady and tap the fabric to focus');
  }
  return result(GuidanceIssue.none, 'Looks good — hold still',
      tip: 'A plain, contrasting background gives steadier results.');
}

List<double> _grid(LumaFrame frame) {
  final grid = List<double>.filled(_gridWidth * _gridHeight, 0);
  final dx = math.max(1, frame.width ~/ (_gridWidth * 4));
  final dy = math.max(1, frame.height ~/ (_gridHeight * 4));
  for (var j = 0; j < _gridHeight; j++) {
    final y = ((j + 0.5) * frame.height / _gridHeight).floor();
    final y0 = math.max(0, y - dy), y1 = math.min(frame.height - 1, y + dy);
    for (var i = 0; i < _gridWidth; i++) {
      final x = ((i + 0.5) * frame.width / _gridWidth).floor();
      final x0 = math.max(0, x - dx), x1 = math.min(frame.width - 1, x + dx);
      grid[j * _gridWidth + i] = (frame.at(x0, y0) +
              frame.at(x1, y0) +
              frame.at(x0, y1) +
              frame.at(x1, y1)) /
          4;
    }
  }
  return grid;
}

/// Variance of the 4-neighbour Laplacian in a square patch centred on (cx, cy).
double _laplacianVariance(LumaFrame frame, int cx, int cy, int size) {
  final half = size ~/ 2;
  final x0 = (cx - half).clamp(1, math.max(1, frame.width - 2 - size)).toInt();
  final y0 = (cy - half).clamp(1, math.max(1, frame.height - 2 - size)).toInt();
  final x1 = math.min(frame.width - 2, x0 + size);
  final y1 = math.min(frame.height - 2, y0 + size);
  var sum = 0.0, squares = 0.0, count = 0;
  for (var y = y0; y <= y1; y++) {
    for (var x = x0; x <= x1; x++) {
      final value = 4 * frame.at(x, y) -
          frame.at(x - 1, y) -
          frame.at(x + 1, y) -
          frame.at(x, y - 1) -
          frame.at(x, y + 1);
      sum += value;
      squares += value * value;
      count++;
    }
  }
  if (count == 0) return 0;
  final mean = sum / count;
  return squares / count - mean * mean;
}

double _mean(List<double> values) =>
    values.isEmpty ? 0 : values.reduce((a, b) => a + b) / values.length;

double _median(List<double> values) {
  if (values.isEmpty) return 0;
  final sorted = [...values]..sort();
  return sorted[sorted.length ~/ 2];
}

/// Robust spread: median absolute deviation scaled to a standard deviation.
double _spread(List<double> values) {
  if (values.isEmpty) return 0;
  final middle = _median(values);
  return _median([for (final v in values) (v - middle).abs()]) * 1.4826;
}
