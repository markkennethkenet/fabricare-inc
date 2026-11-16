import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import '../../services/capture_guidance.dart';
import '../../theme/app_theme.dart';

/// A photo returned by [SmartCaptureScreen].
class CapturedPhoto {
  final String path;

  /// True when the photo came from the guided camera and was cropped to the box.
  final bool guided;
  const CapturedPhoto(this.path, {this.guided = false});
}

/// Text for each photo view in the scan.
class CaptureViewInfo {
  final String title;
  final String instruction;
  final String distance;
  final IconData icon;
  const CaptureViewInfo(this.title, this.instruction, this.distance, this.icon);

  static CaptureViewInfo of(String view) => switch (view) {
        captureViewBack => const CaptureViewInfo(
            'Back of garment',
            'Turn the garment over and fit it inside the box.',
            'About 30–50 cm away',
            Icons.flip_camera_android_outlined),
        captureViewCloseUp => const CaptureViewInfo(
            'Fabric close-up',
            'Fill the box with the main fabric, away from labels and seams.',
            'About 10–15 cm from the fabric',
            Icons.zoom_in_rounded),
        _ => const CaptureViewInfo(
            'Front of garment',
            'Lay the garment flat and fit it inside the box.',
            'About 30–50 cm away',
            Icons.checkroom_outlined),
      };
}

/// Full-screen guided camera with live framing advice and auto-capture.
class SmartCaptureScreen extends StatefulWidget {
  final String view;
  final int step;
  final int totalSteps;

  /// Finds the device cameras; tests replace it because they have no camera.
  final Future<List<CameraDescription>> Function() loadCameras;

  const SmartCaptureScreen({
    super.key,
    required this.view,
    this.step = 1,
    this.totalSteps = 1,
    this.loadCameras = availableCameras,
  });

  @override
  State<SmartCaptureScreen> createState() => _SmartCaptureScreenState();
}

class _SmartCaptureScreenState extends State<SmartCaptureScreen>
    with WidgetsBindingObserver {
  static const _analysisInterval = Duration(milliseconds: 280);
  static const _readyFramesForAutoCapture = 4;

  CameraController? _controller;
  bool _starting = true;
  String? _cameraError;
  bool _capturing = false;
  bool _autoCapture = true;
  bool _torch = false;
  String? _reviewPath;

  CaptureGuidance? _guidance;
  List<double>? _previousGrid;
  DateTime _lastAnalysis = DateTime.fromMillisecondsSinceEpoch(0);
  bool _analyzing = false;
  int _readyFrames = 0;
  Offset? _focusPoint;
  Timer? _focusTimer;

  bool get _liveChecks => !kIsWeb;
  CaptureViewInfo get _info => CaptureViewInfo.of(widget.view);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _startCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    SystemChrome.setPreferredOrientations(const []);
    _focusTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (state == AppLifecycleState.inactive) {
      _controller = null;
      controller?.dispose();
      if (mounted) setState(() => _starting = true);
    } else if (state == AppLifecycleState.resumed && controller == null) {
      _startCamera();
    }
  }

  Future<void> _startCamera() async {
    setState(() {
      _starting = true;
      _cameraError = null;
    });
    try {
      final cameras = await widget.loadCameras();
      if (cameras.isEmpty) {
        throw CameraException('NoCamera', 'No camera found.');
      }
      final camera = cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cameras.first);
      final controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: defaultTargetPlatform == TargetPlatform.iOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      if (!kIsWeb) {
        await controller.lockCaptureOrientation(DeviceOrientation.portraitUp);
        await controller.setFlashMode(FlashMode.off);
      }
      _controller = controller;
      _torch = false;
      await _startStream();
      if (mounted) setState(() => _starting = false);
    } on CameraException catch (error) {
      if (mounted) {
        setState(() {
          _starting = false;
          _cameraError = switch (error.code) {
            'CameraAccessDenied' ||
            'CameraAccessDeniedWithoutPrompt' ||
            'CameraAccessRestricted' =>
              'Camera access is off. Allow it in your phone settings, or choose a photo instead.',
            'NoCamera' ||
            'cameraNotFound' =>
              'No camera was found on this device. Choose a photo instead.',
            _ => 'The camera could not start. Choose a photo or try again.',
          };
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _starting = false;
          _cameraError =
              'The camera could not start. Choose a photo or try again.';
        });
      }
    }
  }

  Future<void> _startStream() async {
    final controller = _controller;
    if (!_liveChecks ||
        controller == null ||
        controller.value.isStreamingImages) {
      return;
    }
    _previousGrid = null;
    _readyFrames = 0;
    await controller.startImageStream(_onFrame);
  }

  Future<void> _stopStream() async {
    final controller = _controller;
    if (controller != null && controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
  }

  void _onFrame(CameraImage image) {
    final now = DateTime.now();
    if (_analyzing ||
        _capturing ||
        now.difference(_lastAnalysis) < _analysisInterval) {
      return;
    }
    _analyzing = true;
    _lastAnalysis = now;
    try {
      final plane = image.planes.first;
      final bgra = image.format.group == ImageFormatGroup.bgra8888;
      final frame = LumaFrame(
        bytes: plane.bytes,
        width: image.width,
        height: image.height,
        rowStride: plane.bytesPerRow,
        pixelStride: bgra ? 4 : (plane.bytesPerPixel ?? 1),
        bgra: bgra,
      );
      final controller = _controller;
      if (controller == null) return;
      final guidance = analyzeFrame(
        frame,
        view: widget.view,
        previewAspect: 1 / controller.value.aspectRatio,
        rotated: image.width > image.height,
        previousGrid: _previousGrid,
      );
      _previousGrid = guidance.grid;
      final wasReady = _guidance?.ready ?? false;
      _readyFrames = guidance.ready ? _readyFrames + 1 : 0;
      if (guidance.ready && !wasReady) HapticFeedback.selectionClick();
      if (mounted) setState(() => _guidance = guidance);
      if (_autoCapture && _readyFrames >= _readyFramesForAutoCapture) {
        _capture();
      }
    } catch (_) {
      // A malformed frame skips one analysis; the next frame is checked.
    } finally {
      _analyzing = false;
    }
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (_capturing || controller == null || !controller.value.isInitialized) {
      return;
    }
    setState(() => _capturing = true);
    HapticFeedback.mediumImpact();
    try {
      await _stopStream();
      final shot = await controller.takePicture();
      final aspect = 1 / controller.value.aspectRatio;
      final path = kIsWeb ? shot.path : await _cropToGuide(shot.path, aspect);
      if (!mounted) return;
      setState(() {
        _reviewPath = path;
        _capturing = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _capturing = false);
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content:
              Text('Could not take the photo. Hold steady and try again.')));
      await _startStream();
    }
  }

  Future<String> _cropToGuide(String source, double previewAspect) async {
    final box = guideBoxFor(widget.view).fractions(previewAspect);
    final directory = await getTemporaryDirectory();
    final target =
        '${directory.path}${Platform.pathSeparator}fabricare_${widget.view}_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final cropped = await compute(cropPhotoToGuide,
        CropRequest(source, target, box.width, box.height, previewAspect));
    if (cropped != source) {
      try {
        await File(source).delete();
      } on FileSystemException {
        // The camera's own temporary file is cleaned up by the system later.
      }
    }
    return cropped;
  }

  Future<void> _retake() async {
    final path = _reviewPath;
    setState(() {
      _reviewPath = null;
      _guidance = null;
    });
    if (path != null && !kIsWeb) {
      try {
        await File(path).delete();
      } on FileSystemException {
        // Ignore: a leftover temporary photo is harmless.
      }
    }
    await _startStream();
  }

  void _usePhoto() {
    final path = _reviewPath;
    if (path != null) {
      Navigator.of(context).pop(CapturedPhoto(path, guided: true));
    }
  }

  Future<void> _pickFromGallery() async {
    await _stopStream();
    final photo = await ImagePicker().pickImage(
        source: ImageSource.gallery,
        imageQuality: 90,
        maxWidth: 1920,
        maxHeight: 1920);
    if (!mounted) return;
    if (photo != null) {
      Navigator.of(context).pop(CapturedPhoto(photo.path));
    } else {
      await _startStream();
    }
  }

  Future<void> _useSystemCamera() async {
    final photo = await ImagePicker().pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
        maxWidth: 1920,
        maxHeight: 1920);
    if (photo != null && mounted) {
      Navigator.of(context).pop(CapturedPhoto(photo.path));
    }
  }

  Future<void> _toggleTorch() async {
    final controller = _controller;
    if (controller == null || kIsWeb) return;
    try {
      await controller.setFlashMode(_torch ? FlashMode.off : FlashMode.torch);
      setState(() => _torch = !_torch);
    } on CameraException {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('This camera has no light.')));
      }
    }
  }

  Future<void> _focusAt(Offset position, Size size) async {
    final controller = _controller;
    if (controller == null || kIsWeb) return;
    final point = Offset((position.dx / size.width).clamp(0.0, 1.0),
        (position.dy / size.height).clamp(0.0, 1.0));
    setState(() => _focusPoint = position);
    _focusTimer?.cancel();
    _focusTimer = Timer(const Duration(milliseconds: 900), () {
      if (mounted) setState(() => _focusPoint = null);
    });
    try {
      await controller.setFocusPoint(point);
      await controller.setExposurePoint(point);
    } on CameraException {
      // Some cameras do not support focus points; the tap is ignored.
    }
  }

  Color get _boxColor {
    final guidance = _guidance;
    if (guidance == null) return Colors.white;
    if (guidance.ready) return const Color(0xFF7FD08F);
    if (guidance.isProblem) return const Color(0xFFFF8A6B);
    return const Color(0xFFF5C451);
  }

  IconData get _guidanceIcon => switch (_guidance?.issue) {
        GuidanceIssue.none => Icons.check_circle_rounded,
        GuidanceIssue.tooDark => Icons.wb_sunny_outlined,
        GuidanceIssue.tooBright => Icons.wb_twilight_rounded,
        GuidanceIssue.moving => Icons.pan_tool_alt_outlined,
        GuidanceIssue.blurry => Icons.center_focus_weak_rounded,
        GuidanceIssue.tooCloseToFocus => Icons.zoom_out_rounded,
        GuidanceIssue.tooFar => Icons.zoom_in_rounded,
        GuidanceIssue.tooClose => Icons.zoom_out_rounded,
        null => Icons.crop_free_rounded,
      };

  String get _guidanceText {
    if (!_liveChecks) return _info.instruction;
    return _guidance?.message ??
        (widget.view == captureViewCloseUp
            ? 'Point the camera at the fabric'
            : 'Point the camera at the garment');
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: _reviewPath != null
            ? _ReviewView(
                path: _reviewPath!,
                info: _info,
                onRetake: _retake,
                onUse: _usePhoto,
              )
            : _cameraError != null
                ? _CameraUnavailable(
                    message: _cameraError!,
                    onGallery: _pickFromGallery,
                    onSystemCamera: _useSystemCamera,
                    onRetry: _startCamera,
                  )
                : _cameraBody(context),
      ),
    );
  }

  Widget _cameraBody(BuildContext context) {
    final controller = _controller;
    final ready =
        !_starting && controller != null && controller.value.isInitialized;
    final previewAspect = ready ? 1 / controller.value.aspectRatio : 3 / 4;
    final progress =
        (_readyFrames / _readyFramesForAutoCapture).clamp(0.0, 1.0);
    return Stack(
      fit: StackFit.expand,
      children: [
        Center(
          child: AspectRatio(
            aspectRatio: previewAspect,
            child: LayoutBuilder(builder: (context, constraints) {
              final size = constraints.biggest;
              return GestureDetector(
                onTapDown: (details) => _focusAt(details.localPosition, size),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (ready)
                      CameraPreview(controller)
                    else
                      const ColoredBox(color: Color(0xFF111111)),
                    CustomPaint(
                      painter: GuideOverlayPainter(
                        box: guideBoxFor(widget.view).fractions(previewAspect),
                        color: _boxColor,
                      ),
                    ),
                    if (_focusPoint != null)
                      Positioned(
                        left: _focusPoint!.dx - 32,
                        top: _focusPoint!.dy - 32,
                        child: IgnorePointer(
                          child: Container(
                            width: 64,
                            height: 64,
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.white, width: 2),
                              borderRadius: BorderRadius.circular(10),
                            ),
                          ),
                        ),
                      ),
                    if (!ready)
                      const Center(
                          child:
                              CircularProgressIndicator(color: Colors.white)),
                  ],
                ),
              );
            }),
          ),
        ),
        // Top bar: close, step, light.
        Positioned(
          left: 0,
          right: 0,
          top: 0,
          child: _TopGradient(
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(8, 4, 8, 0),
                child: Column(
                  children: [
                    Row(
                      children: [
                        IconButton(
                          tooltip: 'Close camera',
                          onPressed: () => Navigator.of(context).maybePop(),
                          icon: const Icon(Icons.close_rounded,
                              color: Colors.white),
                        ),
                        Expanded(
                          child: Column(
                            children: [
                              Text(_info.title,
                                  style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 17,
                                      fontWeight: FontWeight.w800)),
                              const SizedBox(height: 2),
                              Text(
                                widget.totalSteps > 1
                                    ? 'Photo ${widget.step} of ${widget.totalSteps} · ${_info.distance}'
                                    : _info.distance,
                                style: const TextStyle(
                                    color: Colors.white70, fontSize: 12.5),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          tooltip: _torch ? 'Turn light off' : 'Turn light on',
                          onPressed: ready && !kIsWeb ? _toggleTorch : null,
                          icon: Icon(
                            _torch
                                ? Icons.flashlight_on_rounded
                                : Icons.flashlight_off_rounded,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    _GuidancePill(
                      icon: _guidanceIcon,
                      text: _guidanceText,
                      color: _boxColor,
                      tip: _liveChecks ? _guidance?.tip : _info.distance,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        // Bottom controls: gallery, shutter, auto.
        Positioned(
          left: 0,
          right: 0,
          bottom: 0,
          child: _BottomGradient(
            child: SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _info.instruction,
                      textAlign: TextAlign.center,
                      style:
                          const TextStyle(color: Colors.white, fontSize: 13.5),
                    ),
                    const SizedBox(height: 14),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _RoundAction(
                          icon: Icons.photo_library_outlined,
                          label: 'Gallery',
                          onTap: _capturing ? null : _pickFromGallery,
                        ),
                        _ShutterButton(
                          progress: _autoCapture && _liveChecks ? progress : 0,
                          busy: _capturing,
                          ready: _guidance?.ready ?? false,
                          onTap: ready && !_capturing ? _capture : null,
                        ),
                        _RoundAction(
                          icon: _autoCapture
                              ? Icons.motion_photos_auto_rounded
                              : Icons.motion_photos_off_outlined,
                          label: _autoCapture ? 'Auto on' : 'Auto off',
                          selected: _autoCapture,
                          onTap: _liveChecks
                              ? () => setState(() {
                                    _autoCapture = !_autoCapture;
                                    _readyFrames = 0;
                                  })
                              : null,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// Dims the preview outside the guide box and draws the box with corner marks.
class GuideOverlayPainter extends CustomPainter {
  final ({double width, double height}) box;
  final Color color;

  GuideOverlayPainter({required this.box, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromCenter(
      center: size.center(Offset.zero),
      width: size.width * box.width,
      height: size.height * box.height,
    );
    final rounded = RRect.fromRectAndRadius(rect, const Radius.circular(22));
    final shade = Path()
      ..addRect(Offset.zero & size)
      ..addRRect(rounded)
      ..fillType = PathFillType.evenOdd;
    canvas.drawPath(
        shade, Paint()..color = Colors.black.withValues(alpha: 0.5));
    canvas.drawRRect(
        rounded,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = color.withValues(alpha: 0.7));

    final corner = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.round
      ..color = color;
    const arm = 30.0, radius = 22.0;
    void bracket(Offset at, double dx, double dy) {
      final path = Path()
        ..moveTo(at.dx, at.dy + dy * (radius + arm))
        ..lineTo(at.dx, at.dy + dy * radius)
        ..arcToPoint(Offset(at.dx + dx * radius, at.dy),
            radius: const Radius.circular(radius), clockwise: dx * dy > 0)
        ..lineTo(at.dx + dx * (radius + arm), at.dy);
      canvas.drawPath(path, corner);
    }

    bracket(rect.topLeft, 1, 1);
    bracket(rect.topRight, -1, 1);
    bracket(rect.bottomLeft, 1, -1);
    bracket(rect.bottomRight, -1, -1);
  }

  @override
  bool shouldRepaint(GuideOverlayPainter old) =>
      old.box != box || old.color != color;
}

class _GuidancePill extends StatelessWidget {
  final IconData icon;
  final String text;
  final String? tip;
  final Color color;

  const _GuidancePill(
      {required this.icon, required this.text, required this.color, this.tip});

  @override
  Widget build(BuildContext context) {
    return Semantics(
      liveRegion: true,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.62),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: color.withValues(alpha: 0.9), width: 1.5),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: color, size: 22),
                const SizedBox(width: 8),
                Flexible(
                  child: Text(
                    text,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
            if (tip != null) ...[
              const SizedBox(height: 3),
              Text(tip!,
                  textAlign: TextAlign.center,
                  style:
                      const TextStyle(color: Colors.white70, fontSize: 12.5)),
            ],
          ],
        ),
      ),
    );
  }
}

class _ShutterButton extends StatelessWidget {
  final double progress;
  final bool busy;
  final bool ready;
  final VoidCallback? onTap;

  const _ShutterButton({
    required this.progress,
    required this.busy,
    required this.ready,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Take photo',
      child: GestureDetector(
        onTap: onTap,
        child: SizedBox(
          width: 84,
          height: 84,
          child: Stack(
            alignment: Alignment.center,
            children: [
              SizedBox.expand(
                child: CircularProgressIndicator(
                  value: busy ? null : progress,
                  strokeWidth: 4,
                  color: const Color(0xFF7FD08F),
                  backgroundColor: Colors.white24,
                ),
              ),
              AnimatedContainer(
                duration: const Duration(milliseconds: 150),
                width: 66,
                height: 66,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: onTap == null ? Colors.white54 : Colors.white,
                  border: Border.all(
                      color: ready ? const Color(0xFF7FD08F) : Colors.white,
                      width: 3),
                ),
                child: Icon(Icons.camera_alt_rounded,
                    color: ready ? const Color(0xFF2E6B3A) : AppColors.ink,
                    size: 30),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RoundAction extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback? onTap;

  const _RoundAction(
      {required this.icon,
      required this.label,
      this.onTap,
      this.selected = false});

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return Semantics(
      button: true,
      label: label,
      excludeSemantics: true,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(30),
        child: SizedBox(
          width: 76,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 50,
                height: 50,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: selected
                      ? Colors.white
                      : Colors.white.withValues(alpha: 0.16),
                ),
                child: Icon(icon,
                    color: selected
                        ? AppColors.ink
                        : Colors.white.withValues(alpha: enabled ? 1 : 0.4)),
              ),
              const SizedBox(height: 5),
              Text(label,
                  style: TextStyle(
                      color: Colors.white.withValues(alpha: enabled ? 1 : 0.4),
                      fontSize: 12)),
            ],
          ),
        ),
      ),
    );
  }
}

class _TopGradient extends StatelessWidget {
  final Widget child;
  const _TopGradient({required this.child});

  @override
  Widget build(BuildContext context) => DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xCC000000), Color(0x00000000)],
          ),
        ),
        child:
            Padding(padding: const EdgeInsets.only(bottom: 18), child: child),
      );
}

class _BottomGradient extends StatelessWidget {
  final Widget child;
  const _BottomGradient({required this.child});

  @override
  Widget build(BuildContext context) => DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.bottomCenter,
            end: Alignment.topCenter,
            colors: [Color(0xE6000000), Color(0x00000000)],
          ),
        ),
        child: child,
      );
}

/// Shows the cropped photo so the user can retake it or use it.
class _ReviewView extends StatelessWidget {
  final String path;
  final CaptureViewInfo info;
  final VoidCallback onRetake;
  final VoidCallback onUse;

  const _ReviewView({
    required this.path,
    required this.info,
    required this.onRetake,
    required this.onUse,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 12, 20, 16),
        child: Column(
          children: [
            Text('Check your ${info.title.toLowerCase()} photo',
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w800)),
            const SizedBox(height: 4),
            const Text('Is the fabric sharp, evenly lit and inside the frame?',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white70, fontSize: 13.5)),
            const SizedBox(height: 14),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: kIsWeb
                    ? Image.network(path, fit: BoxFit.contain)
                    : Image.file(File(path), fit: BoxFit.contain),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      side: const BorderSide(color: Colors.white54),
                    ),
                    onPressed: onRetake,
                    icon: const Icon(Icons.refresh_rounded),
                    label: const Text('Retake'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF7FD08F),
                      foregroundColor: const Color(0xFF15301B),
                    ),
                    onPressed: onUse,
                    icon: const Icon(Icons.check_rounded),
                    label: const Text('Use photo'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _CameraUnavailable extends StatelessWidget {
  final String message;
  final VoidCallback onGallery;
  final VoidCallback onSystemCamera;
  final VoidCallback onRetry;

  const _CameraUnavailable({
    required this.message,
    required this.onGallery,
    required this.onSystemCamera,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: IconButton(
                tooltip: 'Close camera',
                onPressed: () => Navigator.of(context).maybePop(),
                icon: const Icon(Icons.close_rounded, color: Colors.white),
              ),
            ),
            const Spacer(),
            const Icon(Icons.no_photography_outlined,
                color: Colors.white70, size: 56),
            const SizedBox(height: 16),
            Text(message,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white, fontSize: 16)),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: AppColors.ink),
              onPressed: onGallery,
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('Choose from gallery'),
            ),
            if (!kIsWeb) ...[
              const SizedBox(height: 10),
              OutlinedButton.icon(
                style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white54)),
                onPressed: onSystemCamera,
                icon: const Icon(Icons.photo_camera_outlined),
                label: const Text('Use the phone camera app'),
              ),
            ],
            TextButton(
              onPressed: onRetry,
              child: const Text('Try again',
                  style: TextStyle(color: Colors.white)),
            ),
            const Spacer(),
          ],
        ),
      ),
    );
  }
}

/// Input for [cropPhotoToGuide], sent to a background isolate.
class CropRequest {
  final String source;
  final String target;
  final double boxWidth;
  final double boxHeight;
  final double previewAspect;
  const CropRequest(this.source, this.target, this.boxWidth, this.boxHeight,
      this.previewAspect);
}

/// Crop a captured photo to the guide box shown on the preview.
///
/// The preview may show a centered part of a wider or taller photo, so the box
/// is mapped through the visible region. A 6% margin keeps the garment's edges.
/// Returns [CropRequest.source] unchanged if the photo cannot be cropped safely.
String cropPhotoToGuide(CropRequest request) {
  try {
    final decoded = img.decodeImage(File(request.source).readAsBytesSync());
    if (decoded == null) return request.source;
    final photo = img.bakeOrientation(decoded);
    final rect = guideCropRect(photo.width, photo.height, request.boxWidth,
        request.boxHeight, request.previewAspect);
    if (rect == null) return request.source;
    var cropped = img.copyCrop(photo,
        x: rect.x, y: rect.y, width: rect.width, height: rect.height);
    final longest =
        cropped.width > cropped.height ? cropped.width : cropped.height;
    if (longest > 1920) {
      cropped = cropped.width >= cropped.height
          ? img.copyResize(cropped,
              width: 1920, interpolation: img.Interpolation.average)
          : img.copyResize(cropped,
              height: 1920, interpolation: img.Interpolation.average);
    }
    File(request.target).writeAsBytesSync(img.encodeJpg(cropped, quality: 92));
    return request.target;
  } catch (_) {
    return request.source;
  }
}

/// The pixel rectangle of the guide box in a photo, or null for a photo whose
/// orientation does not match the portrait preview.
({int x, int y, int width, int height})? guideCropRect(int photoWidth,
    int photoHeight, double boxWidth, double boxHeight, double previewAspect,
    {double margin = 0.06}) {
  if (photoWidth <= 0 || photoHeight <= 0 || photoWidth > photoHeight * 1.05) {
    return null;
  }
  final photoAspect = photoWidth / photoHeight;
  var visibleWidth = photoWidth.toDouble(),
      visibleHeight = photoHeight.toDouble();
  if (photoAspect > previewAspect) {
    visibleWidth = photoHeight * previewAspect;
  } else {
    visibleHeight = photoWidth / previewAspect;
  }
  final width = visibleWidth * (boxWidth + 2 * margin).clamp(0.0, 1.0);
  final height = visibleHeight * (boxHeight + 2 * margin).clamp(0.0, 1.0);
  final x = ((photoWidth - width) / 2).round();
  final y = ((photoHeight - height) / 2).round();
  return (
    x: x.clamp(0, photoWidth - 1),
    y: y.clamp(0, photoHeight - 1),
    width: width.round().clamp(1, photoWidth - x.clamp(0, photoWidth - 1)),
    height: height.round().clamp(1, photoHeight - y.clamp(0, photoHeight - 1)),
  );
}
