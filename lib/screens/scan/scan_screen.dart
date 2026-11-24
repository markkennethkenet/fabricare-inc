import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../data/mock_data.dart';
import '../../models/scan_result.dart';
import '../../services/fabricare_api_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/scan_photo.dart';
import '../../widgets/section_card.dart';
import 'fabric_model_picker.dart';
import 'ocr_label_screen.dart';
import 'scan_result_screen.dart';
import 'smart_capture_screen.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});
  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

/// Where a running scan is, for the progress overlay.
enum _ScanStage { quality, analysis, saving }

class _ScanScreenState extends State<ScanScreen> {
  static const _views = ['front', 'back', 'close_up'];
  final _images = <String, ScanImage>{};
  final _quality = <String, Map<String, dynamic>>{};
  final _picker = ImagePicker();
  bool _busy = false;
  String? _checkingView;
  _ScanStage? _stage;
  int _requestedViews = 1;
  String _modelType = 'cnn';
  String _status = '';
  String _condition = 'unspecified';
  String _useFrequency = 'unspecified';
  String _washingFrequency = 'unspecified';
  String _launderingPractice = 'unspecified';
  bool _purchaseDateKnown = false;
  DateTime? _purchaseDate;
  Map<String, dynamic> _careLabel = {};
  final _garmentDetails = <String, String>{};

  Future<void> _editCareLabel() async {
    final label = await Navigator.of(context).push<Map<String, dynamic>>(
      MaterialPageRoute(
        builder: (_) => CareLabelScreen(instructions: _careLabel),
      ),
    );
    if (label != null && mounted) setState(() => _careLabel = label);
  }

  Map<String, dynamic> get _userInputs => {
        ..._garmentDetails,
        'care_label': _careLabel,
        'condition': _condition,
        'use_frequency': _useFrequency,
        'washing_frequency': _washingFrequency,
        'laundering_practice': _launderingPractice,
        'purchase_date_known': _purchaseDateKnown,
        'purchase_date': _purchaseDateKnown && _purchaseDate != null
            ? _purchaseDate!.toIso8601String().split('T').first
            : null,
      };

  Future<void> _pickDate() async {
    final date = await showDatePicker(
        context: context,
        initialDate: _purchaseDate ?? DateTime.now(),
        firstDate: DateTime(1900),
        lastDate: DateTime.now());
    if (date != null && mounted) setState(() => _purchaseDate = date);
  }

  Widget _historyInputs() => SectionCard(
      title: 'Garment history',
      icon: Icons.history,
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        DropdownButtonFormField<String>(
            initialValue: _condition,
            decoration: const InputDecoration(labelText: 'Garment condition'),
            items: const [
              DropdownMenuItem(
                  value: 'unspecified', child: Text('Unspecified')),
              DropdownMenuItem(value: 'new', child: Text('New')),
              DropdownMenuItem(
                  value: 'previously_used', child: Text('Previously used'))
            ],
            onChanged: _busy ? null : (v) => setState(() => _condition = v!)),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
            initialValue: _useFrequency,
            decoration: const InputDecoration(labelText: 'Frequency of use'),
            items: const [
              DropdownMenuItem(
                  value: 'unspecified', child: Text('Unspecified')),
              DropdownMenuItem(value: 'rarely', child: Text('Rarely')),
              DropdownMenuItem(value: 'weekly', child: Text('Weekly')),
              DropdownMenuItem(
                  value: 'several_times_a_week',
                  child: Text('Several times a week')),
              DropdownMenuItem(value: 'daily', child: Text('Daily'))
            ],
            onChanged:
                _busy ? null : (v) => setState(() => _useFrequency = v!)),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
            initialValue: _washingFrequency,
            decoration: const InputDecoration(labelText: 'Washing frequency'),
            items: const [
              DropdownMenuItem(
                  value: 'unspecified', child: Text('Unspecified')),
              DropdownMenuItem(value: 'rarely', child: Text('Rarely')),
              DropdownMenuItem(value: 'monthly', child: Text('Monthly')),
              DropdownMenuItem(value: 'weekly', child: Text('Weekly')),
              DropdownMenuItem(
                  value: 'multiple_per_week',
                  child: Text('Multiple times per week'))
            ],
            onChanged:
                _busy ? null : (v) => setState(() => _washingFrequency = v!)),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          initialValue: _launderingPractice,
          isExpanded: true,
          decoration: const InputDecoration(labelText: 'Usual cleaning method'),
          items: const [
            DropdownMenuItem(value: 'unspecified', child: Text('Unspecified')),
            DropdownMenuItem(
                value: 'gentle handling', child: Text('Gentle handling')),
            DropdownMenuItem(
                value: 'rubbing or wringing',
                child: Text('Rubbing or wringing')),
            DropdownMenuItem(
                value: 'professional cleaning',
                child: Text('Professional cleaning')),
          ],
          onChanged: _busy
              ? null
              : (value) => setState(() => _launderingPractice = value!),
        ),
        CheckboxListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('I know the purchase date'),
            value: _purchaseDateKnown,
            onChanged: _busy
                ? null
                : (v) => setState(() => _purchaseDateKnown = v ?? false)),
        if (_purchaseDateKnown)
          OutlinedButton.icon(
              onPressed: _busy ? null : _pickDate,
              icon: const Icon(Icons.calendar_month),
              label: Text(_purchaseDate == null
                  ? 'Select purchase date'
                  : _purchaseDate!.toIso8601String().split('T').first)),
        const SizedBox(height: 8),
        for (final (key, label) in const [
          ('composition', 'Fiber content from label'),
          ('usage_category', 'Garment use'),
          ('previous_repairs', 'Previous repairs'),
          ('care_practices', 'Usual care practices'),
        ])
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: TextFormField(
              key: ValueKey(key),
              initialValue: _garmentDetails[key],
              enabled: !_busy,
              maxLength: 250,
              decoration: InputDecoration(
                  labelText: '$label (optional)', counterText: ''),
              onChanged: (value) => _garmentDetails[key] = value,
            ),
          ),
      ]));

  /// Open the smart camera for [view].
  Future<void> _capture(String view) async {
    if (_busy) return;
    final index = _views.indexOf(view);
    final photo = await Navigator.of(context).push<CapturedPhoto>(
      MaterialPageRoute(
        fullscreenDialog: true,
        builder: (_) => SmartCaptureScreen(
          view: view,
          step: index + 1,
          totalSteps: _requestedViews > index ? _requestedViews : index + 1,
        ),
      ),
    );
    if (photo != null && mounted) await _setPhoto(view, photo.path);
  }

  Future<void> _pickFromGallery(String view) async {
    if (_busy) return;
    final photo = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 85,
        maxWidth: 1920,
        maxHeight: 1920);
    if (photo != null && mounted) await _setPhoto(view, photo.path);
  }

  Future<void> _setPhoto(String view, String path) async {
    if (_busy) return;
    try {
      setState(() {
        _images[view] = ScanImage(view: view, path: path);
        _quality.remove(view);
        if (view == 'front') {
          _images.remove('back');
          _images.remove('close_up');
          _quality.clear();
          _requestedViews = 1;
        }
        _busy = true;
        _checkingView = view;
        _status = 'Checking focus, light, and framing...';
      });
      final checks = await FabricCareApiService.instance.checkQuality([
        for (final name in _views)
          if (_images.containsKey(name)) _images[name]!,
      ]);
      if (!mounted) return;
      setState(() {
        for (final check in checks) {
          _quality[check['view'] as String] =
              Map<String, dynamic>.from(check['quality'] as Map);
        }
      });
    } catch (_) {
      if (mounted) {
        _message(
            'Could not finish the capture check. Check the API connection and try again.');
      }
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _checkingView = null;
          _status = '';
        });
      }
    }
  }

  void _removePhoto(String view) => setState(() {
        _images.remove(view);
        _quality.remove(view);
        if (view == 'front') {
          _images.clear();
          _quality.clear();
          _requestedViews = 1;
        }
      });

  void _message(String message) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(message)));

  Future<void> _analyze() async {
    if (_busy || !_images.containsKey('front')) return;
    if (_purchaseDateKnown && _purchaseDate == null) {
      _message('Select the known purchase date or mark it unknown.');
      return;
    }
    final accountId = AppData.instance.activeUserId;
    if (accountId == null) {
      _message('Sign in before saving a scan.');
      return;
    }
    var step = 'checking photo quality';
    setState(() {
      _busy = true;
      _stage = _ScanStage.quality;
      _status = 'Checking image quality...';
    });
    try {
      final images = [
        for (final view in _views)
          if (_images.containsKey(view)) _images[view]!
      ];
      final api = FabricCareApiService.instance;
      final quality = await api.checkQuality(images);
      if (!mounted) return;
      setState(() {
        for (final entry in quality) {
          _quality[entry['view'] as String] =
              Map<String, dynamic>.from(entry['quality'] as Map);
        }
      });
      final failures = quality
          .where((q) => (q['quality'] as Map)['passed'] != true)
          .toList();
      var override = false;
      // A photo without fabric (a wall, a cupboard) cannot be scanned at all.
      final noFabric =
          failures.where((q) => (q['quality'] as Map)['no_fabric'] == true);
      if (noFabric.isNotEmpty) {
        setState(() => _stage = null);
        await showDialog<void>(
            context: context,
            builder: (ctx) => AlertDialog(
                  icon: const Icon(Icons.texture_rounded),
                  title: const Text('No fabric detected'),
                  content: Text(
                      '${noFabric.map((q) => _viewTitle(q['view'] as String)).join(', ')}: '
                      'point the camera at the garment so its fabric fills the frame, then retake the photo.'),
                  actions: [
                    TextButton(
                        onPressed: () => Navigator.pop(ctx),
                        child: const Text('Retake photos')),
                  ],
                ));
        return;
      }
      if (failures.isNotEmpty) {
        setState(() => _stage = null);
        override = await showDialog<bool>(
                context: context,
                builder: (ctx) => AlertDialog(
                      icon: const Icon(Icons.photo_camera_outlined),
                      title: const Text('Review photo quality'),
                      content: Text(failures
                          .map((q) =>
                              "${_viewTitle(q['view'] as String)}: ${(q['quality'] as Map)['reason']}")
                          .join('\n\n')),
                      actions: [
                        TextButton(
                            onPressed: () => Navigator.pop(ctx, false),
                            child: const Text('Retake photos')),
                        TextButton(
                            onPressed: () => Navigator.pop(ctx, true),
                            child: const Text('Continue anyway')),
                      ],
                    )) ??
            false;
        if (!override || !mounted) return;
      }
      setState(() {
        _stage = _ScanStage.analysis;
        _status = 'Identifying the fabric and checking for damage...';
      });
      step = 'analyzing the photos';
      final response = await api.analyzeImages(images,
          allowLowQuality: override,
          modelType: _modelType,
          userInputs: _userInputs);
      if (!mounted) return;
      final canAddView =
          images.length == _requestedViews && _requestedViews < 3;
      if (response.fabricRejected) {
        // Below the fabric threshold: ask for a better photo and save nothing.
        final nextView = canAddView ? _views[_requestedViews] : null;
        setState(() => _stage = null);
        final addPhoto = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            icon: const Icon(Icons.help_outline_rounded),
            title: const Text('Fabric not identified'),
            content: Text([
              response.rejectionMessage,
              'Retake the photo in brighter, even light, close enough to show the weave.',
              if (nextView == 'back')
                'Or add the back of the same garment.'
              else if (nextView == 'close_up')
                'Or add a sharp close-up of the main fabric, away from labels.',
              'This scan was not saved.',
            ].join('\n\n')),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(ctx, false),
                  child: const Text('Retake photos')),
              if (nextView != null)
                FilledButton(
                    onPressed: () => Navigator.pop(ctx, true),
                    child: const Text('Add next photo')),
            ],
          ),
        );
        if (mounted && addPhoto == true) setState(() => _requestedViews++);
        return;
      }
      if (response.json['fabric_needs_confirmation'] == true &&
          !override &&
          canAddView) {
        final nextView = _views[_requestedViews];
        final guess =
            '${response.fabricType[0].toUpperCase()}${response.fabricType.substring(1)}'
            '${response.fabricConfidence == null ? '' : ' · ${response.fabricConfidence!.round()}%'}';
        setState(() => _stage = null);
        final addPhoto = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            icon: Icon(nextView == 'back'
                ? Icons.flip_camera_android_outlined
                : Icons.zoom_in_rounded),
            title: const Text('One more photo?'),
            content: Text([
              'Best guess: $guess.',
              nextView == 'back'
                  ? 'Add the back of the same garment in even light to confirm.'
                  : 'Add a sharp close-up of the main fabric, away from labels.',
            ].join('\n\n')),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(ctx, false),
                  child: const Text('See result')),
              FilledButton(
                  onPressed: () => Navigator.pop(ctx, true),
                  child: const Text('Add next photo')),
            ],
          ),
        );
        if (!mounted || addPhoto == null) return;
        if (addPhoto) {
          setState(() => _requestedViews++);
          return;
        }
      }
      step = 'saving scan history';
      setState(() {
        _stage = _ScanStage.saving;
        _status = 'Saving the scan to your history...';
      });
      final result = await AppData.instance.addAnalysis(images, response,
          userInputs: _userInputs, expectedUserId: accountId);
      if (!mounted) return;
      setState(() => _stage = null);
      await Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => ScanResultScreen(result: result)));
    } catch (error, stack) {
      debugPrint('FabriCare scan failed while $step: ${error.runtimeType}'
          '${error is ScanApiException ? ': ${error.message}' : ''}');
      if (error is! ScanApiException) debugPrintStack(stackTrace: stack);
      if (mounted) {
        final message = error is ScanApiException
            ? error.message
            : step == 'saving scan history'
                ? 'Analysis finished, but the scan could not be saved. Check your sign-in and device storage.'
                : 'Could not complete the scan while $step. Try again.';
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(message),
          action: SnackBarAction(label: 'Retry', onPressed: _analyze),
        ));
      }
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _stage = null;
          _status = '';
        });
      }
    }
  }

  void _showPhotoTips() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        icon: const Icon(Icons.tips_and_updates_outlined),
        title: const Text('Photo tips'),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _Tip(Icons.wb_sunny_outlined,
                'Use even daylight. Avoid glare and harsh shadows.'),
            _Tip(Icons.crop_free_rounded,
                'Front and back: lay the garment flat and fit it inside the box, about 30–50 cm away.'),
            _Tip(Icons.zoom_in_rounded,
                'Close-up: fill the box with the main fabric, about 10–15 cm away.'),
            _Tip(Icons.pan_tool_alt_outlined,
                'Hold still. The smart camera captures automatically when everything is right.'),
            _Tip(Icons.texture_rounded,
                'A plain, contrasting background gives steadier results.'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Done'),
          ),
        ],
      ),
    );
  }

  static String _viewTitle(String view) => switch (view) {
        'back' => 'Back photo',
        'close_up' => 'Fabric close-up',
        _ => 'Front photo',
      };

  Widget _photoSlot(String view, String description) {
    final photo = _images[view];
    final quality = _quality[view];
    final index = _views.indexOf(view);
    final info = CaptureViewInfo.of(view);
    return _PhotoSlot(
      number: index + 1,
      title: _viewTitle(view),
      description: description,
      info: info,
      required: view == 'front',
      photo: photo,
      quality: quality,
      checking: _checkingView == view,
      busy: _busy,
      onCapture: () => _capture(view),
      onGallery: () => _pickFromGallery(view),
      onRemove: () => _removePhoto(view),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasFront = _images.containsKey('front');
    final frontPassed = _quality['front']?['passed'] == true;
    final theme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan a Garment'),
        actions: [
          IconButton(
            tooltip: 'Photo tips',
            onPressed: _showPhotoTips,
            icon: const Icon(Icons.help_outline_rounded),
          ),
        ],
      ),
      body: Stack(
        children: [
          SafeArea(
            bottom: false,
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 16),
              children: [
                _ScanHeader(
                  frontPassed: frontPassed,
                  onTips: _showPhotoTips,
                ),
                const SizedBox(height: 14),
                FabricModelCard(
                  selected: _modelType,
                  enabled: !_busy,
                  onChanged: (value) => setState(() => _modelType = value),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(4, 4, 4, 10),
                  child: Text('Garment photos',
                      style: theme.titleMedium
                          ?.copyWith(fontWeight: FontWeight.w800)),
                ),
                _photoSlot('front',
                    'Lay one garment flat in even light. The smart camera tells you when to move closer or back.'),
                if (_requestedViews >= 2)
                  _photoSlot('back',
                      'Turn the same garment over to check the other side.'),
                if (_requestedViews >= 3)
                  _photoSlot('close_up',
                      'Fill the box with the main fabric. Include visible wear if you can.'),
                const SizedBox(height: 4),
                Container(
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: AppColors.border),
                  ),
                  child: Theme(
                    data: Theme.of(context)
                        .copyWith(dividerColor: Colors.transparent),
                    child: ExpansionTile(
                      tilePadding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 2),
                      childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 4),
                      leading: const Icon(Icons.edit_note_rounded,
                          color: AppColors.coral),
                      title: const Text('Garment details (optional)',
                          style: TextStyle(
                              fontWeight: FontWeight.w700, fontSize: 15)),
                      subtitle: const Text(
                          'Care label, condition and use improve the care advice.',
                          style: TextStyle(
                              color: AppColors.inkSoft, fontSize: 12.5)),
                      children: [
                        SectionCard(
                          title: 'Care label (optional)',
                          icon: Icons.sell_outlined,
                          child: OutlinedButton.icon(
                            onPressed: _busy ? null : _editCareLabel,
                            icon: const Icon(Icons.edit_outlined),
                            label: Text(_careLabel.values
                                    .any((v) => v != null && v != 'unknown')
                                ? 'Edit entered instructions'
                                : 'Enter label instructions'),
                          ),
                        ),
                        _historyInputs(),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (_stage != null)
            Positioned.fill(child: _AnalyzingOverlay(stage: _stage!)),
        ],
      ),
      bottomNavigationBar: SafeArea(
        top: false,
        child: Material(
          color: AppColors.background,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 6, 20, 8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (_busy) ...[
                  const LinearProgressIndicator(),
                  const SizedBox(height: 7),
                  Semantics(liveRegion: true, child: Text(_status)),
                  const SizedBox(height: 7),
                ] else if (!hasFront) ...[
                  Text('Add a front photo to continue.',
                      style: theme.bodySmall),
                  const SizedBox(height: 5),
                ],
                ElevatedButton.icon(
                  onPressed: !_busy && hasFront ? _analyze : null,
                  icon: const Icon(Icons.search),
                  label: const Text('Analyze garment'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _Tip extends StatelessWidget {
  final IconData icon;
  final String text;
  const _Tip(this.icon, this.text);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 20, color: AppColors.coral),
            const SizedBox(width: 10),
            Expanded(child: Text(text)),
          ],
        ),
      );
}

/// The green header with the three scan steps.
class _ScanHeader extends StatelessWidget {
  final bool frontPassed;
  final VoidCallback onTips;

  const _ScanHeader({required this.frontPassed, required this.onTips});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(18, 16, 14, 16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF3D5845), Color(0xFF2B3E31)],
        ),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.center_focus_strong_rounded,
                    color: Color(0xFFC6D6AD)),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Smart scan',
                        style: TextStyle(
                            color: AppColors.surface,
                            fontSize: 19,
                            fontWeight: FontWeight.w800)),
                    SizedBox(height: 2),
                    Text(
                        'The camera guides you to a sharp, well-lit, well-framed photo.',
                        style: TextStyle(
                            color: Color(0xFFE7EDDC), fontSize: 12.5)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              const _StepChip(
                  number: 1, label: 'Model', state: _StepState.done),
              const _StepLine(),
              _StepChip(
                  number: 2,
                  label: 'Photos',
                  state: frontPassed ? _StepState.done : _StepState.active),
              const _StepLine(),
              _StepChip(
                  number: 3,
                  label: 'Analyze',
                  state: frontPassed ? _StepState.active : _StepState.todo),
            ],
          ),
          const SizedBox(height: 12),
          InkWell(
            onTap: onTips,
            borderRadius: BorderRadius.circular(10),
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 4),
              child: Text(
                'Start with the front of one garment. Extra views are requested only if the fabric needs another look.',
                style: TextStyle(
                    color: AppColors.surface, fontSize: 13, height: 1.3),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

enum _StepState { done, active, todo }

class _StepChip extends StatelessWidget {
  final int number;
  final String label;
  final _StepState state;
  const _StepChip(
      {required this.number, required this.label, required this.state});

  @override
  Widget build(BuildContext context) {
    final done = state == _StepState.done;
    final active = state == _StepState.active;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 24,
          height: 24,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: done
                ? const Color(0xFFC6D6AD)
                : active
                    ? AppColors.surface
                    : Colors.white.withValues(alpha: 0.15),
          ),
          child: done
              ? const Icon(Icons.check_rounded,
                  size: 16, color: Color(0xFF2B3E31))
              : Text('$number',
                  style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                      color:
                          active ? const Color(0xFF2B3E31) : Colors.white70)),
        ),
        const SizedBox(width: 6),
        Text(label,
            style: TextStyle(
                color: active || done ? AppColors.surface : Colors.white70,
                fontWeight: active ? FontWeight.w800 : FontWeight.w600,
                fontSize: 12.5)),
      ],
    );
  }
}

class _StepLine extends StatelessWidget {
  const _StepLine();

  @override
  Widget build(BuildContext context) => Expanded(
        child: Container(
          height: 2,
          margin: const EdgeInsets.symmetric(horizontal: 8),
          color: Colors.white.withValues(alpha: 0.2),
        ),
      );
}

/// One photo view: empty camera prompt, or the photo with its quality check.
class _PhotoSlot extends StatelessWidget {
  final int number;
  final String title;
  final String description;
  final CaptureViewInfo info;
  final bool required;
  final ScanImage? photo;
  final Map<String, dynamic>? quality;
  final bool checking;
  final bool busy;
  final VoidCallback onCapture;
  final VoidCallback onGallery;
  final VoidCallback onRemove;

  const _PhotoSlot({
    required this.number,
    required this.title,
    required this.description,
    required this.info,
    required this.required,
    required this.photo,
    required this.quality,
    required this.checking,
    required this.busy,
    required this.onCapture,
    required this.onGallery,
    required this.onRemove,
  });

  @override
  Widget build(BuildContext context) {
    final photo = this.photo;
    final quality = this.quality;
    final passed = quality?['passed'] == true;
    final statusColor =
        photo != null ? const Color(0xFF344B3B) : AppColors.inkSoft;
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: photo == null
              ? AppColors.border
              : quality == null
                  ? AppColors.border
                  : passed
                      ? AppColors.sage.withValues(alpha: 0.6)
                      : AppColors.danger.withValues(alpha: 0.5),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                radius: 13,
                backgroundColor: photo != null
                    ? const Color(0xFF344B3B)
                    : AppColors.coral.withValues(alpha: 0.14),
                child: photo != null
                    ? const Icon(Icons.check_rounded,
                        size: 16, color: AppColors.surface)
                    : Text('$number',
                        style: const TextStyle(
                            color: AppColors.coral,
                            fontWeight: FontWeight.w800,
                            fontSize: 12)),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(title,
                    style: const TextStyle(
                        fontWeight: FontWeight.w800, fontSize: 15.5)),
              ),
              Text(
                photo != null
                    ? 'Added'
                    : required
                        ? 'Required'
                        : 'Optional',
                style: Theme.of(context)
                    .textTheme
                    .labelMedium
                    ?.copyWith(color: statusColor, fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(description,
              style: const TextStyle(
                  color: AppColors.inkSoft, fontSize: 12.5, height: 1.3)),
          const SizedBox(height: 12),
          if (photo == null)
            _EmptyCapture(info: info, enabled: !busy, onTap: onCapture)
          else
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Stack(
                children: [
                  SizedBox(
                    height: 190,
                    width: double.infinity,
                    child: ColoredBox(
                      color: AppColors.surfaceMuted,
                      child: ScanPhoto(photo.path,
                          fit: BoxFit.contain,
                          fallback: const _PhotoUnavailable()),
                    ),
                  ),
                  if (checking)
                    const Positioned.fill(
                      child: ColoredBox(
                        color: Color(0x88000000),
                        child: Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              SizedBox(
                                  width: 28,
                                  height: 28,
                                  child: CircularProgressIndicator(
                                      color: Colors.white, strokeWidth: 3)),
                              SizedBox(height: 8),
                              Text('Checking focus and light…',
                                  style: TextStyle(color: Colors.white)),
                            ],
                          ),
                        ),
                      ),
                    )
                  else if (quality != null)
                    Positioned(
                      left: 10,
                      top: 10,
                      child: _QualityBadge(passed: passed),
                    ),
                ],
              ),
            ),
          if (quality != null && !checking) ...[
            const SizedBox(height: 10),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                    passed
                        ? Icons.check_circle_rounded
                        : Icons.error_outline_rounded,
                    size: 18,
                    color: passed ? AppColors.sage : AppColors.danger),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    passed
                        ? 'Focus and lighting check passed. You can analyze this photo.'
                        : quality['reason'] as String,
                    style: TextStyle(
                      fontSize: 13,
                      color:
                          passed ? const Color(0xFF344B3B) : AppColors.danger,
                    ),
                  ),
                ),
              ],
            ),
            if (quality['framing_hint'] case final String hint)
              Padding(
                padding: const EdgeInsets.only(top: 6, left: 26),
                child: Text(hint,
                    style: const TextStyle(
                        fontSize: 12.5, color: AppColors.inkSoft)),
              ),
          ],
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    backgroundColor: AppColors.ink,
                    minimumSize: const Size(0, 46),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14)),
                  ),
                  onPressed: busy ? null : onCapture,
                  icon: const Icon(Icons.camera_alt_rounded, size: 18),
                  label: Text(photo == null ? 'Capture' : 'Retake'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  style:
                      OutlinedButton.styleFrom(minimumSize: const Size(0, 46)),
                  onPressed: busy ? null : onGallery,
                  icon: const Icon(Icons.photo_library_outlined, size: 18),
                  label: Text(photo == null ? 'Gallery' : 'Replace'),
                ),
              ),
              if (photo != null)
                IconButton(
                  tooltip: 'Remove photo',
                  onPressed: busy ? null : onRemove,
                  icon: const Icon(Icons.delete_outline_rounded),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class _EmptyCapture extends StatelessWidget {
  final CaptureViewInfo info;
  final bool enabled;
  final VoidCallback onTap;
  const _EmptyCapture(
      {required this.info, required this.enabled, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Open the smart camera',
      excludeSemantics: true,
      child: InkWell(
        onTap: enabled ? onTap : null,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          height: 150,
          width: double.infinity,
          decoration: BoxDecoration(
            color: AppColors.background,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
                color: AppColors.coral.withValues(alpha: 0.35), width: 1.5),
          ),
          child: Stack(
            children: [
              // Corner marks echo the camera's guide box.
              for (final alignment in const [
                Alignment.topLeft,
                Alignment.topRight,
                Alignment.bottomLeft,
                Alignment.bottomRight
              ])
                Align(
                  alignment: alignment,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: _Corner(alignment: alignment),
                  ),
                ),
              Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 52,
                      height: 52,
                      decoration: BoxDecoration(
                        color: AppColors.coral.withValues(alpha: 0.12),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(info.icon, color: AppColors.coral, size: 28),
                    ),
                    const SizedBox(height: 8),
                    const Text('Tap to open the smart camera',
                        style: TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 2),
                    Text(info.distance,
                        style: const TextStyle(
                            color: AppColors.inkSoft, fontSize: 12.5)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Corner extends StatelessWidget {
  final Alignment alignment;
  const _Corner({required this.alignment});

  @override
  Widget build(BuildContext context) {
    const side = BorderSide(color: AppColors.coral, width: 3);
    return SizedBox(
      width: 18,
      height: 18,
      child: DecoratedBox(
        decoration: BoxDecoration(
          border: Border(
            top: alignment.y < 0 ? side : BorderSide.none,
            bottom: alignment.y > 0 ? side : BorderSide.none,
            left: alignment.x < 0 ? side : BorderSide.none,
            right: alignment.x > 0 ? side : BorderSide.none,
          ),
        ),
      ),
    );
  }
}

class _QualityBadge extends StatelessWidget {
  final bool passed;
  const _QualityBadge({required this.passed});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: passed ? const Color(0xFF344B3B) : AppColors.danger,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(passed ? Icons.check_rounded : Icons.priority_high_rounded,
                size: 15, color: Colors.white),
            const SizedBox(width: 4),
            Text(passed ? 'Clear photo' : 'Needs a retake',
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w700)),
          ],
        ),
      );
}

class _PhotoUnavailable extends StatelessWidget {
  const _PhotoUnavailable();

  @override
  Widget build(BuildContext context) => const Center(
        child: Text('Photo unavailable. Retake or replace it.'),
      );
}

/// Full-screen progress while a scan runs.
class _AnalyzingOverlay extends StatelessWidget {
  final _ScanStage stage;
  const _AnalyzingOverlay({required this.stage});

  @override
  Widget build(BuildContext context) {
    const steps = [
      (
        _ScanStage.quality,
        Icons.photo_camera_outlined,
        'Checking photo quality'
      ),
      (
        _ScanStage.analysis,
        Icons.texture_rounded,
        'Identifying the fabric and checking for damage'
      ),
      (
        _ScanStage.saving,
        Icons.bookmark_added_outlined,
        'Saving to your history'
      ),
    ];
    return ColoredBox(
      color: AppColors.background.withValues(alpha: 0.94),
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(
                width: 72,
                height: 72,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox.expand(
                        child: CircularProgressIndicator(
                            strokeWidth: 5, color: AppColors.coral)),
                    Icon(Icons.checkroom_rounded,
                        size: 32, color: AppColors.ink),
                  ],
                ),
              ),
              const SizedBox(height: 18),
              const Text('Analyzing your garment',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
              const SizedBox(height: 18),
              for (final (value, icon, label) in steps)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Row(
                    children: [
                      Icon(
                        value.index < stage.index
                            ? Icons.check_circle_rounded
                            : icon,
                        color: value.index < stage.index
                            ? AppColors.sage
                            : value == stage
                                ? AppColors.coral
                                : AppColors.inkSoft.withValues(alpha: 0.5),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          label,
                          style: TextStyle(
                            fontWeight: value == stage
                                ? FontWeight.w800
                                : FontWeight.w500,
                            color: value.index <= stage.index
                                ? AppColors.ink
                                : AppColors.inkSoft,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
