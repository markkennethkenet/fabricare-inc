import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../data/mock_data.dart';
import '../../models/scan_result.dart';
import '../../services/fabricare_api_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/section_card.dart';
import 'ocr_label_screen.dart';
import 'scan_result_screen.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});
  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  static const _views = ['front', 'back', 'close_up'];
  static const _fabricModels = {
    'cnn': 'CNN — MobileNetV2',
    'svm': 'GLCM + SVM',
    'glcm_dt': 'GLCM + Decision Tree',
  };
  final _images = <String, ScanImage>{};
  final _quality = <String, Map<String, dynamic>>{};
  final _picker = ImagePicker();
  bool _busy = false;
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

  Future<void> _pick(String view, ImageSource source) async {
    try {
      final photo = await _picker.pickImage(
          source: source, imageQuality: 85, maxWidth: 1920, maxHeight: 1920);
      if (photo == null || !mounted) return;
      setState(() {
        _images[view] = ScanImage(view: view, path: photo.path);
        _quality.remove(view);
      });
    } catch (_) {
      if (mounted) {
        _message('Could not access the camera or gallery. Try again.');
      }
    }
  }

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
      if (failures.isNotEmpty) {
        override = await showDialog<bool>(
                context: context,
                builder: (ctx) => AlertDialog(
                      title: const Text('Review photo quality'),
                      content: Text(failures
                          .map((q) =>
                              "${q['view']}: ${(q['quality'] as Map)['reason']}")
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
      setState(() => _status = 'Analyzing fabric and visible damage...');
      step = 'analyzing the photos';
      final response = await api.analyzeImages(images,
          allowLowQuality: override,
          modelType: _modelType,
          userInputs: _userInputs);
      step = 'saving scan history';
      final result = await AppData.instance.addAnalysis(images, response,
          userInputs: _userInputs, expectedUserId: accountId);
      if (!mounted) return;
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
          _status = '';
        });
      }
    }
  }

  void _showPhotoTips() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Photo tips'),
        content: const Text(
          'Use even lighting and avoid glare. For front and back photos, '
          'start about 30–50 cm away and keep the garment in frame. '
          'For a close-up, start about 10–15 cm away and keep the fabric sharp. '
          'Adjust the distance to fit the garment.',
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

  Widget _photoCard(
      String view, String title, String description, IconData icon) {
    final photo = _images[view];
    final quality = _quality[view];
    final required = view == 'front';
    final theme = Theme.of(context).textTheme;
    return SectionCard(
      title: title,
      icon: icon,
      trailing: Text(
        photo != null
            ? 'Added'
            : required
                ? 'Required'
                : 'Optional',
        style: theme.labelMedium?.copyWith(
          color: photo != null ? const Color(0xFF344B3B) : AppColors.inkSoft,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (photo == null)
            Text(description,
                style: theme.bodySmall?.copyWith(color: AppColors.inkSoft))
          else
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(
                File(photo.path),
                height: 120,
                width: double.infinity,
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) =>
                    const Text('Photo unavailable. Retake or replace it.'),
              ),
            ),
          if (quality != null) ...[
            const SizedBox(height: 8),
            Text(
              quality['passed'] == true
                  ? 'Quality check passed'
                  : quality['reason'] as String,
              style: theme.bodySmall?.copyWith(
                color: quality['passed'] == true
                    ? const Color(0xFF344B3B)
                    : AppColors.danger,
              ),
            ),
          ],
          const SizedBox(height: 9),
          Wrap(
            spacing: 6,
            children: [
              OutlinedButton.icon(
                style: OutlinedButton.styleFrom(minimumSize: const Size(0, 44)),
                onPressed: _busy ? null : () => _pick(view, ImageSource.camera),
                icon: const Icon(Icons.camera_alt_outlined, size: 18),
                label: Text(photo == null ? 'Capture' : 'Retake'),
              ),
              TextButton.icon(
                onPressed:
                    _busy ? null : () => _pick(view, ImageSource.gallery),
                icon: const Icon(Icons.photo_library_outlined, size: 18),
                label: Text(photo == null ? 'Gallery' : 'Replace from gallery'),
              ),
              if (photo != null)
                TextButton(
                  onPressed: _busy
                      ? null
                      : () => setState(() {
                            _images.remove(view);
                            _quality.remove(view);
                          }),
                  child: const Text('Remove'),
                ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasFront = _images.containsKey('front');
    final theme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Scan a Garment')),
      body: SafeArea(
        bottom: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 10),
          children: [
            SectionCard(
              title: 'Choose Fabric Identification Model',
              icon: Icons.tune,
              child: DropdownButtonFormField<String>(
                initialValue: _modelType,
                isExpanded: true,
                decoration: const InputDecoration(labelText: 'Fabric model'),
                itemHeight: null,
                items: [
                  for (final entry in _fabricModels.entries)
                    DropdownMenuItem(
                      value: entry.key,
                      child: Text(entry.value),
                    ),
                ],
                onChanged: _busy
                    ? null
                    : (value) => setState(() {
                          if (value != null) _modelType = value;
                        }),
              ),
            ),
            Container(
              padding: const EdgeInsets.fromLTRB(16, 12, 12, 14),
              decoration: BoxDecoration(
                color: const Color(0xFF344B3B),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          'Garment photos',
                          style: theme.titleLarge
                              ?.copyWith(color: AppColors.surface),
                        ),
                      ),
                      IconButton(
                        tooltip: 'Photo tips',
                        onPressed: _showPhotoTips,
                        color: AppColors.surface,
                        icon: const Icon(Icons.help_outline_rounded),
                      ),
                    ],
                  ),
                  Text(
                    '${_images.length} of 3 photos added',
                    style: const TextStyle(color: Color(0xFFE7EDDC)),
                  ),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: _images.length / _views.length,
                    color: const Color(0xFFC6D6AD),
                    backgroundColor: const Color(0xFF5A705E),
                    borderRadius: BorderRadius.circular(4),
                    semanticsLabel: 'Garment photos added',
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'Add the front first. A back photo and close-up help check more of the garment.',
                    style: TextStyle(color: AppColors.surface),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            _photoCard('front', 'Front photo', 'Keep the full front in frame.',
                Icons.checkroom_outlined),
            _photoCard(
                'back',
                'Back photo',
                'Turn the garment over to check the other side.',
                Icons.flip_camera_android_outlined),
            _photoCard(
                'close_up',
                'Fabric close-up',
                'Capture the weave or an area of visible wear.',
                Icons.zoom_in_rounded),
            SectionCard(
              title: 'Care label (optional)',
              icon: Icons.sell_outlined,
              child: OutlinedButton.icon(
                onPressed: _busy ? null : _editCareLabel,
                icon: const Icon(Icons.edit_outlined),
                label: Text(
                    _careLabel.values.any((v) => v != null && v != 'unknown')
                        ? 'Edit entered instructions'
                        : 'Enter label instructions'),
              ),
            ),
            _historyInputs(),
          ],
        ),
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
