import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../data/mock_data.dart';
import '../../models/damage_detection.dart';
import '../../models/scan_result.dart';
import '../../services/fabricare_api_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/care_insights.dart';
import '../../widgets/detection_image.dart';
import '../../widgets/fabric_result_hero.dart';
import '../../widgets/scan_compare_sheet.dart';
import '../../widgets/section_card.dart';
import 'scan_screen.dart';
import 'smart_capture_screen.dart';

/// Displays assessment supplied by the backend; no local scoring rules.
class ScanResultScreen extends StatefulWidget {
  final ScanResult result;
  const ScanResultScreen({super.key, required this.result});

  @override
  State<ScanResultScreen> createState() => _ScanResultScreenState();
}

class _ScanResultScreenState extends State<ScanResultScreen> {
  late ScanResult result = widget.result;
  late Map<String, bool> _decisions = result.damageReview;
  bool _saving = false;
  String? _addingView;

  static const _views = ['front', 'back', 'close_up'];
  static String _viewName(String view) => switch (view) {
        'back' => 'Back',
        'close_up' => 'Close-up',
        _ => 'Front',
      };

  /// Adds the back or a close-up, then re-analyzes all views together.
  Future<void> _addView(String view) async {
    if (_addingView != null || _saving) return;
    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      showDragHandle: true,
      backgroundColor: AppColors.surface,
      builder: (sheet) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.photo_camera_outlined),
              title: const Text('Take a photo'),
              subtitle: Text(view == 'back'
                  ? 'Lay the garment face down, flat'
                  : 'Fill the box with the main fabric'),
              onTap: () => Navigator.pop(sheet, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library_outlined),
              title: const Text('Choose from gallery'),
              onTap: () => Navigator.pop(sheet, ImageSource.gallery),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
    if (source == null || !mounted) return;
    final String? path;
    if (source == ImageSource.camera) {
      final photo = await Navigator.of(context).push<CapturedPhoto>(
        MaterialPageRoute(
          fullscreenDialog: true,
          builder: (_) => SmartCaptureScreen(
              view: view, step: _views.indexOf(view) + 1, totalSteps: 3),
        ),
      );
      path = photo?.path;
    } else {
      final photo = await ImagePicker().pickImage(
          source: ImageSource.gallery,
          imageQuality: 85,
          maxWidth: 1920,
          maxHeight: 1920);
      path = photo?.path;
    }
    if (path == null || !mounted) return;
    await _rescan(ScanImage(view: view, path: path));
  }

  Future<void> _rescan(ScanImage added) async {
    setState(() => _addingView = added.view);
    final images = [
      ...result.images.where((image) => image.view != added.view),
      added,
    ]..sort((a, b) => _views.indexOf(a.view).compareTo(_views.indexOf(b.view)));
    try {
      final api = FabricCareApiService.instance;
      final quality = await api.checkQuality(images);
      final failures = quality
          .where((check) => (check['quality'] as Map)['passed'] != true)
          .toList();
      var allowLowQuality = false;
      if (failures.isNotEmpty) {
        if (!mounted) return;
        allowLowQuality = await showDialog<bool>(
              context: context,
              builder: (dialog) => AlertDialog(
                icon: const Icon(Icons.photo_camera_outlined),
                title: const Text('Review photo quality'),
                content: Text(failures
                    .map((check) =>
                        '${_viewName(check['view'] as String)}: ${(check['quality'] as Map)['reason']}')
                    .join('\n\n')),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(dialog, false),
                      child: const Text('Retake')),
                  TextButton(
                      onPressed: () => Navigator.pop(dialog, true),
                      child: const Text('Continue anyway')),
                ],
              ),
            ) ??
            false;
        if (!allowLowQuality) return;
      }
      final response = await api.analyzeImages(images,
          allowLowQuality: allowLowQuality,
          modelType: result.analysis['model_type'] as String?,
          userInputs: result.userInputs);
      final saved =
          await AppData.instance.replaceAnalysis(result, images, response);
      if (!mounted) return;
      setState(() {
        result = saved;
        _decisions = saved.damageReview;
      });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content:
              Text('${_viewName(added.view)} photo added. Result updated.')));
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Could not update the scan. $error')));
      }
    } finally {
      if (mounted) setState(() => _addingView = null);
    }
  }

  Widget _wholeGarmentCard() {
    bool has(String view) => result.images.any((image) => image.view == view);
    const extra = ['back', 'close_up'];
    if (!has('front') || extra.every(has)) return const SizedBox.shrink();
    final adding = _addingView;
    final missing = extra.where((view) => !has(view)).toList();
    return SectionCard(
      title: 'Scan the whole garment',
      icon: Icons.checkroom_outlined,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            adding != null
                ? 'Adding the ${_viewName(adding).toLowerCase()} and updating the result…'
                : missing.length == 2
                    ? 'Add the back and a close-up for a steadier result.'
                    : missing.first == 'back'
                        ? 'Add the back for a steadier result.'
                        : 'Add a close-up of the main fabric for a steadier result.',
            style: const TextStyle(color: AppColors.inkSoft),
          ),
          if (adding != null)
            const Padding(
              padding: EdgeInsets.only(top: 10),
              child: LinearProgressIndicator(),
            ),
          const SizedBox(height: 12),
          Row(
            children: [
              for (final (index, view) in extra.indexed) ...[
                if (index > 0) const SizedBox(width: 10),
                Expanded(
                  child: has(view)
                      ? FilledButton.tonalIcon(
                          onPressed: null,
                          icon: const Icon(Icons.check_rounded, size: 18),
                          label: Text(_viewName(view)),
                        )
                      : OutlinedButton.icon(
                          onPressed: adding == null && !_saving
                              ? () => _addView(view)
                              : null,
                          icon: Icon(
                              view == 'back'
                                  ? Icons.flip_camera_android_outlined
                                  : Icons.zoom_in_rounded,
                              size: 20),
                          label: Text(_viewName(view)),
                        ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  void _viewFinding(DamageItem item) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('${_display(item.label)} · ${_display(item.view)}'),
        content: SingleChildScrollView(
          child: SizedBox(
            width: double.maxFinite,
            child: DetectionImage(
              scan: result,
              imageIndex: item.imageIndex,
              focusDamageId: item.id,
            ),
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close')),
        ],
      ),
    );
  }

  Future<void> _saveReview() async {
    if (_saving) return;
    setState(() => _saving = true);
    final decisions = Map<String, bool>.of(_decisions);
    try {
      final assessment =
          await FabricCareApiService.instance.reviewDamage(result, decisions);
      final reviewed = await AppData.instance
          .saveDamageReview(result, assessment, decisions);
      if (!mounted) return;
      setState(() => result = reviewed);
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Review saved. Care advice updated.')));
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not save the review. $error')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  static List<Map> _entries(Object? values) =>
      values is List ? values.whereType<Map>().toList() : const [];

  static String _display(Object? value) {
    final text = value?.toString().replaceAll('_', ' ') ?? 'unavailable';
    return text.isEmpty ? text : text[0].toUpperCase() + text.substring(1);
  }

  static List<CareTip> _legacyTips(List<Map> entries) => [
        for (final entry in entries)
          if (entry['action'] case final String action)
            (
              text: action,
              referenceIds: List<String>.from(
                  (entry['reference_ids'] as List? ?? const [])
                      .whereType<String>()),
            ),
      ];

  Widget _reviewFinding(DamageItem item, Map<String, bool> savedReview) {
    final decision = _decisions[item.id];
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.fromLTRB(12, 10, 8, 10),
      decoration: BoxDecoration(
        color: AppColors.background,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${_display(item.label)} · ${_display(item.view)}',
                    style: const TextStyle(fontWeight: FontWeight.w800)),
                Text(
                  switch (savedReview[item.id]) {
                    true => 'Confirmed by you',
                    false => 'Dismissed',
                    null => '${item.confidencePercent}% detector score',
                  },
                  style:
                      const TextStyle(fontSize: 12.5, color: AppColors.inkSoft),
                ),
                if (item.imageIndex >= 0 &&
                    item.imageIndex < result.images.length)
                  TextButton.icon(
                    style: TextButton.styleFrom(
                        padding: EdgeInsets.zero,
                        visualDensity: VisualDensity.compact),
                    onPressed: () => _viewFinding(item),
                    icon: const Icon(Icons.image_search_rounded, size: 16),
                    label: const Text('See on photo'),
                  ),
              ],
            ),
          ),
          SegmentedButton<bool>(
            showSelectedIcon: false,
            emptySelectionAllowed: true,
            style: const ButtonStyle(visualDensity: VisualDensity.compact),
            segments: const [
              ButtonSegment(
                  value: true,
                  tooltip: 'Confirm',
                  icon: Icon(Icons.check_rounded, size: 18)),
              ButtonSegment(
                  value: false,
                  tooltip: 'Dismiss',
                  icon: Icon(Icons.close_rounded, size: 18)),
            ],
            selected: {if (decision != null) decision},
            onSelectionChanged: _saving
                ? null
                : (value) => setState(() {
                      if (value.isEmpty) {
                        _decisions.remove(item.id);
                      } else {
                        _decisions[item.id] = value.first;
                      }
                    }),
          ),
        ],
      ),
    );
  }

  Widget _details(List<String> lines) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (final line in lines.where((line) => line.isNotEmpty))
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(line),
            ),
        ],
      );

  Future<void> _openReference(String url) async {
    final uri = Uri.tryParse(url);
    try {
      if (uri != null && uri.scheme == 'https' && await launchUrl(uri)) return;
    } on PlatformException {
      // Show the URL if the platform could not open a browser.
    }
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Could not open $url')),
    );
  }

  Widget _careCards() {
    final care = result.section('ai_care');
    final laundry = result.care;
    final providedBy = care['provided_by'] as String? ??
        (care['status'] == 'model_suggestion' && care['model'] is String
            ? '${care['model']} care model'
            : 'FabriCare care rules');
    var tips = careTipsFrom(care['care_tips']);
    if (tips.isEmpty) {
      // Scans saved before the care advice existed.
      tips = _legacyTips(
          _entries(result.section('care_guidance')['recommendations']));
    }
    var laundryTips = careTipsFrom(care['laundry_tips']);
    if (laundryTips.isEmpty) {
      laundryTips = careTipsFrom(care['reasons']);
    }
    if (laundryTips.isEmpty) {
      laundryTips = _legacyTips(_entries(laundry['general_guidance']));
    }
    final modelLabel = care['label'] as String?;
    final headline = care['headline'] as String? ??
        modelLabel ??
        laundry['laundry_mode'] as String? ??
        'Check the care label';
    var alerts = _entries(care['alerts']);
    if (alerts.isEmpty && care.isEmpty) {
      alerts = [
        for (final risk in _entries(result.analysis['risk_alerts']))
          {'level': 'medium', 'action': risk['action']},
      ];
    }
    return Column(
      children: [
        if (tips.isNotEmpty)
          CareAdviceCard(tips: tips, providedBy: providedBy),
        LaundryCard(
          headline: headline,
          tips: laundryTips,
          providedBy: providedBy,
          confidence: headline == modelLabel ? care['confidence'] as num? : null,
          labelSettings: {
            if (care['care_label_entered'] == true) 'Care label': 'applied',
          },
        ),
        if (alerts.isNotEmpty)
          SectionCard(
            title: 'Care alerts',
            icon: Icons.warning_amber_rounded,
            child: CareAlertList(alerts: alerts),
          ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final damage = result.damageDetection.allItems;
    final savedReview = result.damageReview;
    final pendingChanges =
        damage.any((item) => _decisions[item.id] != savedReview[item.id]);
    final fabric = result.section('fabric');
    final condition = result.section('condition_assessment');
    final lifespan = result.section('lifespan');
    final laundry = result.care;
    final observations = _entries(condition['observations']);
    final references = {
      for (final ref in _entries(result.analysis['references'])) ref['id']: ref,
    };
    final earlier = earlierScans(result, AppData.instance.history);
    final score = result.analysis['fabric_confidence'] as num?;
    final needsConfirmation =
        result.analysis['fabric_needs_confirmation'] == true ||
            fabric['review_required'] == true ||
            score == null ||
            result.fabricUncertain;
    final scoreText =
        score?.toStringAsFixed(1).replaceFirst(RegExp(r'\.0$'), '');
    final perImage = result.analysis['per_image'] as List? ?? const [];

    return Scaffold(
      appBar: AppBar(title: const Text('Scan result')),
      bottomNavigationBar: DecoratedBox(
        decoration: const BoxDecoration(
          color: AppColors.background,
          border: Border(top: BorderSide(color: AppColors.border)),
        ),
        child: SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 10, 20, 10),
            child: ElevatedButton.icon(
              onPressed: () => Navigator.of(context).pushReplacement(
                  MaterialPageRoute(builder: (_) => const ScanScreen())),
              icon: const Icon(Icons.camera_alt_rounded),
              label: const Text('Scan another garment'),
            ),
          ),
        ),
      ),
      body: SafeArea(
        bottom: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          children: [
            FabricResultHero(
              result: result,
              score: score?.toDouble(),
              scoreText: scoreText,
              needsConfirmation: needsConfirmation,
              threshold:
                  (fabric['threshold_percent'] as num?)?.toDouble() ?? 85,
            ),
            _wholeGarmentCard(),
            _careCards(),
            SectionCard(
              title: 'Fabric condition',
              icon: Icons.favorite_outline,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Only the referenced method reports a score; legacy records never show one.
                  if (result.section('health_assessment') case final health
                      when health['score'] is num)
                    HealthScoreBadge(health: health),
                  Text(result.healthRating,
                      style: const TextStyle(fontWeight: FontWeight.w600)),
                  if (condition['quality_limited'] == true ||
                      condition['coverage_limited'] == true)
                    const Padding(
                      padding: EdgeInsets.only(top: 4),
                      child: Text('Photo quality may hide some details.',
                          style: TextStyle(
                              fontSize: 12.5, color: AppColors.inkSoft)),
                    ),
                  if (observations.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: [
                        for (final item in observations)
                          Chip(
                            visualDensity: VisualDensity.compact,
                            avatar: Icon(
                                item['confirmed'] == true
                                    ? Icons.check_circle_rounded
                                    : Icons.help_outline_rounded,
                                size: 16),
                            label: Text(
                                '${_display(item['type'])} · ${item['location'] ?? _display(item['view'])}'),
                          ),
                      ],
                    ),
                  ],
                  if (damage.isNotEmpty)
                    ExpansionTile(
                      tilePadding: EdgeInsets.zero,
                      title: const Text('Review findings',
                          style: TextStyle(fontWeight: FontWeight.w700)),
                      subtitle: const Text('Confirm or dismiss each mark',
                          style: TextStyle(fontSize: 12.5)),
                      initiallyExpanded:
                          savedReview.isEmpty && damage.length <= 2,
                      expandedCrossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        for (final item in damage)
                          _reviewFinding(item, savedReview),
                        const SizedBox(height: 12),
                        FilledButton(
                          onPressed: !_saving &&
                                  pendingChanges &&
                                  _decisions.isNotEmpty
                              ? _saveReview
                              : null,
                          child: Text(_saving ? 'Saving…' : 'Save review'),
                        ),
                      ],
                    ),
                ],
              ),
            ),
            SectionCard(
              title: 'Remaining lifespan',
              icon: Icons.hourglass_bottom,
              child: lifespan['status'] == 'experimental_estimate' &&
                      lifespan['remaining_months'] is num
                  ? LifespanEstimate(lifespan: lifespan)
                  : _details([
                      lifespan['summary'] as String? ??
                          'Rescan for a lifespan estimate.',
                    ]),
            ),
            if (earlier.isNotEmpty)
              SectionCard(
                title: 'Track changes',
                icon: Icons.compare_rounded,
                trailing: FilledButton.tonalIcon(
                  onPressed: () => showScanCompareSheet(context,
                      current: result, earlier: earlier),
                  icon: const Icon(Icons.swap_horiz_rounded, size: 18),
                  label: const Text('Compare'),
                ),
                child: Text(
                  '${earlier.length} earlier ${earlier.length == 1 ? 'scan' : 'scans'} to compare with.',
                  style: const TextStyle(color: AppColors.inkSoft),
                ),
              ),
            SectionCard(
              title: 'Sources and method',
              icon: Icons.article_outlined,
              child: ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text('Why this result?'),
                expandedCrossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _details([
                    if (result.fabricModelName case final String model)
                      'Fabric model · $model',
                    if (fabric['reasoning'] case final String reason) reason,
                    if (result.hasScientificAssessment)
                      result.healthStatusNote,
                    if (laundry['explanation'] case final String explanation)
                      explanation,
                  ]),
                  for (final reference in references.values)
                    TextButton.icon(
                      onPressed: () =>
                          _openReference(reference['url'] as String),
                      icon: const Icon(Icons.open_in_new, size: 18),
                      label: Text(reference['title'] as String),
                    ),
                  if (result.hasScientificAssessment)
                    ExpansionTile(
                      tilePadding: EdgeInsets.zero,
                      title: const Text('Advanced details'),
                      children: [
                        _details([
                          'Method · ${result.analysis['methodology_version']}',
                          result.usageStatus,
                          if (fabric['threshold_percent']
                              case final num threshold)
                            'Fabric review threshold · $threshold%',
                          for (final (name, data) in [
                            ('Condition', condition),
                            ('Lifespan', lifespan),
                            ('Laundry', laundry),
                          ])
                            '$name · ${data['status']} · ${data['provenance']}',
                          for (final item in observations)
                            '${item['type']} · ${item['explanation']} '
                                '${item['area_ratio'] is num ? 'Box coverage: ${((item['area_ratio'] as num) * 100).toStringAsFixed(1)}% of photo.' : ''}',
                        ]),
                      ],
                    ),
                ],
              ),
            ),
            if (result.images.isNotEmpty)
              SectionCard(
                title: 'Scan photos',
                icon: Icons.photo_library_outlined,
                child: ExpansionTile(
                  tilePadding: EdgeInsets.zero,
                  title: Text(
                      '${result.images.length} ${result.images.length == 1 ? 'photo' : 'photos'} with marked findings'),
                  children: [
                    for (var i = 0; i < result.images.length; i++) ...[
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Padding(
                          padding: const EdgeInsets.only(bottom: 6, top: 4),
                          child: Text(_display(result.images[i].view),
                              style:
                                  const TextStyle(fontWeight: FontWeight.w700)),
                        ),
                      ),
                      DetectionImage(scan: result, imageIndex: i),
                      if (i < perImage.length &&
                          perImage[i]['quality']?['passed'] == false)
                        Text(
                            'Quality warning · ${perImage[i]['quality']['reason']}'),
                      const SizedBox(height: 10),
                    ],
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
