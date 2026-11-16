import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../data/mock_data.dart';
import '../../models/damage_detection.dart';
import '../../models/scan_result.dart';
import '../../services/fabricare_api_service.dart';
import '../../widgets/detection_image.dart';
import '../../widgets/section_card.dart';
import 'scan_screen.dart';

/// Displays assessment supplied by the backend; no local scoring rules.
class ScanResultScreen extends StatefulWidget {
  final ScanResult result;
  const ScanResultScreen({super.key, required this.result});

  @override
  State<ScanResultScreen> createState() => _ScanResultScreenState();
}

class _ScanResultScreenState extends State<ScanResultScreen> {
  late ScanResult result = widget.result;
  late final Map<String, bool> _decisions = result.damageReview;
  bool _saving = false;
  bool _checkingVision = false;
  Map<String, dynamic>? _visionOpinion;
  String? _visionError;

  Future<void> _askVision(ScanImage front) async {
    setState(() {
      _checkingVision = true;
      _visionError = null;
    });
    try {
      final opinion = await FabricCareApiService.instance.askLocalVision(front);
      if (mounted) setState(() => _visionOpinion = opinion);
    } catch (error) {
      if (mounted) setState(() => _visionError = error.toString());
    } finally {
      if (mounted) setState(() => _checkingVision = false);
    }
  }

  void _viewFinding(DamageItem item) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('${item.label} · ${item.view.replaceAll('_', ' ')}'),
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
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Damage review saved.')));
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not save the damage review. $error')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  static List<Map> _entries(Object? values) =>
      values is List ? values.whereType<Map>().toList() : const [];

  static String _display(Object? value) =>
      value?.toString().replaceAll('_', ' ') ?? 'unavailable';

  static List<String> _actions(List<Map> entries) =>
      entries.map((entry) => entry['action']).whereType<String>().toList();

  Widget _reviewFinding(DamageItem item, Map<String, bool> savedReview) {
    final decision = _decisions[item.id];
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('${item.label} · ${_display(item.view)}',
              style: Theme.of(context).textTheme.titleSmall),
          Text(switch (savedReview[item.id]) {
            true => 'Confirmed by you',
            false => 'Dismissed',
            null => 'Candidate · ${item.confidencePercent}% detector score',
          }),
          if (item.imageIndex >= 0 && item.imageIndex < result.images.length)
            TextButton(
              onPressed: () => _viewFinding(item),
              child: const Text('View marked photo'),
            ),
          Wrap(
            spacing: 8,
            children: [
              ChoiceChip(
                label: const Text('Confirm'),
                selected: decision == true,
                onSelected: _saving
                    ? null
                    : (_) => setState(() => _decisions[item.id] = true),
              ),
              ChoiceChip(
                label: const Text('Dismiss'),
                selected: decision == false,
                onSelected: _saving
                    ? null
                    : (_) => setState(() => _decisions[item.id] = false),
              ),
            ],
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

  Future<void> _comparePhotos(List<ScanResult> previousScans) async {
    final previous = await showDialog<ScanResult>(
      context: context,
      builder: (context) => SimpleDialog(
        title: const Text('Choose an earlier scan of this garment'),
        children: [
          for (final scan in previousScans)
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, scan),
              child: Text('${scan.garmentName} · ${_scanDate(scan)}'),
            ),
        ],
      ),
    );
    if (previous == null || !mounted) return;
    final photos = [
      for (final scan in [previous, result])
        (
          scan,
          scan.images
              .firstWhere((photo) => photo.view == 'front')
              .file
              .readAsBytes()
        ),
    ];
    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Compare front photos'),
        content: SingleChildScrollView(
          child: SizedBox(
            width: 540,
            child: Column(
              children: [
                const Text(
                  'Compare the same areas under similar lighting and framing. '
                  'Differences in the photos can affect what you see.',
                ),
                const SizedBox(height: 12),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (final (scan, photo) in photos)
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                          child: Column(
                            children: [
                              Text(_scanDate(scan)),
                              FutureBuilder(
                                future: photo,
                                builder: (context, snapshot) {
                                  if (snapshot.hasError) {
                                    return const Text('Photo unavailable');
                                  }
                                  final bytes = snapshot.data;
                                  if (bytes == null) {
                                    return const SizedBox(height: 180);
                                  }
                                  return Image.memory(
                                    bytes,
                                    height: 180,
                                    fit: BoxFit.contain,
                                    errorBuilder: (_, __, ___) =>
                                        const Text('Photo unavailable'),
                                  );
                                },
                              ),
                              Text(scan.healthRating),
                            ],
                          ),
                        ),
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  String _scanDate(ScanResult scan) =>
      MaterialLocalizations.of(context).formatMediumDate(scan.dateScanned);

  @override
  Widget build(BuildContext context) {
    final damage = result.damageDetection.allItems;
    final savedReview = result.damageReview;
    final pendingChanges =
        damage.any((item) => _decisions[item.id] != savedReview[item.id]);
    final fabric = result.section('fabric');
    final condition = result.section('condition_assessment');
    final lifespan = result.section('lifespan');
    final outlook = result.section('six_month_outlook');
    final guidance = result.section('care_guidance');
    final laundry = result.care;
    final risks = _entries(result.analysis['risk_alerts']);
    final recommendations = _entries(guidance['recommendations']);
    final observations = _entries(condition['observations']);
    final references = {
      for (final ref in _entries(result.analysis['references'])) ref['id']: ref,
    };
    final confirmedSettings = laundry['confirmed_settings'] as Map? ?? const {};
    final legacyWash = laundry['laundry_mode'] as String?;
    final enteredWash = confirmedSettings['Wash'] ??
        (laundry['confirmed_settings'] == null &&
                !const {
                  'Unknown',
                  'Verify the care label',
                  'Check the care label'
                }.contains(legacyWash)
            ? legacyWash
            : null);
    final previousScans = result.images.any((photo) => photo.view == 'front')
        ? AppData.instance.history
            .where((scan) =>
                scan.userId == result.userId &&
                scan.dateScanned.isBefore(result.dateScanned) &&
                scan.images.any((photo) => photo.view == 'front'))
            .toList()
        : const <ScanResult>[];
    final score = result.analysis['fabric_confidence'] as num?;
    final needsConfirmation =
        result.analysis['fabric_needs_confirmation'] == true ||
            fabric['review_required'] == true ||
            score == null ||
            result.fabricUncertain;
    final scoreText =
        score?.toStringAsFixed(1).replaceFirst(RegExp(r'\.0$'), '');
    final perImage = result.analysis['per_image'] as List? ?? const [];
    final secondOpinion = result.section('fabric_second_opinion');
    final secondLabel = secondOpinion['fabric_type'];
    final secondModel = secondOpinion['model'];
    final frontPhoto =
        result.images.where((photo) => photo.view == 'front').firstOrNull;
    final visionLabel = _visionOpinion?['fabric_type'];

    return Scaffold(
      appBar: AppBar(title: const Text('FabriCare Result')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
          children: [
            SectionCard(
              title: 'Fabric Identification',
              icon: Icons.texture_outlined,
              child: _details([
                result.fabricUncertain
                    ? 'The photos give conflicting fabric results.'
                    : '${result.fabricType}${scoreText == null ? '' : ' — $scoreText% model score'}',
                if (result.fabricUncertain)
                  'Retake the front and back of one garment. Scan other garments separately.',
                if (result.analysis['fabric_close_up_excluded'] == true)
                  'The close-up suggested a different fabric. Identification used the clear front or back photos.',
                if (needsConfirmation) 'Needs confirmation',
              ]),
            ),
            if (secondOpinion.isNotEmpty)
              SectionCard(
                title: 'AI second opinion',
                icon: Icons.fact_check_outlined,
                child: _details([
                  if (secondLabel is String && secondModel is String)
                    '$secondModel suggests $secondLabel.',
                  switch (secondOpinion['status']) {
                    'agrees' =>
                      'Both models suggest the same fabric. Check the fiber label if you need to confirm the composition.',
                    'disagrees' =>
                      'The models suggest different fabrics. Treat the fabric result as provisional and check the fiber label or capture another view.',
                    'inconclusive' =>
                      'The models could not settle on one fabric. Try another clear view.',
                    _ => 'The second model was unavailable for this scan.',
                  },
                ]),
              ),
            if (frontPhoto != null)
              SectionCard(
                title: 'Local Qwen opinion',
                icon: Icons.visibility_outlined,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Qwen3-VL reviews the front photo on your PC.'),
                    if (_checkingVision) ...[
                      const LinearProgressIndicator(),
                      const Text('Qwen is reviewing the photo…'),
                    ] else if (_visionError != null)
                      Text(_visionError!)
                    else if (_visionOpinion != null)
                      Text(switch (visionLabel) {
                        null => 'Qwen returned no final answer.',
                        'uncertain' => 'Qwen was uncertain about the fabric.',
                        _ when result.fabricUncertain =>
                          'Qwen suggests $visionLabel. FabriCare is uncertain.',
                        _ when visionLabel == result.fabricType =>
                          'Both suggest $visionLabel. Check the fiber label to confirm.',
                        _ =>
                          'Qwen suggests $visionLabel. FabriCare suggests ${result.fabricType}. Check the fiber label.',
                      }),
                    OutlinedButton(
                      onPressed:
                          _checkingVision ? null : () => _askVision(frontPhoto),
                      child: Text(_visionOpinion == null
                          ? 'Ask Qwen'
                          : 'Run Qwen again'),
                    ),
                  ],
                ),
              ),
            SectionCard(
              title: 'Visible Fabric Condition',
              icon: Icons.favorite_outline,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(result.healthRating),
                  if (condition['quality_limited'] == true)
                    const Text('Photo quality may hide details.'),
                  if (condition['coverage_limited'] == true)
                    const Text('Some photos could not be checked for damage.'),
                  for (final item in observations)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text('${_display(item['type'])} · '
                          '${item['location'] ?? _display(item['view'])} · '
                          '${item['confirmed'] == true || item['provenance'] == 'user_reported' ? 'Confirmed by you' : 'Candidate'}'),
                    ),
                  if (damage.isNotEmpty)
                    ExpansionTile(
                      key: const PageStorageKey('damage-review'),
                      tilePadding: EdgeInsets.zero,
                      title: const Text('Review Damage'),
                      initiallyExpanded:
                          savedReview.isEmpty && damage.length <= 2,
                      expandedCrossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                            'Check each marked area. You can save the findings you have reviewed so far.'),
                        for (final item in damage)
                          _reviewFinding(item, savedReview),
                        FilledButton(
                          onPressed: !_saving &&
                                  pendingChanges &&
                                  _decisions.isNotEmpty
                              ? _saveReview
                              : null,
                          child:
                              Text(_saving ? 'Saving…' : 'Save damage review'),
                        ),
                        if (pendingChanges)
                          const Text(
                              'Save to update the assessment and recommendations.'),
                      ],
                    ),
                ],
              ),
            ),
            SectionCard(
              title: 'Durability and remaining lifespan',
              icon: Icons.hourglass_bottom,
              child: _details([
                lifespan['summary'] as String? ??
                    'Check seams and high-friction areas for changes across scans.',
                'Remaining months: undetermined',
                if (lifespan.isEmpty) 'Rescan for current guidance.',
              ]),
            ),
            SectionCard(
              title: 'Six-Month Condition Outlook',
              icon: Icons.timeline,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _details([
                    'Care scenario',
                    ...List<String>.from(
                        outlook['concerns'] as List? ?? const []),
                    if (outlook.isEmpty)
                      'Save a new scan for a baseline comparison.',
                  ]),
                  if (previousScans.isNotEmpty)
                    OutlinedButton.icon(
                      onPressed: () => _comparePhotos(previousScans),
                      icon: const Icon(Icons.compare_outlined),
                      label: const Text('Compare with an earlier scan'),
                    ),
                ],
              ),
            ),
            SectionCard(
              title: 'Care Recommendations',
              icon: Icons.checkroom_outlined,
              child: _details([
                ..._actions(recommendations),
                if (recommendations.isEmpty)
                  'Set aside bleach and direct heat until you can check the garment instructions.',
              ]),
            ),
            SectionCard(
              title: 'Laundry Suggestions',
              icon: Icons.local_laundry_service_outlined,
              child: _details([
                'Confirmed Laundry Instructions',
                enteredWash == null
                    ? 'Confirmed washing mode: Unknown'
                    : 'Wash · $enteredWash',
                for (final entry in confirmedSettings.entries)
                  if (entry.key != 'Wash') '${entry.key} · ${entry.value}',
                if (laundry['intensity'] case final String intensity
                    when intensity.isNotEmpty)
                  intensity,
                'General Fabric Care Guidance',
                ..._actions(_entries(laundry['general_guidance'])),
                if (_entries(laundry['general_guidance']).isEmpty)
                  'Set uncertain items aside before laundering.',
                'Unknown or Restricted Settings',
                ...List<String>.from(laundry['unknown_settings'] as List? ??
                    const [
                      'Check the complete instructions for any missing settings.'
                    ]),
              ]),
            ),
            if (risks.isNotEmpty)
              SectionCard(
                title: 'Care Risk Alerts',
                icon: Icons.warning_amber_rounded,
                child: _details(_actions(risks)),
              ),
            SectionCard(
              title: 'Scientific Basis',
              icon: Icons.article_outlined,
              child: ExpansionTile(
                key: const PageStorageKey('why-this-result'),
                tilePadding: EdgeInsets.zero,
                title: const Text('Why this result?'),
                expandedCrossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _details([
                    if (result.fabricModelName case final String model) model,
                    if (fabric['reasoning'] case final String reason) reason,
                    if (result.hasScientificAssessment) result.healthStatusNote,
                    if (lifespan['explanation'] case final String explanation)
                      explanation,
                    if (outlook['assumptions'] case final String assumptions)
                      assumptions,
                    if (outlook['explanation'] case final String explanation)
                      explanation,
                    if (laundry['explanation'] case final String explanation)
                      explanation,
                    for (final entry in [...recommendations, ...risks]) ...[
                      entry['action'] as String,
                      [
                        entry['selected_because'],
                        entry['reason'],
                        entry['restriction']
                      ]
                          .whereType<String>()
                          .where((text) => text.isNotEmpty)
                          .join(' '),
                      (entry['reference_ids'] as List? ?? const [])
                          .map((id) => references[id]?['title'])
                          .whereType<String>()
                          .join(' · '),
                    ],
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
                      title: const Text('Advanced scientific details'),
                      children: [
                        _details([
                          'Method · ${result.analysis['methodology_version']}',
                          result.usageStatus,
                          if (fabric['threshold_percent']
                              case final num threshold)
                            'Model review threshold · $threshold%. Scores are uncalibrated.',
                          for (final (name, data) in [
                            ('Condition', condition),
                            ('Lifespan', lifespan),
                            ('Outlook', outlook),
                            ('Care', guidance),
                            ('Laundry', laundry),
                          ])
                            '$name · ${data['status']} · ${data['provenance']}',
                          for (final item in observations)
                            '${item['type']} · ${item['explanation']} '
                                '${item['area_ratio'] is num ? 'Box coverage: ${((item['area_ratio'] as num) * 100).toStringAsFixed(1)}% of photo.' : ''}',
                          'Box coverage = (box width × box height) ÷ (photo width × photo height). '
                              'The box center identifies the region in a three-by-three photo grid.',
                        ]),
                      ],
                    ),
                ],
              ),
            ),
            if (result.images.isNotEmpty)
              ExpansionTile(
                title: const Text('Scan photos'),
                children: [
                  for (var i = 0; i < result.images.length; i++) ...[
                    Text(_display(result.images[i].view)),
                    DetectionImage(scan: result, imageIndex: i),
                    if (i < perImage.length &&
                        perImage[i]['quality']?['passed'] == false)
                      Text(
                          'Quality warning · ${perImage[i]['quality']['reason']}'),
                  ],
                ],
              ),
            OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pushReplacement(
                  MaterialPageRoute(builder: (_) => const ScanScreen())),
              icon: const Icon(Icons.camera_alt_outlined),
              label: const Text('Scan another garment'),
            ),
          ],
        ),
      ),
    );
  }
}
