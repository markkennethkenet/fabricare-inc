import 'package:flutter/material.dart';
import '../../data/mock_data.dart';
import '../../models/scan_result.dart';
import '../../theme/app_theme.dart';
import '../../widgets/care_insights.dart';
import '../../widgets/fabric_thumb.dart';
import '../../widgets/section_card.dart';
import '../scan/scan_result_screen.dart';

class GarmentProfileScreen extends StatefulWidget {
  final ScanResult result;
  const GarmentProfileScreen({super.key, required this.result});

  @override
  State<GarmentProfileScreen> createState() => _GarmentProfileScreenState();
}

class _GarmentProfileScreenState extends State<GarmentProfileScreen> {
  late final TextEditingController _notes =
      TextEditingController(text: widget.result.notes);

  @override
  Widget build(BuildContext context) {
    final r = AppData.instance.history.firstWhere(
      (scan) =>
          scan.id == widget.result.id && scan.userId == widget.result.userId,
      orElse: () => widget.result,
    );
    return Scaffold(
      appBar: AppBar(title: const Text('Saved scan')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
          children: [
            Center(
              child: FabricThumb(
                  imagePath: r.imagePath,
                  fabricType: r.fabricType,
                  size: 120,
                  radius: 24),
            ),
            const SizedBox(height: 16),
            Center(
              child: Text(r.garmentName,
                  style: const TextStyle(
                      fontWeight: FontWeight.w800, fontSize: 19)),
            ),
            const SizedBox(height: 4),
            Center(
              child: Text('Scanned ${_fmtDate(r.dateScanned)}',
                  style: const TextStyle(
                      color: AppColors.inkSoft, fontSize: 12.5)),
            ),
            const SizedBox(height: 22),
            SectionCard(
              title: 'Overview',
              icon: Icons.info_outline_rounded,
              child: Column(
                children: [
                  LabelRow(
                      icon: Icons.texture_outlined,
                      label: 'Fabric',
                      value: r.fabricUncertain || !r.hasFabricScore
                          ? r.fabricLabel
                          : '${r.fabricLabel} · ${r.confidencePercent}%'),
                  if (r.fabricModelName case final modelName?)
                    LabelRow(
                        icon: Icons.tune,
                        label: 'Model',
                        value: modelName),
                  if (r.healthScore case final int score)
                    LabelRow(
                        icon: Icons.favorite_outline_rounded,
                        label: 'Health',
                        value:
                            '$score/100${r.healthGrade == null ? '' : ' · ${r.healthGrade}'}')
                  else if (r.hasHealthAssessment)
                    LabelRow(
                        icon: Icons.favorite_outline_rounded,
                        label: 'Condition',
                        value: r.healthRating),
                  LabelRow(
                      icon: Icons.hourglass_bottom_rounded,
                      label: 'Lifespan',
                      value: r.section('lifespan')['remaining_months'] is num
                          ? r.lifespanRange
                          : 'Not estimated'),
                  LabelRow(
                      icon: Icons.local_laundry_service_outlined,
                      label: 'Laundry',
                      value: r.laundryHeadline),
                ],
              ),
            ),
            if (careTipsFrom(r.section('ai_care')['care_tips'])
                case final tips when tips.isNotEmpty)
              CareAdviceCard(
                tips: tips,
                providedBy: r.section('ai_care')['provided_by'] as String? ??
                    'FabriCare care model',
              ),
            ElevatedButton.icon(
                onPressed: () async {
                  await Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => ScanResultScreen(result: r)));
                  if (mounted) setState(() {});
                },
                icon: const Icon(Icons.open_in_full_rounded, size: 18),
                label: const Text('Open full result')),
            const SizedBox(height: 12),
            SectionCard(
              title: 'Notes',
              icon: Icons.edit_note_rounded,
              child: TextField(
                controller: _notes,
                maxLines: 3,
                decoration: const InputDecoration(
                    hintText: 'Add a note about this garment…'),
                onChanged: (v) async {
                  try {
                    await AppData.instance.updateNotes(r.id, v);
                  } catch (_) {
                    if (context.mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                          content: Text('Could not save notes.')));
                    }
                  }
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _notes.dispose();
    super.dispose();
  }

  static String _fmtDate(DateTime d) {
    final months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec'
    ];
    return '${months[d.month - 1]} ${d.day}, ${d.year}';
  }
}
