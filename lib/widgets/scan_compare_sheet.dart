import 'package:flutter/material.dart';

import '../models/scan_result.dart';
import '../theme/app_theme.dart';
import 'fabric_result_hero.dart';
import 'scan_photo.dart';

/// Earlier scans in the same account that have a front photo, newest first.
List<ScanResult> earlierScans(
        ScanResult current, Iterable<ScanResult> history) =>
    [
      for (final scan in history)
        if (scan.id != current.id &&
            scan.userId == current.userId &&
            scan.dateScanned.isBefore(current.dateScanned) &&
            scan.images.any((photo) => photo.view == 'front'))
          scan,
    ]..sort((a, b) => b.dateScanned.compareTo(a.dateScanned));

Future<void> showScanCompareSheet(BuildContext context,
        {required ScanResult current, required List<ScanResult> earlier}) =>
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      backgroundColor: AppColors.surface,
      builder: (_) => ScanCompareSheet(current: current, earlier: earlier),
    );

/// Side-by-side comparison of this scan with an earlier one the user picks.
class ScanCompareSheet extends StatefulWidget {
  final ScanResult current;
  final List<ScanResult> earlier;
  const ScanCompareSheet(
      {super.key, required this.current, required this.earlier});

  @override
  State<ScanCompareSheet> createState() => _ScanCompareSheetState();
}

class _ScanCompareSheetState extends State<ScanCompareSheet> {
  // Start with the latest scan of the same fabric, if there is one.
  late ScanResult _earlier = widget.earlier.firstWhere(
      (scan) => scan.fabricType == widget.current.fabricType,
      orElse: () => widget.earlier.first);

  String _date(ScanResult scan) =>
      MaterialLocalizations.of(context).formatMediumDate(scan.dateScanned);

  static String fabricName(ScanResult scan) => scan.fabricUncertain
      ? 'Uncertain'
      : FabricResultHero.displayName(scan.fabricType);

  static num? _health(ScanResult scan) => scan.analysis['health_score'] as num?;

  static int _findings(ScanResult scan) =>
      (scan.section('condition_assessment')['observations'] as List?)
          ?.length ??
      scan.damageDetection.allItems.length;

  static num? _months(ScanResult scan) =>
      scan.section('lifespan')['remaining_months'] as num?;

  @override
  Widget build(BuildContext context) {
    final current = widget.current;
    final before = _earlier;
    final healthBefore = _health(before), healthNow = _health(current);
    final monthsBefore = _months(before), monthsNow = _months(current);
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Compare scans',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900)),
          const SizedBox(height: 14),
          DropdownButtonFormField<String>(
            initialValue: before.id,
            isExpanded: true,
            decoration: const InputDecoration(
              labelText: 'Earlier scan',
              prefixIcon: Icon(Icons.history_rounded),
            ),
            items: [
              for (final scan in widget.earlier)
                DropdownMenuItem(
                  value: scan.id,
                  child: Text('${fabricName(scan)} · ${_date(scan)}',
                      overflow: TextOverflow.ellipsis),
                ),
            ],
            onChanged: (id) => setState(() => _earlier =
                widget.earlier.firstWhere((scan) => scan.id == id)),
          ),
          const SizedBox(height: 16),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                  child: _PhotoPanel(
                      label: 'Earlier · ${_date(before)}', scan: before)),
              const SizedBox(width: 12),
              Expanded(
                  child: _PhotoPanel(
                      label: 'Now · ${_date(current)}',
                      scan: current,
                      highlight: true)),
            ],
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.background,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              children: [
                _CompareRow(
                    label: 'Fabric',
                    before: fabricName(before),
                    now: fabricName(current)),
                if (healthBefore != null && healthNow != null)
                  _CompareRow(
                    label: 'Health',
                    before: '${healthBefore.round()}/100',
                    now: '${healthNow.round()}/100',
                    change: healthNow.round() - healthBefore.round(),
                  ),
                _CompareRow(
                  label: 'Findings',
                  before: '${_findings(before)}',
                  now: '${_findings(current)}',
                  change: _findings(current) - _findings(before),
                  lowerIsBetter: true,
                ),
                if (monthsBefore != null && monthsNow != null)
                  _CompareRow(
                    label: 'Lifespan',
                    before: '~${monthsBefore.round()} mo',
                    now: '~${monthsNow.round()} mo',
                    change: monthsNow.round() - monthsBefore.round(),
                    last: true,
                  ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          const Row(
            children: [
              Icon(Icons.wb_sunny_outlined, size: 15, color: AppColors.inkSoft),
              SizedBox(width: 6),
              Expanded(
                child: Text('Same light and framing give a fair comparison.',
                    style: TextStyle(fontSize: 12.5, color: AppColors.inkSoft)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _PhotoPanel extends StatelessWidget {
  final String label;
  final ScanResult scan;
  final bool highlight;
  const _PhotoPanel(
      {required this.label, required this.scan, this.highlight = false});

  @override
  Widget build(BuildContext context) {
    final front = scan.images.firstWhere((photo) => photo.view == 'front');
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
                fontSize: 12.5,
                fontWeight: FontWeight.w800,
                color: highlight ? AppColors.coral : AppColors.inkSoft)),
        const SizedBox(height: 6),
        AspectRatio(
          aspectRatio: 3 / 4,
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                  color: highlight ? AppColors.coral : AppColors.border,
                  width: highlight ? 2 : 1),
            ),
            clipBehavior: Clip.antiAlias,
            child: ScanPhoto(
              front.path,
              fit: BoxFit.cover,
              fallback: const ColoredBox(
                color: AppColors.surfaceMuted,
                child: Center(
                    child: Icon(Icons.image_not_supported_outlined,
                        color: AppColors.inkSoft)),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _CompareRow extends StatelessWidget {
  final String label;
  final String before;
  final String now;
  final int? change;
  final bool lowerIsBetter;
  final bool last;
  const _CompareRow({
    required this.label,
    required this.before,
    required this.now,
    this.change,
    this.lowerIsBetter = false,
    this.last = false,
  });

  @override
  Widget build(BuildContext context) {
    final delta = change;
    final better = delta == null || delta == 0
        ? null
        : (delta > 0) != lowerIsBetter;
    final color = switch (better) {
      true => AppColors.sage,
      false => AppColors.danger,
      null => AppColors.inkSoft,
    };
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        border: last
            ? null
            : const Border(bottom: BorderSide(color: AppColors.border)),
      ),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(label,
                style: const TextStyle(
                    color: AppColors.inkSoft, fontWeight: FontWeight.w600)),
          ),
          Expanded(
              child: Text(before,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: AppColors.inkSoft))),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 6),
            child: Icon(Icons.arrow_forward_rounded,
                size: 16, color: AppColors.inkSoft),
          ),
          Expanded(
              child: Text(now,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontWeight: FontWeight.w800))),
          if (delta != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                delta == 0 ? 'Same' : (delta > 0 ? '+$delta' : '−${-delta}'),
                style: TextStyle(
                    color: color, fontWeight: FontWeight.w800, fontSize: 12),
              ),
            ),
        ],
      ),
    );
  }
}
