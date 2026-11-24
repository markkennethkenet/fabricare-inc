import 'package:flutter/material.dart';

import '../models/scan_result.dart';
import '../theme/app_theme.dart';
import 'scan_photo.dart';

/// The large fabric result at the top of the scan result screen.
class FabricResultHero extends StatelessWidget {
  final ScanResult result;
  final double? score;
  final String? scoreText;
  final bool needsConfirmation;
  final double threshold;

  const FabricResultHero({
    super.key,
    required this.result,
    required this.score,
    required this.scoreText,
    required this.needsConfirmation,
    this.threshold = 85,
  });

  static String displayName(String label) => label.isEmpty
      ? label
      : label
          .replaceAll('_', ' ')
          .split(' ')
          .map((word) =>
              word.isEmpty ? word : word[0].toUpperCase() + word.substring(1))
          .join(' ');

  ({String label, Color color, IconData icon}) get _level {
    final value = score;
    if (result.fabricUnsupported) {
      return (
        label: 'Unsupported fabric type',
        color: AppColors.danger,
        icon: Icons.block_rounded
      );
    }
    if (result.fabricUncertain) {
      return (
        label: 'Uncertain result',
        color: AppColors.danger,
        icon: Icons.help_outline_rounded
      );
    }
    if (value == null) {
      return (
        label: 'No model score for this model',
        color: AppColors.inkSoft,
        icon: Icons.info_outline
      );
    }
    if (value >= threshold) {
      return (
        label: 'Confident result',
        color: AppColors.sage,
        icon: Icons.verified_rounded
      );
    }
    if (value >= threshold - 15) {
      return (
        label: 'Likely · worth a check',
        color: AppColors.mustard,
        icon: Icons.adjust_rounded
      );
    }
    return (
      label: 'Low model score',
      color: AppColors.danger,
      icon: Icons.error_outline_rounded
    );
  }

  List<MapEntry<String, double>> get _alternatives {
    final probabilities = result.analysis['fabric_probabilities'];
    if (probabilities is! Map) return const [];
    final entries = [
      for (final entry in probabilities.entries)
        if (entry.key is String &&
            entry.value is num &&
            entry.key != result.fabricType)
          MapEntry(entry.key as String, (entry.value as num).toDouble()),
    ]..sort((a, b) => b.value.compareTo(a.value));
    return entries.where((entry) => entry.value >= 0.03).take(3).toList();
  }

  @override
  Widget build(BuildContext context) {
    final level = _level;
    final front =
        result.images.where((photo) => photo.view == 'front').firstOrNull;
    final name = result.fabricUnsupported
        ? 'Unsupported'
        : result.fabricUncertain
            ? 'Uncertain'
            : displayName(result.fabricType);
    final alternatives = _alternatives;
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFFFFFCF3), Color(0xFFF6E3D3)],
        ),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.texture_rounded,
                            size: 16, color: AppColors.coral),
                        SizedBox(width: 6),
                        Text('IDENTIFIED FABRIC',
                            style: TextStyle(
                                color: AppColors.coral,
                                fontWeight: FontWeight.w800,
                                fontSize: 12,
                                letterSpacing: 1.1)),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Semantics(
                      header: true,
                      child: FittedBox(
                        fit: BoxFit.scaleDown,
                        alignment: Alignment.centerLeft,
                        child: Text(
                          name,
                          style: const TextStyle(
                            fontSize: 44,
                            height: 1.05,
                            fontWeight: FontWeight.w900,
                            color: AppColors.ink,
                            letterSpacing: -0.5,
                          ),
                        ),
                      ),
                    ),
                    if (result.fabricModelName case final String model) ...[
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppColors.ink.withValues(alpha: 0.06),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(model,
                            style: const TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                color: AppColors.ink)),
                      ),
                    ],
                  ],
                ),
              ),
              if (front != null) ...[
                const SizedBox(width: 12),
                ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: SizedBox(
                    width: 92,
                    height: 92,
                    child: ScanPhoto(front.path,
                        fit: BoxFit.cover, fallback: const _NoPhoto()),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 18),
          if (score != null &&
              !result.fabricUncertain &&
              !result.fabricUnsupported) ...[
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text('$scoreText%',
                    style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.w900,
                        color: level.color,
                        height: 1)),
                const SizedBox(width: 8),
                const Padding(
                  padding: EdgeInsets.only(bottom: 3),
                  child: Text('model score',
                      style: TextStyle(
                          color: AppColors.inkSoft,
                          fontWeight: FontWeight.w600)),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: (score! / 100).clamp(0.0, 1.0),
                minHeight: 10,
                color: level.color,
                backgroundColor: Colors.white.withValues(alpha: 0.7),
              ),
            ),
            const SizedBox(height: 8),
          ],
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: [
              _Pill(icon: level.icon, label: level.label, color: level.color),
              if (needsConfirmation)
                const _Pill(
                    icon: Icons.fact_check_outlined,
                    label: 'Needs confirmation',
                    color: Color(0xFF8A6417)),
            ],
          ),
          if (result.fabricUncertain) ...[
            const SizedBox(height: 12),
            const Text('The photos give conflicting fabric results.',
                style: TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            const Text(
                'Retake the front and back of one garment. Scan other garments separately.'),
          ],
          if (result.analysis['fabric_close_up_excluded'] == true) ...[
            const SizedBox(height: 10),
            const Text(
                'The close-up suggested a different fabric. Identification used the clear front or back photos.'),
          ],
          if (needsConfirmation && !result.fabricUncertain) ...[
            const SizedBox(height: 10),
            const Text(
              'Check the fibre label to confirm.',
              style: TextStyle(color: AppColors.inkSoft, fontSize: 13),
            ),
          ],
          if (alternatives.isNotEmpty) ...[
            const SizedBox(height: 14),
            const Text('Other possibilities',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
            const SizedBox(height: 6),
            for (final entry in alternatives)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                  children: [
                    SizedBox(
                      width: 84,
                      child: Text(displayName(entry.key),
                          style: const TextStyle(fontSize: 13)),
                    ),
                    Expanded(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(5),
                        child: LinearProgressIndicator(
                          value: entry.value.clamp(0.0, 1.0),
                          minHeight: 6,
                          color: AppColors.inkSoft,
                          backgroundColor: Colors.white.withValues(alpha: 0.7),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(
                      width: 40,
                      child: Text('${(entry.value * 100).round()}%',
                          textAlign: TextAlign.right,
                          style: const TextStyle(
                              fontSize: 12.5, color: AppColors.inkSoft)),
                    ),
                  ],
                ),
              ),
          ],
        ],
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  const _Pill({required this.icon, required this.label, required this.color});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.13),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 15, color: color),
            const SizedBox(width: 5),
            Text(label,
                style: TextStyle(
                    color: color, fontWeight: FontWeight.w800, fontSize: 12.5)),
          ],
        ),
      );
}

class _NoPhoto extends StatelessWidget {
  const _NoPhoto();

  @override
  Widget build(BuildContext context) => const ColoredBox(
        color: AppColors.surfaceMuted,
        child: Icon(Icons.checkroom_rounded, color: AppColors.inkSoft),
      );
}
