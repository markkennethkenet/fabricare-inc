import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

typedef CareTip = ({String text, List<String> referenceIds});

/// Reads `[{text, reference_ids}]` entries from the API.
List<CareTip> careTipsFrom(Object? values) => [
      for (final entry in (values is List ? values : const []).whereType<Map>())
        if (entry['text'] case final String text when text.isNotEmpty)
          (
            text: text,
            referenceIds: List<String>.from(
                (entry['reference_ids'] as List? ?? const [])
                    .whereType<String>()),
          ),
    ];

/// Up to three care recommendations written for this scan (API `ai_care.care_tips`).
class CareAdviceCard extends StatelessWidget {
  final List<CareTip> tips;
  final String providedBy;
  const CareAdviceCard(
      {super.key, required this.tips, required this.providedBy});

  @override
  Widget build(BuildContext context) => _AdviceCard(
        title: 'Care recommendations',
        icon: Icons.auto_awesome_rounded,
        providedBy: providedBy,
        children: [
          for (final (index, tip) in tips.take(3).indexed)
            _TipRow(leading: _NumberBadge(index + 1), text: tip.text),
        ],
      );
}

/// The laundry mode and up to three laundry steps (API `ai_care`).
class LaundryCard extends StatelessWidget {
  final String headline;
  final List<CareTip> tips;
  final String providedBy;
  final num? confidence;
  final Map<String, String> labelSettings;
  const LaundryCard({
    super.key,
    required this.headline,
    required this.tips,
    required this.providedBy,
    this.confidence,
    this.labelSettings = const {},
  });

  static const _icons = [
    Icons.local_laundry_service_outlined,
    Icons.soap_outlined,
    Icons.dry_outlined,
  ];

  @override
  Widget build(BuildContext context) => _AdviceCard(
        title: 'Laundry suggestion',
        icon: Icons.local_laundry_service_outlined,
        providedBy: providedBy,
        children: [
          Text(headline,
              style: const TextStyle(
                  fontSize: 21, fontWeight: FontWeight.w900, height: 1.2)),
          if (confidence != null || labelSettings.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                if (confidence != null)
                  _Chip(
                      icon: Icons.speed_rounded,
                      label: '${confidence!.round()}% model confidence'),
                for (final entry in labelSettings.entries)
                  _Chip(
                      icon: Icons.label_outline_rounded,
                      label: '${entry.key}: ${entry.value}'),
              ],
            ),
          ],
          const SizedBox(height: 4),
          for (final (index, tip) in tips.take(3).indexed)
            _TipRow(
              leading: Icon(_icons[index % _icons.length],
                  size: 20, color: AppColors.coral),
              text: tip.text,
            ),
        ],
      );
}

/// Automated care-risk alerts (API `ai_care.alerts`).
class CareAlertList extends StatelessWidget {
  final List<Map> alerts;
  const CareAlertList({super.key, required this.alerts});

  static Color alertColor(String? level) => switch (level) {
        'high' => AppColors.danger,
        'medium' => const Color(0xFF9A6B12),
        _ => AppColors.sage,
      };

  @override
  Widget build(BuildContext context) => Column(
        children: [
          for (final alert in alerts)
            Container(
              width: double.infinity,
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: alertColor(alert['level'] as String?)
                    .withValues(alpha: 0.09),
                borderRadius: BorderRadius.circular(14),
                border: Border(
                  left: BorderSide(
                      color: alertColor(alert['level'] as String?), width: 4),
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    switch (alert['level']) {
                      'high' => Icons.report_rounded,
                      'medium' => Icons.warning_amber_rounded,
                      _ => Icons.info_outline_rounded,
                    },
                    size: 20,
                    color: alertColor(alert['level'] as String?),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (alert['title'] case final String title)
                          Text(title,
                              style:
                                  const TextStyle(fontWeight: FontWeight.w800)),
                        Text(alert['action'] as String? ?? '',
                            style: const TextStyle(height: 1.3)),
                      ],
                    ),
                  ),
                ],
              ),
            ),
        ],
      );
}

/// Visible-condition health score (API `health_assessment`).
class HealthScoreBadge extends StatelessWidget {
  final Map<String, dynamic> health;
  const HealthScoreBadge({super.key, required this.health});

  static Color colorFor(int score) => score >= 85
      ? AppColors.sage
      : score >= 70
          ? const Color(0xFF6E8B3D)
          : score >= 50
              ? AppColors.mustard
              : AppColors.danger;

  @override
  Widget build(BuildContext context) {
    final score = (health['score'] as num).round();
    final grade = health['grade'] as String? ?? '';
    final color = colorFor(score);
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          SizedBox(
            width: 64,
            height: 64,
            child: Stack(
              alignment: Alignment.center,
              children: [
                SizedBox.expand(
                  child: CircularProgressIndicator(
                    value: score / 100,
                    strokeWidth: 6,
                    color: color,
                    backgroundColor: AppColors.surfaceMuted,
                  ),
                ),
                Text('$score',
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w900,
                        color: color)),
              ],
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Fabric health $score/100',
                    style: const TextStyle(
                        fontWeight: FontWeight.w800, fontSize: 15)),
                Text(grade,
                    style:
                        TextStyle(color: color, fontWeight: FontWeight.w700)),
                const Text('From the visible findings in your photos.',
                    style: TextStyle(fontSize: 12, color: AppColors.inkSoft)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Referenced remaining-lifespan range (API `lifespan`, status experimental_estimate).
class LifespanEstimate extends StatelessWidget {
  final Map<String, dynamic> lifespan;
  const LifespanEstimate({super.key, required this.lifespan});

  static String sourceLabel(String? source) => switch (source) {
        'D' => 'Laitala & Klepp 2020',
        'user' => 'Your details',
        'fabricare_assumption' => 'FabriCare assumption',
        'health_index' => 'Fabric Health Score',
        _ => '',
      };

  @override
  Widget build(BuildContext context) {
    final low = lifespan['remaining_months_low'] as num? ?? 0;
    final high = lifespan['remaining_months_high'] as num? ?? 0;
    final central = lifespan['remaining_months'] as num? ?? 0;
    final factors = (lifespan['factors'] as List? ?? const []).whereType<Map>();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('About ${low.round()}–${high.round()} months',
            style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w900)),
        Text('Estimated remaining use · middle estimate ${central.round()} months',
            style: const TextStyle(color: AppColors.inkSoft)),
        if (lifespan['summary'] case final String summary) ...[
          const SizedBox(height: 10),
          Text(summary),
        ],
        const SizedBox(height: 4),
        ExpansionTile(
          tilePadding: EdgeInsets.zero,
          title: const Text('How this was estimated',
              style: TextStyle(fontSize: 14)),
          expandedCrossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (final factor in factors)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(child: Text(factor['factor'] as String? ?? '')),
                    const SizedBox(width: 8),
                    Text(
                      _months(factor['effect_months'] as num?,
                          first: factor == factors.first),
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ],
                ),
              ),
            for (final source in {
              for (final f in factors) sourceLabel(f['source'] as String?)
            }.where((s) => s.isNotEmpty))
              Text('Source: $source',
                  style:
                      const TextStyle(fontSize: 12, color: AppColors.inkSoft)),
            if (lifespan['explanation'] case final String explanation)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(explanation,
                    style: const TextStyle(
                        fontSize: 12.5, color: AppColors.inkSoft)),
              ),
          ],
        ),
      ],
    );
  }

  static String _months(num? value, {bool first = false}) {
    if (value == null) return '';
    final months = value.round();
    if (first) return '$months mo';
    return months > 0 ? '+$months mo' : '$months mo';
  }
}

class _AdviceCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final String providedBy;
  final List<Widget> children;
  const _AdviceCard({
    required this.title,
    required this.icon,
    required this.providedBy,
    required this.children,
  });

  @override
  Widget build(BuildContext context) => Container(
        width: double.infinity,
        margin: const EdgeInsets.only(bottom: 14),
        padding: const EdgeInsets.fromLTRB(18, 16, 18, 14),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: AppColors.border),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 18, color: AppColors.coral),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(title,
                      style: const TextStyle(
                          fontWeight: FontWeight.w800, fontSize: 15.5)),
                ),
              ],
            ),
            const SizedBox(height: 10),
            ...children,
            const SizedBox(height: 12),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Padding(
                  padding: EdgeInsets.only(top: 1),
                  child: Icon(Icons.memory_rounded,
                      size: 14, color: AppColors.inkSoft),
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text('Provided by our $providedBy',
                      style: const TextStyle(
                          fontSize: 12, color: AppColors.inkSoft)),
                ),
              ],
            ),
          ],
        ),
      );
}

class _TipRow extends StatelessWidget {
  final Widget leading;
  final String text;
  const _TipRow({required this.leading, required this.text});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(width: 24, child: Center(child: leading)),
            const SizedBox(width: 10),
            Expanded(
              child: Text(text,
                  style: const TextStyle(fontSize: 14.5, height: 1.35)),
            ),
          ],
        ),
      );
}

class _NumberBadge extends StatelessWidget {
  final int number;
  const _NumberBadge(this.number);

  @override
  Widget build(BuildContext context) => Container(
        width: 22,
        height: 22,
        alignment: Alignment.center,
        decoration: const BoxDecoration(
            color: AppColors.coral, shape: BoxShape.circle),
        child: Text('$number',
            style: const TextStyle(
                color: AppColors.surface,
                fontSize: 12,
                fontWeight: FontWeight.w800)),
      );
}

class _Chip extends StatelessWidget {
  final IconData icon;
  final String label;
  const _Chip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: AppColors.ink.withValues(alpha: 0.06),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: AppColors.ink),
            const SizedBox(width: 5),
            Flexible(
              child: Text(label,
                  style: const TextStyle(
                      fontSize: 12, fontWeight: FontWeight.w600)),
            ),
          ],
        ),
      );
}
