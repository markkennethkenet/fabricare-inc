import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// AI laundry-mode suggestion and automated care alerts (API `ai_care`).
class AiCareCard extends StatelessWidget {
  final Map<String, dynamic> care;
  const AiCareCard({super.key, required this.care});

  static Color alertColor(String? level) => switch (level) {
        'high' => AppColors.danger,
        'medium' => const Color(0xFF9A6B12),
        _ => AppColors.sage,
      };

  @override
  Widget build(BuildContext context) {
    final confidence = care['confidence'] as num?;
    final reasons = (care['reasons'] as List? ?? const []).whereType<Map>();
    final alerts = (care['alerts'] as List? ?? const []).whereType<Map>().toList();
    final trained = care['status'] == 'model_suggestion';
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.local_laundry_service_outlined, size: 18, color: AppColors.coral),
              SizedBox(width: 8),
              Expanded(
                child: Text('AI laundry suggestion',
                    style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(care['label'] as String? ?? 'Unavailable',
              style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w900, height: 1.15)),
          if (care['detail'] case final String detail) ...[
            const SizedBox(height: 6),
            Text(detail, style: const TextStyle(color: AppColors.inkSoft, height: 1.3)),
          ],
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: [
              _Chip(
                icon: trained ? Icons.auto_awesome_rounded : Icons.rule_rounded,
                label: care['model'] as String? ?? 'Care rules',
              ),
              if (confidence != null)
                _Chip(icon: Icons.speed_rounded, label: '${confidence.round()}% model confidence'),
            ],
          ),
          for (final reason in reasons)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 2),
                    child: Icon(Icons.check_circle_outline_rounded, size: 16, color: AppColors.sage),
                  ),
                  const SizedBox(width: 8),
                  Expanded(child: Text(reason['text'] as String? ?? '')),
                ],
              ),
            ),
          if (care['note'] case final String note) ...[
            const SizedBox(height: 10),
            Text(note, style: const TextStyle(fontSize: 12.5, color: AppColors.inkSoft)),
          ],
          if (alerts.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text('Care alerts', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 14)),
            const SizedBox(height: 6),
            for (final alert in alerts)
              Container(
                margin: const EdgeInsets.only(top: 6),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: alertColor(alert['level'] as String?).withValues(alpha: 0.09),
                  borderRadius: BorderRadius.circular(14),
                  border: Border(
                    left: BorderSide(color: alertColor(alert['level'] as String?), width: 4),
                  ),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      alert['level'] == 'high'
                          ? Icons.report_rounded
                          : alert['level'] == 'medium'
                              ? Icons.warning_amber_rounded
                              : Icons.info_outline_rounded,
                      size: 20,
                      color: alertColor(alert['level'] as String?),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(alert['title'] as String? ?? '',
                              style: const TextStyle(fontWeight: FontWeight.w800)),
                          const SizedBox(height: 2),
                          Text(alert['action'] as String? ?? '', style: const TextStyle(height: 1.3)),
                        ],
                      ),
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

/// Visible-condition health score (API `health_assessment`).
class HealthScoreBadge extends StatelessWidget {
  final Map<String, dynamic> health;
  const HealthScoreBadge({super.key, required this.health});

  @override
  Widget build(BuildContext context) {
    final score = (health['score'] as num).round();
    final grade = health['grade'] as String? ?? '';
    final color = score >= 85
        ? AppColors.sage
        : score >= 70
            ? const Color(0xFF6E8B3D)
            : score >= 50
                ? AppColors.mustard
                : AppColors.danger;
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
                Text('$score', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: color)),
              ],
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Fabric health $score/100',
                    style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15)),
                Text(grade, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
                const Text('Based on the visible findings in your photos.',
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
        const SizedBox(height: 10),
        ExpansionTile(
          tilePadding: EdgeInsets.zero,
          title: const Text('How this was estimated', style: TextStyle(fontSize: 14)),
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
                      _months(factor['effect_months'] as num?, first: factor == factors.first),
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ],
                ),
              ),
            for (final source in {for (final f in factors) sourceLabel(f['source'] as String?)}
                .where((s) => s.isNotEmpty))
              Text('Source: $source', style: const TextStyle(fontSize: 12, color: AppColors.inkSoft)),
            if (lifespan['explanation'] case final String explanation)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(explanation, style: const TextStyle(fontSize: 12.5, color: AppColors.inkSoft)),
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
            Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
          ],
        ),
      );
}
