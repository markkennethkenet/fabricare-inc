import 'package:flutter/material.dart';

import '../../theme/app_theme.dart';

/// What the scanner's three fabric models do, in plain language.
///
/// Accuracy is each model's result on the same 353 held-out photos of the
/// FabriCare V3 dataset (see docs/SIX_CLASS_DATASET.md and model_runtime.py).
class FabricModelInfo {
  final String id;
  final String name;
  final String tagline;
  final String description;
  final String bestFor;
  final String score;
  final double testAccuracy;
  final bool recommended;
  final IconData icon;

  const FabricModelInfo({
    required this.id,
    required this.name,
    required this.tagline,
    required this.description,
    required this.bestFor,
    required this.score,
    required this.testAccuracy,
    required this.icon,
    this.recommended = false,
  });

  static FabricModelInfo byId(String id) => fabricModels
      .firstWhere((model) => model.id == id, orElse: () => fabricModels.first);
}

const fabricModels = [
  FabricModelInfo(
    id: 'cnn',
    name: 'CNN — MobileNetV2',
    tagline: 'Most accurate · best for everyday scans',
    description:
        'A deep-learning model that reads colour, weave and texture together. '
        'It learned from about 3,400 labelled photos of the six fabrics.',
    bestFor:
        'Everyday scans of cotton, denim, linen, polyester, silk and wool.',
    score: 'Shows a confidence score',
    testAccuracy: 81.87,
    icon: Icons.auto_awesome_rounded,
    recommended: true,
  ),
  FabricModelInfo(
    id: 'svm',
    name: 'GLCM + SVM',
    tagline: 'Texture only · useful second opinion',
    description:
        'Measures how the weave pattern repeats across the photo (72 texture '
        'features) and separates fabrics with a support vector machine. '
        'It ignores colour.',
    bestFor: 'Comparing a texture-only answer with the CNN.',
    score: 'No confidence score',
    testAccuracy: 58.92,
    icon: Icons.grid_on_rounded,
  ),
  FabricModelInfo(
    id: 'glcm_dt',
    name: 'GLCM + Decision Tree',
    tagline: 'Easy to explain · lowest accuracy',
    description:
        'Uses the same 72 texture measurements with a tree of simple yes/no '
        'rules, so every decision can be traced step by step.',
    bestFor: 'Learning or explaining how texture rules classify fabric.',
    score: 'Rough vote score, not a true confidence',
    testAccuracy: 50.42,
    icon: Icons.account_tree_outlined,
  ),
];

/// The "Choose Fabric Identification Model" card on the scan page.
class FabricModelCard extends StatelessWidget {
  final String selected;
  final bool enabled;
  final ValueChanged<String> onChanged;

  const FabricModelCard({
    super.key,
    required this.selected,
    required this.onChanged,
    this.enabled = true,
  });

  Future<void> _choose(BuildContext context) async {
    final choice = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      backgroundColor: AppColors.background,
      builder: (_) => FabricModelSheet(selected: selected),
    );
    if (choice != null) onChanged(choice);
  }

  @override
  Widget build(BuildContext context) {
    final model = FabricModelInfo.byId(selected);
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: enabled ? () => _choose(context) : null,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 12, 14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Row(
                children: [
                  Icon(Icons.tune_rounded, size: 18, color: AppColors.coral),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text('Choose Fabric Identification Model',
                        style: TextStyle(
                            fontWeight: FontWeight.w700, fontSize: 15)),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  _ModelIcon(model: model, selected: true),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Wrap(
                          spacing: 6,
                          runSpacing: 4,
                          crossAxisAlignment: WrapCrossAlignment.center,
                          children: [
                            Text(model.name,
                                style: const TextStyle(
                                    fontWeight: FontWeight.w800,
                                    fontSize: 15.5)),
                            if (model.recommended) const RecommendedBadge(),
                          ],
                        ),
                        const SizedBox(height: 3),
                        Text(model.tagline,
                            style: const TextStyle(
                                color: AppColors.inkSoft, fontSize: 12.5)),
                      ],
                    ),
                  ),
                  TextButton(
                    onPressed: enabled ? () => _choose(context) : null,
                    child: const Text('Change'),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              AccuracyBar(accuracy: model.testAccuracy),
              if (!model.recommended) ...[
                const SizedBox(height: 10),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppColors.mustard.withValues(alpha: 0.14),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.info_outline_rounded,
                          size: 18, color: Color(0xFF8A6417)),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Less accurate than the recommended CNN. Use it to compare results.',
                          style: TextStyle(
                              fontSize: 12.5, color: Color(0xFF6B4D10)),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

/// Bottom sheet listing every model with its details.
class FabricModelSheet extends StatelessWidget {
  final String selected;
  const FabricModelSheet({super.key, required this.selected});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ConstrainedBox(
        constraints:
            BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.86),
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text('Choose Fabric Identification Model',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
              const SizedBox(height: 6),
              const Text(
                'Not sure? Keep the Recommended model. It is the most accurate of the three.',
                style: TextStyle(color: AppColors.inkSoft),
              ),
              const SizedBox(height: 16),
              for (final model in fabricModels)
                _ModelOption(
                  model: model,
                  selected: model.id == selected,
                  onTap: () => Navigator.of(context).pop(model.id),
                ),
              const SizedBox(height: 4),
              const Text(
                'Accuracy is measured on 353 held-out photos from the FabriCare dataset. '
                'Real scans depend on light, focus and framing, and no model can confirm '
                'fibre content. Check the care label when it matters.',
                style: TextStyle(color: AppColors.inkSoft, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ModelOption extends StatelessWidget {
  final FabricModelInfo model;
  final bool selected;
  final VoidCallback onTap;

  const _ModelOption(
      {required this.model, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Semantics(
        button: true,
        selected: selected,
        label: '${model.name}${model.recommended ? ', recommended' : ''}',
        child: Material(
          color: AppColors.surface,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
            side: BorderSide(
              color: selected ? AppColors.coral : AppColors.border,
              width: selected ? 2 : 1,
            ),
          ),
          child: InkWell(
            borderRadius: BorderRadius.circular(18),
            onTap: onTap,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _ModelIcon(model: model, selected: selected),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Wrap(
                              spacing: 6,
                              runSpacing: 4,
                              crossAxisAlignment: WrapCrossAlignment.center,
                              children: [
                                Text(model.name,
                                    style: const TextStyle(
                                        fontWeight: FontWeight.w800,
                                        fontSize: 16)),
                                if (model.recommended) const RecommendedBadge(),
                              ],
                            ),
                            const SizedBox(height: 2),
                            Text(model.tagline,
                                style: const TextStyle(
                                    color: AppColors.inkSoft, fontSize: 12.5)),
                          ],
                        ),
                      ),
                      Icon(
                        selected
                            ? Icons.radio_button_checked
                            : Icons.radio_button_off,
                        color: selected ? AppColors.coral : AppColors.inkSoft,
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(model.description, style: const TextStyle(height: 1.35)),
                  const SizedBox(height: 12),
                  AccuracyBar(accuracy: model.testAccuracy),
                  const SizedBox(height: 10),
                  _Detail(
                      icon: Icons.thumb_up_alt_outlined,
                      label: 'Best for',
                      value: model.bestFor),
                  _Detail(
                      icon: Icons.speed_rounded,
                      label: 'Score',
                      value: model.score),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _ModelIcon extends StatelessWidget {
  final FabricModelInfo model;
  final bool selected;
  const _ModelIcon({required this.model, required this.selected});

  @override
  Widget build(BuildContext context) => Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: model.recommended
              ? AppColors.coral.withValues(alpha: 0.12)
              : AppColors.surfaceMuted,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Icon(model.icon,
            color: model.recommended ? AppColors.coral : AppColors.ink,
            size: 24),
      );
}

class _Detail extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  const _Detail({required this.icon, required this.label, required this.value});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 16, color: AppColors.inkSoft),
            const SizedBox(width: 8),
            SizedBox(
              width: 62,
              child: Text(label,
                  style: const TextStyle(
                      color: AppColors.inkSoft, fontSize: 12.5)),
            ),
            Expanded(
              child: Text(value,
                  style: const TextStyle(
                      fontWeight: FontWeight.w600, fontSize: 12.5)),
            ),
          ],
        ),
      );
}

/// "Recommended" pill for the default model.
class RecommendedBadge extends StatelessWidget {
  const RecommendedBadge({super.key});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(
          color: AppColors.sage.withValues(alpha: 0.16),
          borderRadius: BorderRadius.circular(20),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.verified_rounded, size: 13, color: Color(0xFF3F6A44)),
            SizedBox(width: 3),
            Text('Recommended',
                style: TextStyle(
                    color: Color(0xFF3F6A44),
                    fontWeight: FontWeight.w800,
                    fontSize: 11.5)),
          ],
        ),
      );
}

/// Test accuracy as a labelled bar.
class AccuracyBar extends StatelessWidget {
  final double accuracy;
  const AccuracyBar({super.key, required this.accuracy});

  @override
  Widget build(BuildContext context) {
    final color = accuracy >= 75
        ? AppColors.sage
        : accuracy >= 55
            ? AppColors.mustard
            : AppColors.danger;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text('Test accuracy',
                style: TextStyle(color: AppColors.inkSoft, fontSize: 12.5)),
            const Spacer(),
            Text('${accuracy.toStringAsFixed(1)}%',
                style: TextStyle(
                    fontWeight: FontWeight.w800, color: color, fontSize: 13)),
          ],
        ),
        const SizedBox(height: 5),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: accuracy / 100,
            minHeight: 8,
            color: color,
            backgroundColor: AppColors.surfaceMuted,
          ),
        ),
      ],
    );
  }
}
