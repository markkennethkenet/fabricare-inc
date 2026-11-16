import 'package:flutter/material.dart';

import '../../data/mock_data.dart';
import '../../theme/app_theme.dart';

void showFabricTip(BuildContext context, FabricTip tip) {
  FocusScope.of(context).unfocus();
  showModalBottomSheet<void>(
    context: context,
    useRootNavigator: true,
    useSafeArea: true,
    isScrollControlled: true,
    showDragHandle: true,
    backgroundColor: AppColors.surface,
    constraints: const BoxConstraints(maxWidth: 600),
    builder: (context) => SafeArea(top: false, child: _TipDetails(tip: tip)),
  );
}

enum _Topic {
  all('All tips'),
  fabrics('By fabric'),
  laundry('Laundry care');

  const _Topic(this.label);
  final String label;
}

class TipsScreen extends StatefulWidget {
  const TipsScreen({super.key});

  @override
  State<TipsScreen> createState() => _TipsScreenState();
}

class _TipsScreenState extends State<TipsScreen> {
  final _search = TextEditingController();
  _Topic _topic = _Topic.all;

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final query = _search.text.trim().toLowerCase();
    final terms =
        query.isEmpty ? const <String>[] : query.split(RegExp(r'\s+'));
    final matching = AppData.tips.where((tip) {
      final general = tip.fabricType == 'General';
      if (_topic == _Topic.fabrics && general ||
          _topic == _Topic.laundry && !general) {
        return false;
      }
      final content = '${tip.title} ${tip.fabricType} ${tip.points.join(' ')}'
          .toLowerCase();
      return terms.every(content.contains);
    }).toList();
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Fabric Care Guide')),
      body: SafeArea(
        bottom: false,
        child: ListView(
          keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 28),
          children: [
            if (query.isEmpty) ...[
              const _CareIntro(),
              const SizedBox(height: 20),
            ],
            TextField(
              controller: _search,
              textInputAction: TextInputAction.search,
              onChanged: (_) => setState(() {}),
              decoration: InputDecoration(
                labelText: 'Search care tips',
                hintText: 'Try linen or stains',
                prefixIcon: const Icon(Icons.search_rounded),
                suffixIcon: _search.text.isEmpty
                    ? null
                    : IconButton(
                        tooltip: 'Clear search',
                        onPressed: () => setState(_search.clear),
                        icon: const Icon(Icons.close_rounded),
                      ),
              ),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              children: [
                for (final topic in _Topic.values)
                  ChoiceChip(
                    label: Text(topic.label),
                    selected: _topic == topic,
                    selectedColor: AppColors.surfaceMuted,
                    backgroundColor: AppColors.surface,
                    onSelected: (_) {
                      FocusScope.of(context).unfocus();
                      setState(() => _topic = topic);
                    },
                  ),
              ],
            ),
            const SizedBox(height: 18),
            Text(
              query.isNotEmpty
                  ? 'Search results'
                  : switch (_topic) {
                      _Topic.all => 'Browse guides',
                      _Topic.fabrics => 'Fabric guides',
                      _Topic.laundry => 'Laundry tips',
                    },
              style: theme.textTheme.titleLarge
                  ?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 12),
            if (matching.isEmpty)
              _EmptyTips(
                onReset: () => setState(() {
                  _search.clear();
                  _topic = _Topic.all;
                }),
              )
            else
              for (final tip in matching)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: _TipCard(
                      tip: tip, onTap: () => showFabricTip(context, tip)),
                ),
          ],
        ),
      ),
    );
  }
}

class _CareIntro extends StatelessWidget {
  const _CareIntro();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFF344B3B),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Care for what\nyou wear',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          color: AppColors.surface,
                          fontWeight: FontWeight.w800,
                          height: 1.1,
                        ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Start with the garment label, then find tips for your fabric or laundry problem.',
                    style: TextStyle(color: Color(0xFFE7EDDC), height: 1.4),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 16),
            const DecoratedBox(
              decoration: BoxDecoration(
                color: Color(0xFF4D6753),
                shape: BoxShape.circle,
              ),
              child: Padding(
                padding: EdgeInsets.all(14),
                child: Icon(Icons.local_laundry_service_outlined,
                    size: 30, color: AppColors.surface),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TipCard extends StatelessWidget {
  const _TipCard({required this.tip, required this.onTap});

  final FabricTip tip;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final general = tip.fabricType == 'General';
    final shape = RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(18),
      side: const BorderSide(color: AppColors.border),
    );
    return Material(
      color: AppColors.surface,
      shape: shape,
      clipBehavior: Clip.antiAlias,
      child: Semantics(
        button: true,
        child: InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: general
                        ? AppColors.mustard.withValues(alpha: 0.19)
                        : AppColors.sage.withValues(alpha: 0.18),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(tip.icon, color: AppColors.ink, size: 22),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        general ? 'Laundry care' : tip.fabricType,
                        style: theme.textTheme.labelMedium
                            ?.copyWith(color: AppColors.inkSoft),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        tip.title,
                        style: theme.textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        tip.points.first,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodyMedium
                            ?.copyWith(color: AppColors.inkSoft),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                const Icon(Icons.chevron_right_rounded,
                    color: AppColors.inkSoft),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _TipDetails extends StatelessWidget {
  const _TipDetails({required this.tip});

  final FabricTip tip;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListView(
      shrinkWrap: true,
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
      children: [
        Row(
          children: [
            Icon(tip.icon, color: AppColors.ink),
            const SizedBox(width: 12),
            Expanded(
              child: Semantics(
                header: true,
                child: Text(tip.title, style: theme.textTheme.headlineSmall),
              ),
            ),
            IconButton(
              tooltip: 'Close tip',
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.close_rounded),
            ),
          ],
        ),
        const SizedBox(height: 14),
        for (final point in tip.points)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.check_circle_outline,
                    color: AppColors.sage, size: 21),
                const SizedBox(width: 12),
                Expanded(child: Text(point, style: theme.textTheme.bodyLarge)),
              ],
            ),
          ),
        const Divider(),
        const Text('Follow the care instructions on your garment label.'),
      ],
    );
  }
}

class _EmptyTips extends StatelessWidget {
  const _EmptyTips({required this.onReset});

  final VoidCallback onReset;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const Icon(Icons.search_off_rounded, color: AppColors.inkSoft),
            const SizedBox(height: 8),
            Text('No matching tips',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            const Text(
              'Try another fabric or care topic.',
              textAlign: TextAlign.center,
            ),
            TextButton(onPressed: onReset, child: const Text('Show all tips')),
          ],
        ),
      ),
    );
  }
}
