import 'package:flutter/material.dart';

import '../../data/mock_data.dart';
import '../../models/scan_result.dart';
import '../../theme/app_theme.dart';
import '../../widgets/fabric_thumb.dart';
import '../../widgets/section_card.dart';
import '../scan/scan_screen.dart';
import 'garment_profile_screen.dart';

enum _HistoryFilter {
  all('All scans'),
  needsReview('Needs review'),
  reviewed('Reviewed');

  const _HistoryFilter(this.label);
  final String label;

  static _HistoryFilter? reviewState(ScanResult scan) {
    final findings = scan.damageDetection.allItems;
    if (findings.isEmpty) return null;
    final decisions = scan.damageReview;
    return findings.every((finding) => decisions.containsKey(finding.id))
        ? reviewed
        : needsReview;
  }
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key, this.onScan});

  final VoidCallback? onScan;

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _search = TextEditingController();
  _HistoryFilter _filter = _HistoryFilter.all;

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  void _startScan() {
    FocusScope.of(context).unfocus();
    if (widget.onScan case final onScan?) {
      onScan();
    } else {
      Navigator.of(context).push(
        MaterialPageRoute<void>(builder: (_) => const ScanScreen()),
      );
    }
  }

  void _resetFilters() {
    setState(() {
      _search.clear();
      _filter = _HistoryFilter.all;
    });
  }

  @override
  Widget build(BuildContext context) {
    final data = AppData.instance;
    final theme = Theme.of(context);
    final localizations = MaterialLocalizations.of(context);
    return ListenableBuilder(
      listenable: data,
      builder: (context, _) {
        final history = data.history;
        final query = _search.text.trim().toLowerCase();
        final entries = [
          for (final scan in history)
            (
              scan: scan,
              status: _HistoryFilter.reviewState(scan),
              date: localizations.formatMediumDate(scan.dateScanned.toLocal()),
            ),
        ];
        final needsReview = entries
            .where((entry) => entry.status == _HistoryFilter.needsReview)
            .length;
        final visible = entries.where((entry) {
          if (_filter != _HistoryFilter.all && entry.status != _filter) {
            return false;
          }
          final scan = entry.scan;
          final searchable =
              '${scan.garmentName} ${scan.fabricLabel} ${scan.healthRating} ${entry.date}'
                  .toLowerCase();
          return searchable.contains(query);
        }).toList();

        return Scaffold(
          appBar: AppBar(
            title: const Text('Scan History'),
            actions: [
              if (history.isNotEmpty)
                PopupMenuButton<bool>(
                  tooltip: 'History options',
                  onSelected: (_) => _confirmClear(context),
                  itemBuilder: (_) => const [
                    PopupMenuItem(value: true, child: Text('Clear history')),
                  ],
                ),
            ],
          ),
          body: SafeArea(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
              keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
              children: [
                if (history.isEmpty)
                  _HistoryMessage(
                    icon: data.storageError == null
                        ? Icons.checkroom_outlined
                        : Icons.error_outline_rounded,
                    title: data.storageError == null
                        ? 'Your wardrobe starts here'
                        : 'History is unavailable',
                    message: data.storageError == null
                        ? 'Scan a garment to save its photos, condition, and care recommendations here.'
                        : 'Your saved scans could not be opened.',
                    action: data.storageError == null
                        ? FilledButton.icon(
                            onPressed: _startScan,
                            icon: const Icon(Icons.camera_alt_outlined),
                            label: const Text('Start a scan'),
                          )
                        : null,
                  )
                else ...[
                  _HistoryOverview(
                    total: history.length,
                    needsReview: needsReview,
                    onNeedsReview: needsReview == 0
                        ? null
                        : () => setState(() {
                              _search.clear();
                              _filter = _HistoryFilter.needsReview;
                            }),
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: _search,
                    textInputAction: TextInputAction.search,
                    onChanged: (_) => setState(() {}),
                    decoration: InputDecoration(
                      labelText: 'Search scans',
                      hintText: 'Garment, fabric, or condition',
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
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    children: [
                      for (final filter in _HistoryFilter.values)
                        ChoiceChip(
                          label: Text(filter.label),
                          selected: _filter == filter,
                          selectedColor: AppColors.surfaceMuted,
                          backgroundColor: AppColors.surface,
                          onSelected: (_) {
                            FocusScope.of(context).unfocus();
                            setState(() => _filter = filter);
                          },
                        ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  if (visible.isEmpty)
                    _HistoryMessage(
                      icon: Icons.search_off_rounded,
                      title: 'No matching scans',
                      message: 'Try another search or show all saved scans.',
                      action: TextButton(
                        onPressed: _resetFilters,
                        child: const Text('Show all scans'),
                      ),
                    )
                  else ...[
                    Text(
                      query.isEmpty && _filter == _HistoryFilter.all
                          ? 'Recent scans'
                          : '${visible.length} matching ${visible.length == 1 ? 'scan' : 'scans'}',
                      style: theme.textTheme.titleLarge
                          ?.copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 10),
                    for (final entry in visible)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: _ScanCard(
                          scan: entry.scan,
                          date: entry.date,
                          status: entry.status,
                          onOpen: () {
                            FocusScope.of(context).unfocus();
                            Navigator.of(context).push(
                              MaterialPageRoute<void>(
                                builder: (_) =>
                                    GarmentProfileScreen(result: entry.scan),
                              ),
                            );
                          },
                          onDelete: () => _delete(context, entry.scan.id),
                        ),
                      ),
                  ],
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  void _confirmClear(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('Clear scan history?'),
        content: const Text(
            'This will remove all saved scans for your account on this device.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              try {
                await AppData.instance.clearHistory();
              } catch (_) {
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                      content: Text('Could not clear history. Try again.')));
                }
              }
            },
            child:
                const Text('Clear', style: TextStyle(color: AppColors.danger)),
          ),
        ],
      ),
    );
  }

  Future<void> _delete(BuildContext context, String id) async {
    final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
                title: const Text('Delete this scan?'),
                content: const Text(
                    'The saved scan and its photos will be removed.'),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(ctx, false),
                      child: const Text('Cancel')),
                  TextButton(
                      onPressed: () => Navigator.pop(ctx, true),
                      child: const Text('Delete'))
                ]));
    if (confirmed != true) return;
    try {
      await AppData.instance.deleteScan(id);
    } catch (_) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not delete scan. Try again.')));
      }
    }
  }
}

class _HistoryOverview extends StatelessWidget {
  const _HistoryOverview({
    required this.total,
    required this.needsReview,
    this.onNeedsReview,
  });

  final int total;
  final int needsReview;
  final VoidCallback? onNeedsReview;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: const Color(0xFF344B3B),
      borderRadius: BorderRadius.circular(22),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 15, 18, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Your wardrobe',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: AppColors.surface,
                    fontWeight: FontWeight.w800,
                  ),
            ),
            const SizedBox(height: 4),
            Text(
              '$total saved ${total == 1 ? 'scan' : 'scans'}',
              style: const TextStyle(color: Color(0xFFE7EDDC)),
            ),
            if (onNeedsReview != null)
              TextButton.icon(
                onPressed: onNeedsReview,
                style: TextButton.styleFrom(
                  foregroundColor: AppColors.surface,
                  padding: EdgeInsets.zero,
                  alignment: Alignment.centerLeft,
                ),
                icon: const Icon(Icons.fact_check_outlined, size: 19),
                label: Text(
                    '$needsReview ${needsReview == 1 ? 'scan needs' : 'scans need'} review'),
              ),
          ],
        ),
      ),
    );
  }
}

class _ScanCard extends StatelessWidget {
  const _ScanCard({
    required this.scan,
    required this.date,
    required this.status,
    required this.onOpen,
    required this.onDelete,
  });

  final ScanResult scan;
  final String date;
  final _HistoryFilter? status;
  final VoidCallback onOpen;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final styles = Theme.of(context).textTheme;
    return Material(
      color: AppColors.surface,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: const BorderSide(color: AppColors.border),
      ),
      child: InkWell(
        onTap: onOpen,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(14, 14, 4, 14),
          child: Row(
            children: [
              FabricThumb(
                  image: scan.image, fabricType: scan.fabricType, size: 62),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(scan.garmentName,
                        style: styles.titleMedium
                            ?.copyWith(fontWeight: FontWeight.w700)),
                    Text('${scan.fabricLabel} · $date',
                        style: styles.bodySmall
                            ?.copyWith(color: AppColors.inkSoft)),
                    const SizedBox(height: 7),
                    Wrap(
                      spacing: 8,
                      runSpacing: 5,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        if (scan.hasHealthAssessment)
                          SeverityBadge(label: scan.healthRating),
                        if (status != null)
                          Text(status!.label,
                              style: styles.labelMedium
                                  ?.copyWith(color: AppColors.inkSoft)),
                      ],
                    ),
                  ],
                ),
              ),
              PopupMenuButton<bool>(
                tooltip: 'Scan options',
                onSelected: (_) => onDelete(),
                itemBuilder: (_) => const [
                  PopupMenuItem(value: true, child: Text('Delete scan')),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _HistoryMessage extends StatelessWidget {
  const _HistoryMessage({
    required this.icon,
    required this.title,
    required this.message,
    this.action,
  });

  final IconData icon;
  final String title;
  final String message;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 28),
        child: Column(
          children: [
            CircleAvatar(
              radius: 26,
              backgroundColor: AppColors.surfaceMuted,
              foregroundColor: AppColors.ink,
              child: Icon(icon),
            ),
            const SizedBox(height: 14),
            Text(title,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(message,
                textAlign: TextAlign.center,
                style: const TextStyle(color: AppColors.inkSoft)),
            if (action != null) ...[
              const SizedBox(height: 14),
              action!,
            ],
          ],
        ),
      ),
    );
  }
}
