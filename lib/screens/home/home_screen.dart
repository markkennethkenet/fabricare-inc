import 'package:flutter/material.dart';

import '../../data/mock_data.dart';
import '../../models/scan_result.dart';
import '../../services/auth_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/fabric_thumb.dart';
import '../history/garment_profile_screen.dart';
import '../tips/tips_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({
    super.key,
    required this.onScan,
    required this.onHistory,
    required this.onTips,
    required this.onProfile,
  });

  final VoidCallback onScan;
  final VoidCallback onHistory;
  final VoidCallback onTips;
  final VoidCallback onProfile;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  static final _featuredTips = AppData.tips
      .where((tip) => tip.fabricType == 'General')
      .toList(growable: false);
  int _tipIndex = 0;

  static String _greeting() {
    final hour = DateTime.now().hour;
    return hour < 12
        ? 'Good morning'
        : hour < 18
            ? 'Good afternoon'
            : 'Good evening';
  }

  @override
  Widget build(BuildContext context) {
    final data = AppData.instance;
    final styles = Theme.of(context).textTheme;
    final fullName = AuthService.instance.currentUser?.fullName.trim() ?? '';
    final firstName = fullName.split(RegExp(r'\s+')).first;
    final tip = _featuredTips[_tipIndex];

    return ListenableBuilder(
      listenable: data,
      builder: (context, _) {
        final history = data.history;
        final count = history.length;
        return SafeArea(
          bottom: false,
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 20),
            children: [
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text.rich(
                          TextSpan(
                            style: TextStyle(
                                fontSize: 22, fontWeight: FontWeight.w800),
                            children: [
                              TextSpan(text: 'Fabri'),
                              TextSpan(
                                text: 'Care',
                                style: TextStyle(color: AppColors.coral),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          firstName.isEmpty
                              ? _greeting()
                              : '${_greeting()}, $firstName',
                          style: styles.bodyMedium
                              ?.copyWith(color: AppColors.inkSoft),
                        ),
                      ],
                    ),
                  ),
                  IconButton.filledTonal(
                    tooltip: 'Open profile',
                    onPressed: widget.onProfile,
                    style: IconButton.styleFrom(
                      backgroundColor: AppColors.surfaceMuted,
                      foregroundColor: AppColors.ink,
                    ),
                    icon: AuthService.instance.photoUrl == null
                        ? const Icon(Icons.person_outline_rounded)
                        : CircleAvatar(
                            radius: 14,
                            backgroundColor: AppColors.surfaceMuted,
                            foregroundImage:
                                NetworkImage(AuthService.instance.photoUrl!),
                            onForegroundImageError: (_, __) {},
                            child: const Icon(Icons.person_outline_rounded,
                                size: 18),
                          ),
                  ),
                ],
              ),
              const SizedBox(height: 18),
              _ScanHero(onScan: widget.onScan),
              const SizedBox(height: 12),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: _HomeShortcut(
                      icon: Icons.history_rounded,
                      tint: const Color(0xFFEFE2C7),
                      title: 'History',
                      subtitle:
                          count == 1 ? '1 saved scan' : '$count saved scans',
                      onTap: widget.onHistory,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: _HomeShortcut(
                      icon: Icons.menu_book_outlined,
                      tint: const Color(0xFFE5EBD9),
                      title: 'Care guide',
                      subtitle: 'Browse fabric tips',
                      onTap: widget.onTips,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              _FeaturedTip(
                tip: tip,
                onOpen: () => showFabricTip(context, tip),
                onNext: () => setState(() {
                  _tipIndex = (_tipIndex + 1) % _featuredTips.length;
                }),
              ),
              if (history.isNotEmpty) ...[
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: Text('Recent scans', style: styles.titleLarge),
                    ),
                    TextButton(
                      onPressed: widget.onHistory,
                      child: const Text('View all'),
                    ),
                  ],
                ),
                for (final scan in history.take(2))
                  Padding(
                    padding: const EdgeInsets.only(bottom: 9),
                    child: _RecentScan(scan: scan),
                  ),
              ],
            ],
          ),
        );
      },
    );
  }
}

class _ScanHero extends StatelessWidget {
  const _ScanHero({required this.onScan});

  final VoidCallback onScan;

  @override
  Widget build(BuildContext context) {
    final styles = Theme.of(context).textTheme;
    return Material(
      color: const Color(0xFF344B3B),
      borderRadius: BorderRadius.circular(24),
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 19, 20, 17),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    'Scan a garment',
                    style: styles.headlineSmall?.copyWith(
                      color: AppColors.surface,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                const CircleAvatar(
                  radius: 22,
                  backgroundColor: Color(0xFF4D6753),
                  foregroundColor: AppColors.surface,
                  child: Icon(Icons.center_focus_strong_rounded),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Fabric, condition and care advice in one photo.',
              style: styles.bodyMedium?.copyWith(
                color: const Color(0xFFE7EDDC),
                height: 1.4,
              ),
            ),
            const SizedBox(height: 14),
            FilledButton.icon(
              onPressed: onScan,
              style: FilledButton.styleFrom(
                backgroundColor: AppColors.surface,
                foregroundColor: AppColors.ink,
              ),
              icon: const Icon(Icons.camera_alt_outlined, size: 19),
              label: const Text('Start scan'),
            ),
          ],
        ),
      ),
    );
  }
}

class _HomeShortcut extends StatelessWidget {
  const _HomeShortcut({
    required this.icon,
    required this.tint,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final Color tint;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final styles = Theme.of(context).textTheme;
    return _HomeCard(
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            radius: 17,
            backgroundColor: tint,
            foregroundColor: AppColors.ink,
            child: Icon(icon, size: 17),
          ),
          const SizedBox(height: 8),
          Text(title, style: styles.titleSmall),
          const SizedBox(height: 2),
          Text(subtitle,
              style: styles.bodySmall?.copyWith(color: AppColors.inkSoft)),
        ],
      ),
    );
  }
}

class _FeaturedTip extends StatelessWidget {
  const _FeaturedTip({
    required this.tip,
    required this.onOpen,
    required this.onNext,
  });

  final FabricTip tip;
  final VoidCallback onOpen;
  final VoidCallback onNext;

  @override
  Widget build(BuildContext context) {
    final styles = Theme.of(context).textTheme;
    return _HomeCard(
      color: const Color(0xFFF2E6CB),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.lightbulb_outline_rounded, size: 19),
              const SizedBox(width: 8),
              Expanded(child: Text('Care tip', style: styles.labelLarge)),
              IconButton(
                tooltip: 'Next care tip',
                onPressed: onNext,
                icon: const Icon(Icons.next_plan_outlined),
              ),
            ],
          ),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 180),
            layoutBuilder: (child, previous) => Stack(
              alignment: Alignment.topLeft,
              children: [...previous, if (child != null) child],
            ),
            child: Column(
              key: ValueKey(tip.title),
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(tip.title, style: styles.titleMedium),
                const SizedBox(height: 4),
                Text(tip.points.first, style: styles.bodyMedium),
              ],
            ),
          ),
          TextButton(
            onPressed: onOpen,
            style: TextButton.styleFrom(padding: EdgeInsets.zero),
            child: const Text('Read full tip'),
          ),
        ],
      ),
    );
  }
}

class _RecentScan extends StatelessWidget {
  const _RecentScan({required this.scan});

  final ScanResult scan;

  @override
  Widget build(BuildContext context) {
    final styles = Theme.of(context).textTheme;
    final subtitle = [
      scan.fabricLabel,
      if (scan.healthScore case final int score) 'Health $score',
      MaterialLocalizations.of(context).formatShortMonthDay(scan.dateScanned),
    ].join(' · ');
    return _HomeCard(
      onTap: () => Navigator.of(context).push<void>(
        MaterialPageRoute(builder: (_) => GarmentProfileScreen(result: scan)),
      ),
      child: Row(
        children: [
          FabricThumb(imagePath: scan.imagePath, fabricType: scan.fabricType),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(scan.garmentName, style: styles.titleSmall),
                Text(subtitle,
                    style:
                        styles.bodySmall?.copyWith(color: AppColors.inkSoft)),
              ],
            ),
          ),
          const Icon(Icons.chevron_right_rounded, color: AppColors.inkSoft),
        ],
      ),
    );
  }
}

class _HomeCard extends StatelessWidget {
  const _HomeCard({
    required this.child,
    this.color = AppColors.surface,
    this.onTap,
  });

  final Widget child;
  final Color color;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final content = Padding(padding: const EdgeInsets.all(13), child: child);
    return Material(
      color: color,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(17),
        side: const BorderSide(color: AppColors.border),
      ),
      clipBehavior: Clip.antiAlias,
      child: onTap == null ? content : InkWell(onTap: onTap, child: content),
    );
  }
}
