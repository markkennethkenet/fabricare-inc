import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';
import '../history/history_screen.dart';
import '../profile/profile_screen.dart';
import '../scan/scan_screen.dart';
import '../tips/tips_screen.dart';
import 'home_screen.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  static const scanIndex = 1;
  int _index = 0;

  void _selectTab(int index) {
    FocusScope.of(context).unfocus();
    setState(() => _index = index);
  }

  // Page indexes stay the same; only the bar order changes so Scan sits in
  // the middle: Home · History · [Scan] · Tips · Profile.
  static const _left = [
    (
      index: 0,
      icon: Icons.home_outlined,
      activeIcon: Icons.home_rounded,
      label: 'Home'
    ),
    (
      index: 2,
      icon: Icons.history_rounded,
      activeIcon: Icons.history_rounded,
      label: 'History'
    ),
  ];
  static const _right = [
    (
      index: 3,
      icon: Icons.menu_book_outlined,
      activeIcon: Icons.menu_book_rounded,
      label: 'Tips'
    ),
    (
      index: 4,
      icon: Icons.person_outline_rounded,
      activeIcon: Icons.person_rounded,
      label: 'Profile'
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _index, children: [
        HomeScreen(
          onScan: () => _selectTab(scanIndex),
          onHistory: () => _selectTab(2),
          onTips: () => _selectTab(3),
          onProfile: () => _selectTab(4),
        ),
        const ScanScreen(),
        HistoryScreen(onScan: () => _selectTab(scanIndex)),
        const TipsScreen(),
        ProfileScreen(onHistory: () => _selectTab(2)),
      ]),
      bottomNavigationBar: _NavBar(
        index: _index,
        left: _left,
        right: _right,
        onSelect: _selectTab,
      ),
    );
  }
}

typedef _NavItem = ({
  int index,
  IconData icon,
  IconData activeIcon,
  String label
});

class _NavBar extends StatelessWidget {
  final int index;
  final List<_NavItem> left;
  final List<_NavItem> right;
  final ValueChanged<int> onSelect;

  const _NavBar({
    required this.index,
    required this.left,
    required this.right,
    required this.onSelect,
  });

  static const _barHeight = 68.0;
  // The Scan circle (64 + a 10 px ring) is centred on the bar's top edge so
  // its label stays visible underneath.
  static const _lift = 37.0;
  static const _scanSize = 64.0;

  @override
  Widget build(BuildContext context) {
    final scanSelected = index == _HomeShellState.scanIndex;
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 0, 14, 10),
        child: SizedBox(
          height: _barHeight + _lift,
          child: Stack(
            clipBehavior: Clip.none,
            alignment: Alignment.topCenter,
            children: [
              Positioned.fill(
                top: _lift,
                child: Container(
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: AppColors.border),
                    boxShadow: [
                      BoxShadow(
                        color: AppColors.ink.withValues(alpha: 0.06),
                        blurRadius: 18,
                        offset: const Offset(0, 6),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      for (final item in left)
                        _NavTab(
                            item: item,
                            selected: item.index == index,
                            onTap: onSelect),
                      Expanded(
                        child: Semantics(
                          excludeSemantics: true,
                          child: GestureDetector(
                            behavior: HitTestBehavior.opaque,
                            onTap: () => onSelect(_HomeShellState.scanIndex),
                            child: Align(
                              alignment: Alignment.bottomCenter,
                              child: Padding(
                                padding: const EdgeInsets.only(bottom: 9),
                                child: Text(
                                  'Scan',
                                  style: TextStyle(
                                    fontSize: 11.5,
                                    fontWeight: FontWeight.w800,
                                    color: scanSelected
                                        ? AppColors.coral
                                        : AppColors.ink,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      for (final item in right)
                        _NavTab(
                            item: item,
                            selected: item.index == index,
                            onTap: onSelect),
                    ],
                  ),
                ),
              ),
              Positioned(
                top: 0,
                child: _ScanButton(
                  size: _scanSize,
                  selected: scanSelected,
                  onTap: () => onSelect(_HomeShellState.scanIndex),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NavTab extends StatelessWidget {
  final _NavItem item;
  final bool selected;
  final ValueChanged<int> onTap;

  const _NavTab(
      {required this.item, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final color = selected ? AppColors.coral : AppColors.inkSoft;
    return Expanded(
      child: Semantics(
        button: true,
        selected: selected,
        label: item.label,
        excludeSemantics: true,
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: () => onTap(item.index),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(selected ? item.activeIcon : item.icon,
                  color: color, size: 24),
              const SizedBox(height: 3),
              Text(
                item.label,
                style: TextStyle(
                  fontSize: 10.5,
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                  color: color,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// The large raised Scan button in the middle of the bar.
class _ScanButton extends StatelessWidget {
  final double size;
  final bool selected;
  final VoidCallback onTap;

  const _ScanButton(
      {required this.size, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      selected: selected,
      label: 'Scan a garment',
      excludeSemantics: true,
      child: Tooltip(
        message: 'Scan a garment',
        child: Container(
          width: size + 10,
          height: size + 10,
          decoration: const BoxDecoration(
            shape: BoxShape.circle,
            color: AppColors.background,
          ),
          alignment: Alignment.center,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: size,
            height: size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: selected
                    ? const [Color(0xFFB8401F), Color(0xFF8F2F15)]
                    : const [Color(0xFFE06A45), AppColors.coral],
              ),
              boxShadow: [
                BoxShadow(
                  color:
                      AppColors.coral.withValues(alpha: selected ? 0.45 : 0.35),
                  blurRadius: selected ? 20 : 14,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Material(
              type: MaterialType.transparency,
              shape: const CircleBorder(),
              clipBehavior: Clip.antiAlias,
              child: InkWell(
                onTap: onTap,
                child: const Icon(
                  Icons.center_focus_strong_rounded,
                  color: Colors.white,
                  size: 34,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
