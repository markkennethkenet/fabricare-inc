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
  int _index = 0;

  void _selectTab(int index) {
    FocusScope.of(context).unfocus();
    setState(() => _index = index);
  }

  final _items = const [
    (icon: Icons.home_outlined, activeIcon: Icons.home_rounded, label: 'Home'),
    (
      icon: Icons.camera_alt_outlined,
      activeIcon: Icons.camera_alt_rounded,
      label: 'Scan'
    ),
    (
      icon: Icons.history_rounded,
      activeIcon: Icons.history_rounded,
      label: 'History'
    ),
    (
      icon: Icons.menu_book_outlined,
      activeIcon: Icons.menu_book_rounded,
      label: 'Tips'
    ),
    (
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
          onScan: () => _selectTab(1),
          onHistory: () => _selectTab(2),
          onTips: () => _selectTab(3),
          onProfile: () => _selectTab(4),
        ),
        const ScanScreen(),
        HistoryScreen(onScan: () => _selectTab(1)),
        const TipsScreen(),
        ProfileScreen(onHistory: () => _selectTab(2)),
      ]),
      bottomNavigationBar: SafeArea(
        child: Container(
          margin: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: List.generate(_items.length, (i) {
              final selected = i == _index;
              final item = _items[i];
              return InkWell(
                borderRadius: BorderRadius.circular(16),
                onTap: () => _selectTab(i),
                child: Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        selected ? item.activeIcon : item.icon,
                        color: selected ? AppColors.coral : AppColors.inkSoft,
                        size: 24,
                      ),
                      const SizedBox(height: 3),
                      Text(
                        item.label,
                        style: TextStyle(
                          fontSize: 10.5,
                          fontWeight:
                              selected ? FontWeight.w700 : FontWeight.w500,
                          color: selected ? AppColors.coral : AppColors.inkSoft,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }),
          ),
        ),
      ),
    );
  }
}
