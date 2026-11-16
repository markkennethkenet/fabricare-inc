import 'package:flutter/material.dart';

import '../../data/mock_data.dart';
import '../../services/auth_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/section_card.dart';
import '../auth/login_screen.dart';
import 'change_password_screen.dart';
import 'edit_profile_screen.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key, required this.onHistory});

  final VoidCallback onHistory;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: AuthService.instance,
      builder: (context, _) {
        final user = AuthService.instance.currentUser;
        if (user == null) {
          return const Scaffold(
              body: Center(child: Text('Sign in to view your profile.')));
        }
        return ListenableBuilder(
          listenable: AppData.instance,
          builder: (context, _) {
            final history = AppData.instance.history;
            final styles = Theme.of(context).textTheme;
            final displayName =
                user.fullName.trim().isEmpty ? 'Your account' : user.fullName;
            return Scaffold(
              appBar: AppBar(title: const Text('Profile')),
              body: SafeArea(
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
                  children: [
                    Container(
                      padding: const EdgeInsets.all(18),
                      decoration: BoxDecoration(
                        color: const Color(0xFF344B3B),
                        borderRadius: BorderRadius.circular(22),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              CircleAvatar(
                                radius: 25,
                                backgroundColor: const Color(0xFF4D6753),
                                foregroundColor: AppColors.surface,
                                child: Text(user.initials,
                                    style: styles.titleLarge
                                        ?.copyWith(color: AppColors.surface)),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  displayName,
                                  style: styles.headlineSmall?.copyWith(
                                    color: AppColors.surface,
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 14),
                          FilledButton.icon(
                            style: FilledButton.styleFrom(
                              backgroundColor: AppColors.surface,
                              foregroundColor: AppColors.ink,
                            ),
                            onPressed: () => Navigator.of(context).push<void>(
                              MaterialPageRoute(
                                  builder: (_) => const EditProfileScreen()),
                            ),
                            icon: const Icon(Icons.edit_outlined, size: 18),
                            label: const Text('Edit profile'),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 14),
                    SectionCard(
                      title: 'Account details',
                      icon: Icons.badge_outlined,
                      child: Column(
                        children: [
                          _AccountField(
                              icon: Icons.email_outlined,
                              label: 'Email',
                              value: user.email),
                          const Divider(height: 18),
                          _AccountField(
                              icon: Icons.phone_outlined,
                              label: 'Phone',
                              value: user.phone),
                          const Divider(height: 18),
                          _AccountField(
                              icon: Icons.location_on_outlined,
                              label: 'Location',
                              value: user.location),
                        ],
                      ),
                    ),
                    _SettingsGroup(
                      title: 'Settings',
                      tiles: [
                        _SettingsTile(
                          icon: Icons.lock_outline_rounded,
                          label: 'Change password',
                          onTap: () => Navigator.of(context).push<void>(
                            MaterialPageRoute(
                                builder: (_) => const ChangePasswordScreen()),
                          ),
                        ),
                        _SettingsTile(
                          icon: Icons.history_rounded,
                          label: 'Scan history',
                          subtitle:
                              '${history.length} saved ${history.length == 1 ? 'scan' : 'scans'}',
                          onTap: onHistory,
                        ),
                        if (history.isNotEmpty)
                          _SettingsTile(
                            icon: Icons.delete_outline_rounded,
                            label: 'Clear scan history',
                            destructive: true,
                            onTap: () => _confirmClear(context),
                          ),
                        _SettingsTile(
                          icon: Icons.info_outline_rounded,
                          label: 'About FabriCare',
                          onTap: () => _showAbout(context),
                        ),
                        _SettingsTile(
                          icon: Icons.logout_rounded,
                          label: 'Sign out',
                          destructive: true,
                          onTap: () => _confirmLogout(context),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  void _confirmClear(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Clear scan history?',
            style: TextStyle(fontWeight: FontWeight.w800)),
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
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                        content: Text('Could not clear history. Try again.')),
                  );
                }
              }
            },
            child: const Text('Clear',
                style: TextStyle(
                    color: AppColors.danger, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }

  void _confirmLogout(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Log out?',
            style: TextStyle(fontWeight: FontWeight.w800)),
        content: const Text('You will be returned to the login screen.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              await AuthService.instance.logout();
              if (!context.mounted) return;
              Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (_) => const LoginScreen()),
                (route) => false,
              );
            },
            child: const Text('Logout',
                style: TextStyle(
                    color: AppColors.danger, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
  }

  void _showAbout(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text.rich(
          TextSpan(children: [
            TextSpan(
                text: 'Fabri',
                style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18)),
            TextSpan(
                text: 'Care',
                style: TextStyle(
                    fontWeight: FontWeight.w800,
                    fontSize: 18,
                    color: AppColors.coral)),
          ]),
        ),
        content: const Text(
          'FabriCare helps you identify fabric types, assess garment condition, and get smart care recommendations to make your clothes last longer.\n\nVersion 1.0.0 · UI Prototype',
          style: TextStyle(height: 1.5),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx), child: const Text('Close')),
        ],
      ),
    );
  }
}

class _AccountField extends StatelessWidget {
  const _AccountField(
      {required this.icon, required this.label, required this.value});

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final styles = Theme.of(context).textTheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: AppColors.inkSoft),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label,
                  style:
                      styles.labelMedium?.copyWith(color: AppColors.inkSoft)),
              const SizedBox(height: 3),
              Text(value.trim().isEmpty ? 'Not added' : value,
                  style: styles.bodyLarge),
            ],
          ),
        ),
      ],
    );
  }
}

class _SettingsGroup extends StatelessWidget {
  final String? title;
  final List<_SettingsTile> tiles;
  const _SettingsGroup({this.title, required this.tiles});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (title != null) ...[
          Padding(
            padding: const EdgeInsets.only(left: 4, bottom: 8),
            child: Text(
              title!,
              style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 12.5,
                  color: AppColors.inkSoft),
            ),
          ),
        ],
        Container(
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
            children: List.generate(tiles.length * 2 - 1, (i) {
              if (i.isOdd)
                return Divider(
                    height: 1,
                    indent: 52,
                    color: AppColors.border.withValues(alpha: 0.7));
              return tiles[i ~/ 2];
            }),
          ),
        ),
      ],
    );
  }
}

class _SettingsTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String? subtitle;
  final VoidCallback onTap;
  final bool destructive;

  const _SettingsTile({
    required this.icon,
    required this.label,
    required this.onTap,
    this.subtitle,
    this.destructive = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = destructive ? AppColors.danger : AppColors.ink;
    final bgColor = destructive
        ? AppColors.danger.withValues(alpha: 0.1)
        : AppColors.surfaceMuted;
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      leading: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
            color: bgColor, borderRadius: BorderRadius.circular(10)),
        child: Icon(icon, color: color, size: 18),
      ),
      title: Text(label,
          style: TextStyle(
              color: color, fontWeight: FontWeight.w600, fontSize: 13.5)),
      subtitle: subtitle != null
          ? Text(subtitle!,
              style: const TextStyle(color: AppColors.inkSoft, fontSize: 11.5))
          : null,
      trailing: const Icon(Icons.chevron_right_rounded,
          color: AppColors.inkSoft, size: 20),
      onTap: onTap,
    );
  }
}
