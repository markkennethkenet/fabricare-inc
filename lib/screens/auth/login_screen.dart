import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';

import '../../services/auth_service.dart';
import '../../services/google_auth.dart';
import '../../theme/app_theme.dart';
import '../../widgets/google_web_button.dart';
import '../home/home_shell.dart';

/// Google sign-in, with email login and sign-up as an alternative. A returning
/// Google user is signed back in without a prompt when Google allows it; new
/// Google users get an account named after their Google profile.
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _google = GoogleAuth.instance;
  StreamSubscription<GoogleSignInAuthenticationEvent>? _events;
  bool _busy = false;
  bool _webReady = false;
  String? _error;
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _showEmail = false;
  bool _creating = false;
  bool _emailBusy = false;
  bool _obscure = true;

  @override
  void initState() {
    super.initState();
    if (!_google.available) return;
    if (kIsWeb) {
      _events = _google.events.listen(_onWebEvent, onError: _onWebError);
      _google.ensureReady().then((_) {
        if (!mounted) return;
        setState(() => _webReady = true);
        _google.attemptSilent().ignore(); // One Tap; results arrive as events.
      }, onError: (Object error) {
        if (mounted) setState(() => _error = GoogleAuth.messageFor(error));
      });
    } else {
      _signInReturningUser();
    }
  }

  @override
  void dispose() {
    _events?.cancel();
    _name.dispose();
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _submitEmail() async {
    if (_busy || _emailBusy || !_formKey.currentState!.validate()) return;
    FocusScope.of(context).unfocus();
    setState(() {
      _emailBusy = true;
      _error = null;
    });
    try {
      if (_creating) {
        await AuthService.instance.register(
            fullName: _name.text.trim(),
            email: _email.text.trim(),
            password: _password.text);
      } else {
        await AuthService.instance
            .login(email: _email.text.trim(), password: _password.text);
      }
      if (!mounted) return;
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(builder: (_) => const HomeShell()),
        (route) => false,
      );
      return;
    } on AuthException catch (error) {
      if (mounted) setState(() => _error = error.message);
    }
    if (mounted) setState(() => _emailBusy = false);
  }

  static final _emailPattern = RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$');

  Widget _emailForm() => Form(
        key: _formKey,
        child: AutofillGroup(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SegmentedButton<bool>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(value: false, label: Text('Log in')),
                  ButtonSegment(value: true, label: Text('Create account')),
                ],
                selected: {_creating},
                onSelectionChanged: _emailBusy
                    ? null
                    : (value) => setState(() {
                          _creating = value.first;
                          _error = null;
                        }),
              ),
              const SizedBox(height: 14),
              if (_creating) ...[
                TextFormField(
                  controller: _name,
                  enabled: !_emailBusy,
                  textCapitalization: TextCapitalization.words,
                  textInputAction: TextInputAction.next,
                  autofillHints: const [AutofillHints.name],
                  decoration: const InputDecoration(
                    labelText: 'Full name',
                    prefixIcon: Icon(Icons.person_outline_rounded),
                  ),
                  validator: (value) => value == null || value.trim().isEmpty
                      ? 'Enter your name'
                      : null,
                ),
                const SizedBox(height: 12),
              ],
              TextFormField(
                controller: _email,
                enabled: !_emailBusy,
                keyboardType: TextInputType.emailAddress,
                textInputAction: TextInputAction.next,
                autofillHints: const [AutofillHints.email],
                decoration: const InputDecoration(
                  labelText: 'Email',
                  prefixIcon: Icon(Icons.email_outlined),
                ),
                validator: (value) =>
                    _emailPattern.hasMatch(value?.trim() ?? '')
                        ? null
                        : 'Enter a valid email',
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _password,
                enabled: !_emailBusy,
                obscureText: _obscure,
                textInputAction: TextInputAction.done,
                autofillHints: [
                  _creating ? AutofillHints.newPassword : AutofillHints.password
                ],
                onFieldSubmitted: (_) => _submitEmail(),
                decoration: InputDecoration(
                  labelText: 'Password',
                  prefixIcon: const Icon(Icons.lock_outline_rounded),
                  suffixIcon: IconButton(
                    tooltip: _obscure ? 'Show password' : 'Hide password',
                    icon: Icon(_obscure
                        ? Icons.visibility_outlined
                        : Icons.visibility_off_outlined),
                    onPressed: () => setState(() => _obscure = !_obscure),
                  ),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Enter your password';
                  }
                  if (_creating && value.length < 8) {
                    return 'Use at least 8 characters';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 14),
              ElevatedButton(
                onPressed: _busy || _emailBusy ? null : _submitEmail,
                child: _emailBusy
                    ? const SizedBox.square(
                        dimension: 20,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white),
                      )
                    : Text(_creating ? 'Create account' : 'Log in'),
              ),
            ],
          ),
        ),
      );

  Future<void> _signInReturningUser() async {
    setState(() => _busy = true);
    try {
      // Never leave the button spinning if Google does not answer.
      final account =
          await _google.attemptSilent().timeout(const Duration(seconds: 8));
      if (account != null) {
        await _finish(account);
        return;
      }
    } catch (_) {
      // No saved Google account; the button still works.
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _signIn() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final account = await _google.authenticate();
      if (account != null) {
        await _finish(account);
        return;
      }
    } catch (error) {
      if (mounted) setState(() => _error = GoogleAuth.messageFor(error));
    }
    if (mounted) setState(() => _busy = false);
  }

  void _onWebEvent(GoogleSignInAuthenticationEvent event) {
    if (event is GoogleSignInAuthenticationEventSignIn && !_busy) {
      setState(() {
        _busy = true;
        _error = null;
      });
      _finish(event.user).whenComplete(() {
        if (mounted) setState(() => _busy = false);
      });
    }
  }

  void _onWebError(Object error) {
    if (!GoogleAuth.isCancel(error) && mounted) {
      setState(() => _error = GoogleAuth.messageFor(error));
    }
  }

  Future<void> _finish(GoogleSignInAccount account) async {
    try {
      await AuthService.instance.signInWithGoogle(account);
      if (!mounted) return;
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(builder: (_) => const HomeShell()),
        (route) => false,
      );
    } on AuthException catch (error) {
      await _google.signOut(); // Let the user pick an account again.
      if (mounted) {
        setState(() {
          _busy = false;
          _error = error.message;
        });
      }
    }
  }

  Widget _signInArea() {
    if (!_google.isSupportedPlatform) {
      return const _Notice(
          'Google sign-in works in the phone app and on the web.');
    }
    if (!_google.isConfigured) {
      return const _Notice('Google sign-in is not set up in this build yet.');
    }
    if (kIsWeb) {
      return SizedBox(
        height: 54,
        child: Center(
          child: _busy || !_webReady
              ? const _Progress('Signing in…')
              : googleWebButton(),
        ),
      );
    }
    return _GoogleButton(
        busy: _busy, onPressed: _busy || _emailBusy ? null : _signIn);
  }

  @override
  Widget build(BuildContext context) {
    final canPop = Navigator.of(context).canPop();
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(24, 12, 24, 24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 440),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Row(
                    children: [
                      if (canPop)
                        IconButton(
                          tooltip: 'Back',
                          onPressed: () => Navigator.of(context).maybePop(),
                          icon: const Icon(Icons.arrow_back_rounded),
                        ),
                      const _LogoMark(),
                      const SizedBox(width: 10),
                      const Text.rich(
                        TextSpan(children: [
                          TextSpan(
                              text: 'Fabri',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 22)),
                          TextSpan(
                              text: 'Care',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800,
                                  fontSize: 22,
                                  color: AppColors.coral)),
                        ]),
                      ),
                    ],
                  ),
                  const SizedBox(height: 28),
                  Container(
                    padding: const EdgeInsets.fromLTRB(22, 24, 22, 18),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(28),
                      border: Border.all(color: AppColors.border),
                      gradient: const LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [Color(0xFFFFFCF3), Color(0xFFF6E3D3)],
                      ),
                    ),
                    child: const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Know your fabric.\nCare for it right.',
                            style: TextStyle(
                                fontSize: 30,
                                height: 1.15,
                                fontWeight: FontWeight.w900,
                                letterSpacing: -0.4)),
                        SizedBox(height: 10),
                        Text('Scan a garment to see its fabric, condition and care.',
                            style: TextStyle(
                                color: AppColors.inkSoft,
                                fontSize: 15,
                                height: 1.35)),
                        SizedBox(height: 18),
                        _Feature(Icons.texture_rounded,
                            'Fabric identified in seconds'),
                        _Feature(Icons.favorite_outline_rounded,
                            'Condition and lifespan check'),
                        _Feature(Icons.auto_awesome_rounded,
                            'Care advice for every scan'),
                      ],
                    ),
                  ),
                  const SizedBox(height: 28),
                  _signInArea(),
                  const SizedBox(height: 14),
                  const Row(children: [
                    Expanded(child: Divider()),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 10),
                      child: Text('or',
                          style: TextStyle(color: AppColors.inkSoft)),
                    ),
                    Expanded(child: Divider()),
                  ]),
                  const SizedBox(height: 8),
                  AnimatedSize(
                    duration: const Duration(milliseconds: 200),
                    alignment: Alignment.topCenter,
                    child: _showEmail
                        ? _emailForm()
                        : TextButton.icon(
                            onPressed: _busy
                                ? null
                                : () => setState(() => _showEmail = true),
                            icon: const Icon(Icons.mail_outline_rounded),
                            label: const Text('Use email instead'),
                          ),
                  ),
                  AnimatedSize(
                    duration: const Duration(milliseconds: 200),
                    child: _error == null
                        ? const SizedBox(width: double.infinity)
                        : _ErrorBanner(_error!),
                  ),
                  const SizedBox(height: 14),
                  const Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.lock_outline_rounded,
                          size: 14, color: AppColors.inkSoft),
                      SizedBox(width: 6),
                      Flexible(
                        child: Text('We only use your name and email.',
                            style: TextStyle(
                                fontSize: 12.5, color: AppColors.inkSoft)),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _LogoMark extends StatelessWidget {
  const _LogoMark();

  @override
  Widget build(BuildContext context) => Container(
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: AppColors.ink,
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Icon(Icons.checkroom_rounded,
            color: AppColors.surface, size: 22),
      );
}

class _Feature extends StatelessWidget {
  final IconData icon;
  final String label;
  const _Feature(this.icon, this.label);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: AppColors.coral.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, size: 18, color: AppColors.coral),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(label,
                  style: const TextStyle(fontWeight: FontWeight.w700)),
            ),
          ],
        ),
      );
}

class _GoogleButton extends StatelessWidget {
  final bool busy;
  final VoidCallback? onPressed;
  const _GoogleButton({required this.busy, required this.onPressed});

  @override
  Widget build(BuildContext context) => SizedBox(
        height: 56,
        child: OutlinedButton(
          onPressed: onPressed,
          style: OutlinedButton.styleFrom(
            backgroundColor: Colors.white,
            disabledBackgroundColor: Colors.white,
            side: const BorderSide(color: Color(0xFFDADCE0)),
            shape: const StadiumBorder(),
          ),
          child: busy
              ? const _Progress('Signing in…')
              : const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox.square(dimension: 20, child: _GoogleLogo()),
                    SizedBox(width: 12),
                    Text('Continue with Google',
                        style: TextStyle(
                            color: Color(0xFF1F1F1F),
                            fontSize: 16,
                            fontWeight: FontWeight.w600)),
                  ],
                ),
        ),
      );
}

class _Progress extends StatelessWidget {
  final String label;
  const _Progress(this.label);

  @override
  Widget build(BuildContext context) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox.square(
              dimension: 18, child: CircularProgressIndicator(strokeWidth: 2)),
          const SizedBox(width: 12),
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      );
}

class _Notice extends StatelessWidget {
  final String text;
  const _Notice(this.text);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.surfaceMuted,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            const Icon(Icons.info_outline_rounded, color: AppColors.inkSoft),
            const SizedBox(width: 10),
            Expanded(child: Text(text)),
          ],
        ),
      );
}

class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner(this.message);

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.only(top: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppColors.danger.withValues(alpha: 0.09),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            const Icon(Icons.error_outline_rounded,
                color: AppColors.danger, size: 20),
            const SizedBox(width: 10),
            Expanded(
              child: Text(message,
                  style: const TextStyle(color: AppColors.danger)),
            ),
          ],
        ),
      );
}

/// The four-colour Google "G".
class _GoogleLogo extends StatelessWidget {
  const _GoogleLogo();

  @override
  Widget build(BuildContext context) =>
      const CustomPaint(painter: _GoogleLogoPainter());
}

class _GoogleLogoPainter extends CustomPainter {
  const _GoogleLogoPainter();

  static const _blue = Color(0xFF4285F4);
  static const _green = Color(0xFF34A853);
  static const _yellow = Color(0xFFFBBC05);
  static const _red = Color(0xFFEA4335);

  @override
  void paint(Canvas canvas, Size size) {
    final side = size.shortestSide;
    final stroke = side * 0.22;
    final rect = Rect.fromLTWH(
        stroke / 2, stroke / 2, side - stroke, side - stroke);
    Paint arc(Color color) => Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke;
    double degrees(double value) => value * math.pi / 180;
    canvas
      ..drawArc(rect, degrees(-40), degrees(-105), false, arc(_red))
      ..drawArc(rect, degrees(135), degrees(80), false, arc(_yellow))
      ..drawArc(rect, degrees(45), degrees(90), false, arc(_green))
      ..drawArc(rect, degrees(0), degrees(45), false, arc(_blue))
      ..drawRect(
          Rect.fromLTRB(side * 0.5, side * 0.5 - stroke / 2, side,
              side * 0.5 + stroke / 2),
          Paint()..color = _blue);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
