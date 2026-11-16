import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';

import 'api_config.dart';

/// Google sign-in on Android, iOS and the web.
///
/// Phones show Google's account picker from [authenticate]. The web uses
/// Google's own button instead, and its sign-ins arrive on [events].
class GoogleAuth {
  GoogleAuth._();
  static final GoogleAuth instance = GoogleAuth._();

  Future<void>? _ready;

  bool get isConfigured => ApiConfig.googleWebClientId.isNotEmpty;

  bool get isSupportedPlatform =>
      kIsWeb ||
      defaultTargetPlatform == TargetPlatform.android ||
      defaultTargetPlatform == TargetPlatform.iOS;

  bool get available => isConfigured && isSupportedPlatform;

  Future<void> ensureReady() => _ready ??= GoogleSignIn.instance.initialize(
        clientId: kIsWeb ? ApiConfig.googleWebClientId : null,
        serverClientId: kIsWeb ? null : ApiConfig.googleWebClientId,
      );

  Stream<GoogleSignInAuthenticationEvent> get events =>
      GoogleSignIn.instance.authenticationEvents;

  /// Shows Google's account picker. Returns null when the user closes it.
  Future<GoogleSignInAccount?> authenticate() async {
    await ensureReady();
    try {
      return await GoogleSignIn.instance.authenticate();
    } on GoogleSignInException catch (error) {
      if (isCancel(error)) return null;
      rethrow;
    }
  }

  /// Signs back in without a prompt when Google allows it (a returning user).
  /// On the web the result arrives on [events] instead.
  Future<GoogleSignInAccount?> attemptSilent() async {
    await ensureReady();
    return await GoogleSignIn.instance.attemptLightweightAuthentication();
  }

  Future<void> signOut() async {
    if (_ready == null) return;
    try {
      await GoogleSignIn.instance.signOut();
    } catch (error) {
      debugPrint('Google sign-out failed: $error');
    }
  }

  static bool isCancel(Object error) =>
      error is GoogleSignInException &&
      (error.code == GoogleSignInExceptionCode.canceled ||
          error.code == GoogleSignInExceptionCode.interrupted ||
          error.code == GoogleSignInExceptionCode.uiUnavailable);

  static String messageFor(Object error) {
    if (error is GoogleSignInException) {
      return switch (error.code) {
        GoogleSignInExceptionCode.clientConfigurationError ||
        GoogleSignInExceptionCode.providerConfigurationError =>
          'Google sign-in is not set up for this app yet.',
        _ => 'Google sign-in did not finish. Try again.',
      };
    }
    return 'Google sign-in did not finish. Try again.';
  }
}
