import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/user.dart';
import 'auth_api_service.dart';
import 'google_auth.dart';

class AuthException implements Exception {
  final String message;
  const AuthException(this.message);

  @override
  String toString() => message;
}

class AuthService extends ChangeNotifier {
  AuthService._();
  static final AuthService instance = AuthService._();

  static const _photoKey = 'fabricare_google_photo';

  User? _currentUser;
  User? get currentUser => _currentUser;
  bool get isLoggedIn => _currentUser != null;

  /// The Google profile photo of the signed-in account, when it has one.
  String? _photoUrl;
  String? get photoUrl => _photoUrl;

  Future<void> init() async {
    await AuthApiService.instance.loadStoredToken();
    try {
      _currentUser = await AuthApiService.instance.fetchCurrentUser();
      if (_currentUser != null) {
        _photoUrl = (await SharedPreferences.getInstance()).getString(_photoKey);
      }
    } catch (e) {
      debugPrint('Auth restore failed: $e');
      _currentUser = null;
    }
    notifyListeners();
  }

  /// Signs in to FabriCare with the Google account the user picked.
  Future<User> signInWithGoogle(GoogleSignInAccount account) async {
    final idToken = account.authentication.idToken;
    if (idToken == null || idToken.isEmpty) {
      throw const AuthException(
          'Google did not return a sign-in token. Try again.');
    }
    try {
      final session =
          await AuthApiService.instance.signInWithGoogle(idToken);
      _currentUser = session.user;
      _photoUrl = session.picture ?? account.photoUrl;
      final prefs = await SharedPreferences.getInstance();
      if (_photoUrl == null) {
        await prefs.remove(_photoKey);
      } else {
        await prefs.setString(_photoKey, _photoUrl!);
      }
      notifyListeners();
      return _currentUser!;
    } on AuthApiException catch (e) {
      throw AuthException(e.message);
    } on AuthException {
      rethrow;
    } catch (e) {
      throw const AuthException('Could not finish signing in. Try again.');
    }
  }

  Future<void> logout() async {
    await AuthApiService.instance.logout();
    // Forget the Google choice so the next sign-in can pick another account.
    await GoogleAuth.instance.signOut();
    _currentUser = null;
    _photoUrl = null;
    await (await SharedPreferences.getInstance()).remove(_photoKey);
    notifyListeners();
  }

  Future<void> updateProfile({
    required String fullName,
    required String email,
    String? phone,
    String? location,
  }) async {
    if (_currentUser == null) throw const AuthException('Not logged in.');

    try {
      _currentUser = await AuthApiService.instance.updateProfile(
        fullName: fullName,
        email: email,
        phone: phone,
        location: location,
      );
      notifyListeners();
    } on AuthApiException catch (e) {
      throw AuthException(e.message);
    }
  }
}
