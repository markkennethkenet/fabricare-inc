import 'package:fabricare/models/scan_result.dart';
import 'package:fabricare/screens/auth/login_screen.dart';
import 'package:fabricare/screens/onboarding/onboarding_screen.dart';
import 'package:fabricare/widgets/scan_compare_sheet.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

ScanResult scan(String id, DateTime date, String fabric,
        {int? health, int findings = 0, int? months}) =>
    ScanResult(
      id: id,
      userId: '1',
      garmentName: '$fabric Garment',
      images: [ScanImage(view: 'front', path: 'missing/$id.jpg')],
      dateScanned: date,
      analysis: {
        'fabric_type': fabric,
        if (health != null) 'health_score': health,
        'condition_assessment': {
          'observations': [
            for (var i = 0; i < findings; i++) {'type': 'stain'}
          ],
        },
        if (months != null) 'lifespan': {'remaining_months': months},
      },
    );

void main() {
  testWidgets('login offers Google only and explains a build without it',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: LoginScreen()));
    expect(find.text('Know your fabric.\nCare for it right.'), findsOneWidget);
    await tester.pumpAndSettle(); // the silent sign-in attempt finds no account
    expect(find.text('Continue with Google'), findsOneWidget);
    expect(find.text('We only use your Google name and email.'),
        findsOneWidget);
    for (final gone in ['Continue with Apple', 'Sign Up', 'Password', 'Email']) {
      expect(find.textContaining(gone), findsNothing, reason: gone);
    }
    expect(tester.takeException(), isNull);
  });

  testWidgets('onboarding ends at the Google sign-in', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: OnboardingScreen()));
    expect(find.textContaining('Create Account'), findsNothing);
    expect(find.textContaining('Already have an account'), findsNothing);
    await tester.tap(find.text('Next'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Next'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Get started'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400)); // page transition
    expect(find.byType(LoginScreen), findsOneWidget);
    await tester.pump(const Duration(seconds: 9)); // silent sign-in gives up
    expect(find.text('Continue with Google'), findsOneWidget);
  });

  test('earlier scans are the same account, older, with a front photo', () {
    final now = scan('now', DateTime(2026, 11, 17), 'denim');
    final older = scan('a', DateTime(2026, 11, 10), 'denim');
    final newer = scan('b', DateTime(2026, 11, 18), 'denim');
    final other = ScanResult(
        id: 'c',
        userId: '2',
        garmentName: 'x',
        images: older.images,
        dateScanned: DateTime(2026, 11, 1),
        analysis: const {});
    final oldest = scan('d', DateTime(2026, 10, 1), 'cotton');
    expect(earlierScans(now, [oldest, now, newer, other, older]).map((s) => s.id),
        ['a', 'd']);
  });

  testWidgets('compare sheet picks an earlier scan from a dropdown',
      (tester) async {
    final current =
        scan('now', DateTime(2026, 11, 17), 'denim', health: 85, findings: 1, months: 40);
    final earlier = [
      scan('cotton', DateTime(2026, 11, 12), 'cotton', health: 95, months: 50),
      scan('denim', DateTime(2026, 11, 2), 'denim', health: 90, findings: 3, months: 42),
    ];
    await tester.pumpWidget(MaterialApp(
        home: Scaffold(
            body: ScanCompareSheet(current: current, earlier: earlier))));
    // The latest scan of the same fabric is chosen first.
    expect(find.textContaining('Denim · '), findsOneWidget);
    expect(find.text('90/100'), findsOneWidget);
    expect(find.text('−5'), findsOneWidget); // health fell
    expect(find.text('−2'), findsWidgets); // fewer findings, shorter lifespan
    await tester.tap(find.textContaining('Denim · '));
    await tester.pumpAndSettle();
    await tester.tap(find.textContaining('Cotton · ').last);
    await tester.pumpAndSettle();
    expect(find.text('95/100'), findsOneWidget);
    expect(find.text('−10'), findsWidgets);
    expect(find.text('+1'), findsOneWidget); // one more finding than before
    expect(tester.takeException(), isNull);
  });
}
