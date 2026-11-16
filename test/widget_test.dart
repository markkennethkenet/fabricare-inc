import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fabricare/models/scan_result.dart';
import 'package:fabricare/screens/home/home_shell.dart';
import 'package:fabricare/screens/profile/edit_profile_screen.dart';
import 'package:fabricare/screens/scan/scan_result_screen.dart';
import 'package:fabricare/screens/scan/scan_screen.dart';
import 'package:fabricare/screens/tips/tips_screen.dart';

void main() {
  testWidgets('home shortcuts switch tabs and featured tips open in place',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: HomeShell()));

    final nextTip = find.byTooltip('Next care tip');
    await tester.ensureVisible(nextTip);
    await tester.tap(nextTip);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 220));
    expect(find.text('How to avoid shrinkage'), findsOneWidget);
    final readTip = find.text('Read full tip');
    await tester.ensureVisible(readTip);
    await tester.tap(readTip);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));
    expect(find.text('Avoid over-drying in the dryer.'), findsOneWidget);
    await tester.tap(find.byTooltip('Close tip'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 350));

    final startScan = find.text('Start scan');
    await tester.ensureVisible(startScan);
    await tester.tap(startScan);
    await tester.pump();
    expect(find.byType(ScanScreen), findsOneWidget);
    expect(find.text('Start scan'), findsNothing);
  });

  testWidgets('the big Scan button sits in the middle of the nav bar',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: HomeShell()));
    final labels = ['Home', 'History', 'Scan', 'Tips', 'Profile'];
    final centers = [
      for (final label in labels)
        tester.getCenter(find.text(label).last).dx,
    ];
    for (var i = 1; i < centers.length; i++) {
      expect(centers[i], greaterThan(centers[i - 1]),
          reason: '${labels[i]} should follow ${labels[i - 1]}');
    }
    final scanButton = find.byTooltip('Scan a garment');
    final screenWidth = tester.view.physicalSize.width / tester.view.devicePixelRatio;
    expect(tester.getCenter(scanButton).dx, closeTo(screenWidth / 2, 1));
    expect(tester.getSize(scanButton).width, greaterThanOrEqualTo(70));

    await tester.tap(scanButton);
    await tester.pump();
    expect(find.text('Scan a Garment'), findsOneWidget);

    await tester.tap(find.text('History').last);
    await tester.pump();
    expect(find.text('Scan a Garment'), findsNothing);
  });

  testWidgets('care guide filters tips and preserves search after reading',
      (tester) async {
    tester.view.physicalSize = const Size(340, 700);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(const MaterialApp(home: TipsScreen()));

    final search = find.byType(TextField);
    await tester.enterText(search, '  LINEN  ');
    await tester.pumpAndSettle();
    final guide = find.text('How to care for Linen');
    expect(guide, findsOneWidget);
    expect(find.text('How to care for Cotton'), findsNothing);
    await tester.ensureVisible(guide);
    await tester.pumpAndSettle();
    await tester.tap(guide);
    await tester.pumpAndSettle();
    expect(find.textContaining('Reshape the garment'), findsOneWidget);
    await tester.tap(find.byTooltip('Close tip'));
    await tester.pumpAndSettle();
    expect(guide, findsOneWidget);

    final laundry = find.widgetWithText(ChoiceChip, 'Laundry care');
    await tester.ensureVisible(laundry);
    await tester.pumpAndSettle();
    await tester.tap(laundry);
    await tester.pumpAndSettle();
    expect(find.text('No matching tips'), findsOneWidget);
    await tester.tap(find.byTooltip('Clear search'));
    await tester.pumpAndSettle();
    expect(find.text('How to prevent fading'), findsOneWidget);
    expect(guide, findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('guided scan requires front and offers garment history inputs',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ScanScreen()));
    expect(find.text('Front photo'), findsOneWidget);
    expect(find.text('Required'), findsOneWidget);
    expect(find.text('Back photo'), findsNothing);
    expect(find.text('Fabric close-up'), findsNothing);
    expect(find.text('Add a front photo to continue.'), findsOneWidget);
    expect(
        tester
            .widget<ElevatedButton>(
                find.widgetWithText(ElevatedButton, 'Analyze garment'))
            .onPressed,
        isNull);
    expect(find.text('Tap to open the smart camera'), findsOneWidget);
    final details = find.text('Garment details (optional)');
    await tester.scrollUntilVisible(details, 300);
    await tester.tap(details);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Garment history'), 300,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('I know the purchase date'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('model picker explains each model and marks the recommended one',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ScanScreen()));
    expect(find.text('Choose Fabric Identification Model'), findsOneWidget);
    expect(find.text('CNN — ConvNeXt-Tiny'), findsOneWidget);
    expect(find.text('Recommended'), findsOneWidget);
    expect(find.text('86.7%'), findsOneWidget);

    await tester.tap(find.text('Change'));
    await tester.pumpAndSettle();
    expect(find.text('GLCM + SVM'), findsOneWidget);
    expect(find.text('GLCM + Decision Tree'), findsOneWidget);
    expect(find.text('59.5%'), findsOneWidget);
    expect(find.text('55.5%'), findsOneWidget);
    expect(find.text('Best for'), findsNWidgets(3));
    expect(find.textContaining('Keep the Recommended model'), findsOneWidget);

    await tester.tap(find.text('GLCM + SVM'));
    await tester.pumpAndSettle();
    expect(find.text('GLCM + SVM'), findsOneWidget);
    expect(find.text('Recommended'), findsNothing);
    expect(find.textContaining('Less accurate than the recommended CNN'),
        findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('profile form keeps fields labeled and validates before saving',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: EditProfileScreen()));
    expect(find.text('Phone number (optional)'), findsOneWidget);
    expect(find.text('Location (optional)'), findsOneWidget);
    await tester.tap(find.text('Save changes'));
    await tester.pump();
    expect(find.text('Please enter your full name'), findsOneWidget);
    expect(find.text('Enter a valid email'), findsOneWidget);
  });

  testWidgets(
      'older results retain detections and withhold legacy health claims',
      (tester) async {
    final result = ScanResult(
        id: '1',
        userId: '1',
        garmentName: 'Test',
        images: [],
        dateScanned: DateTime(2026),
        analysis: {
          'fabric_type': 'Future blend',
          'fabric_confidence': 88,
          'health_score': 70,
          'condition_rating': 'Backend grade',
          'estimated_lifespan': 'About 12 months',
          'care_recommendation': {'laundry_mode': 'Normal wash'},
          'damages': [
            {'type': 'Future damage', 'confidence': 85, 'severity': 'minor'}
          ],
        });
    await tester
        .pumpWidget(MaterialApp(home: ScanResultScreen(result: result)));
    expect(find.text('Future Blend'), findsOneWidget);
    expect(find.text('88%'), findsOneWidget);
    expect(find.text('High model score'), findsOneWidget);
    expect(find.textContaining('Future damage'), findsOneWidget);
    final scrollable = find.byType(Scrollable).first;
    await tester.scrollUntilVisible(find.text('Assessment unavailable'), 180,
        scrollable: scrollable);
    expect(find.textContaining('70/100'), findsNothing);
    await tester.scrollUntilVisible(
        find.text('Remaining months: undetermined'), 180,
        scrollable: scrollable);
    expect(find.textContaining('About 12 months'), findsNothing);
    expect(find.text('Normal wash'), findsNothing);
  });
  testWidgets('shows the AI laundry suggestion, alerts, health and lifespan',
      (tester) async {
    final result = ScanResult(
      id: '2',
      userId: '1',
      garmentName: 'Wool sweater',
      images: [],
      dateScanned: DateTime(2026, 11, 16),
      analysis: const {
        'fabric_type': 'wool',
        'fabric_confidence': 91.2,
        'methodology_version': 'visible-condition-and-label-care/2.0',
        'health_assessment': {'score': 80, 'grade': 'Good', 'deductions': []},
        'condition_assessment': {'category': 'Possible hole · awaiting review'},
        'lifespan': {
          'status': 'experimental_estimate',
          'remaining_months': 45,
          'remaining_months_low': 27,
          'remaining_months_high': 63,
          'summary': 'Review the marked areas first.',
          'factors': [
            {'factor': 'Typical total lifespan for wool', 'effect_months': 73, 'source': 'D'},
            {'factor': 'Possible hole', 'effect_months': -11, 'source': 'fabricare_assumption'},
          ],
        },
        'ai_care': {
          'status': 'model_suggestion',
          'model': 'Random Forest',
          'laundry_mode': 'repair_first',
          'label': 'Repair before washing',
          'detail': 'Mend the opening first; washing and pulling can enlarge it.',
          'confidence': 97.0,
          'reasons': [
            {'text': 'An opening was detected.', 'reference_ids': ['I']}
          ],
          'alerts': [
            {'level': 'high', 'title': 'Possible opening', 'action': 'Mend it first.', 'reference_ids': ['I']},
            {'level': 'medium', 'title': 'Felting risk', 'action': 'Avoid hot water.', 'reference_ids': ['H']},
          ],
          'note': 'A suggestion from the scan. Check the garment care label before washing.',
        },
      },
    );
    await tester.pumpWidget(MaterialApp(home: ScanResultScreen(result: result)));
    expect(find.text('AI laundry suggestion'), findsOneWidget);
    expect(find.text('Repair before washing'), findsOneWidget);
    expect(find.text('Random Forest'), findsOneWidget);
    expect(find.text('97% model confidence'), findsOneWidget);
    expect(find.text('Possible opening'), findsOneWidget);
    expect(find.text('Felting risk'), findsOneWidget);
    final scrollable = find.byType(Scrollable).first;
    await tester.scrollUntilVisible(find.text('Fabric health 80/100'), 200, scrollable: scrollable);
    expect(find.text('Good'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('About 27–63 months'), 200, scrollable: scrollable);
    expect(find.text('Remaining months: undetermined'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('shows a candidate and its care precaution at 62.5 percent',
      (tester) async {
    final result = ScanResult(
      id: '1',
      userId: '1',
      garmentName: 'Test',
      images: [],
      dateScanned: DateTime(2026),
      analysis: const {
        'fabric_type': 'Cotton',
        'fabric_confidence': 62.5,
        'fabric_needs_confirmation': true,
        'methodology_version': 'test-method',
        'condition_assessment': {
          'category': 'Possible stain · awaiting review',
          'observations': [
            {
              'type': 'stain',
              'location': 'Front photo · upper left',
              'confirmed': false
            }
          ],
        },
        'care_guidance': {
          'recommendations': [
            {
              'action': 'Avoid strong stain treatments.',
              'reason': 'The mark is unverified.',
              'selected_because': 'The detector flagged the front photo.',
              'reference_ids': ['N']
            },
            {'action': 'If this garment is cotton, sort it by colour.'},
          ],
        },
        'laundry_settings': {
          'confirmed_settings': {},
          'general_guidance': [
            {'action': 'Set uncertain items aside before washing.'},
          ],
          'unknown_settings': ['Wash temperature is unknown.'],
        },
        'references': [
          {
            'id': 'N',
            'title': 'Oregon State University Extension · Stain removal',
            'url': 'https://extension.oregonstate.edu/'
          },
        ],
      },
    );
    await tester
        .pumpWidget(MaterialApp(home: ScanResultScreen(result: result)));
    expect(find.text('Cotton'), findsOneWidget);
    expect(find.text('62.5%'), findsOneWidget);
    expect(find.text('Moderate model score'), findsOneWidget);
    expect(find.text('Needs confirmation'), findsOneWidget);
    final nameStyle = tester.widget<Text>(find.text('Cotton')).style;
    expect(nameStyle?.fontSize, greaterThanOrEqualTo(40));
    expect(find.text('stain · Front photo · upper left · Candidate'),
        findsOneWidget);
    final scrollable = find.byType(Scrollable).first;
    await tester.scrollUntilVisible(find.text('Care Recommendations'), 180,
        scrollable: scrollable);
    expect(find.text('Avoid strong stain treatments.'), findsOneWidget);
    expect(find.text('If this garment is cotton, sort it by colour.'),
        findsOneWidget);
    await tester.scrollUntilVisible(find.text('Why this result?'), 180,
        scrollable: scrollable);
    expect(find.text('Advanced scientific details'), findsNothing);
    final why = find.text('Why this result?');
    await tester.ensureVisible(why);
    await tester.pumpAndSettle();
    await tester.tap(why);
    await tester.pumpAndSettle();
    expect(
        find.textContaining('Oregon State University Extension'), findsWidgets);
  });
}
