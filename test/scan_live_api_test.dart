import 'dart:io';

import 'package:fabricare/data/mock_data.dart';
import 'package:fabricare/models/scan_result.dart';
import 'package:fabricare/screens/scan/scan_result_screen.dart';
import 'package:fabricare/screens/scan/scan_screen.dart';
import 'package:fabricare/services/fabricare_api_service.dart';
import 'package:fabricare/widgets/section_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hive/hive.dart';

void main() {
  const photoPath = String.fromEnvironment('SCAN_TEST_PHOTO');
  const reviewDamage = bool.fromEnvironment('SCAN_TEST_REVIEW_DAMAGE');
  const expectedVerification = String.fromEnvironment('SCAN_TEST_VERIFICATION');
  const classifier =
      String.fromEnvironment('SCAN_TEST_CLASSIFIER', defaultValue: 'cnn');
  const modelNames = {
    'cnn': 'MobileNetV2 (six fabrics)',
    'svm': 'GLCM + SVM',
    'glcm_dt': 'GLCM + Decision Tree',
  };
  testWidgets('selected photo completes a scan against the live API',
      (tester) async {
    final previousOverride = HttpOverrides.current;
    HttpOverrides.global = null;
    addTearDown(() => HttpOverrides.global = previousOverride);

    final storage = Directory.systemTemp.createTempSync('fabricare-scan-');
    addTearDown(() async {
      await Hive.close();
      for (var attempt = 0;; attempt++) {
        try {
          await storage.delete(recursive: true);
          break;
        } on PathAccessException {
          if (attempt == 4) rethrow;
          // Windows image decoders may still be releasing a copied photo.
          await Future<void>.delayed(const Duration(milliseconds: 100));
        }
      }
    });
    final messenger = tester.binding.defaultBinaryMessenger;
    const picker = MethodChannel('plugins.flutter.io/image_picker');
    const pathProvider = MethodChannel('plugins.flutter.io/path_provider');
    const launcher = MethodChannel('plugins.flutter.io/url_launcher');
    String? openedReference;
    messenger.setMockMethodCallHandler(launcher, (call) async {
      expect(call.method, 'launch');
      openedReference = (call.arguments as Map)['url'] as String;
      return true;
    });
    messenger.setMockMethodCallHandler(picker, (call) async {
      expect(call.method, 'pickImage');
      return File(photoPath).absolute.path;
    });
    messenger.setMockMethodCallHandler(pathProvider, (call) async {
      expect(call.method, 'getApplicationSupportDirectory');
      return storage.path;
    });
    addTearDown(() {
      messenger.setMockMethodCallHandler(picker, null);
      messenger.setMockMethodCallHandler(pathProvider, null);
      messenger.setMockMethodCallHandler(launcher, null);
      AppData.instance.setUserId(null);
    });

    final api = FabricCareApiService.instance;
    await tester.runAsync(() async {
      await AppData.instance.init();
      AppData.instance.setUserId('1');
      await AppData.instance.addAnalysis(
        [ScanImage(view: 'front', path: File(photoPath).absolute.path)],
        const AnalyzeResponse({'fabric_type': 'Earlier'}),
      );
      await api.init();
    });
    expect(api.isReady, isTrue, reason: api.loadError);
    expect(api.fabricLabels,
        ['cotton', 'denim', 'linen', 'polyester', 'silk', 'wool']);
    await tester.pumpWidget(const MaterialApp(home: ScanScreen()));
    expect(find.textContaining('CNN — MobileNetV2'), findsAtLeastNWidgets(1));
    if (classifier != 'cnn') {
      await tester.tap(find.textContaining('CNN — MobileNetV2').first);
      await tester.pumpAndSettle();
      await tester.tap(find.textContaining(modelNames[classifier]!).last);
      await tester.pumpAndSettle();
    }
    await tester.scrollUntilVisible(find.text('Gallery').first, 150);
    await tester.tap(find.text('Gallery').first);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Enter label instructions'), 150);
    await tester.tap(find.text('Enter label instructions'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Unknown').first);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Dry clean only').last);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Use these instructions'), 150);
    await tester.tap(find.text('Use these instructions'));
    await tester.pumpAndSettle();

    final analyze = find.text('Analyze garment');
    await tester.ensureVisible(analyze);
    await tester.pumpAndSettle();

    await tester.runAsync(() async {
      await tester.tap(analyze);
      final deadline = DateTime.now().add(const Duration(seconds: 45));
      while (find.byType(ScanResultScreen).evaluate().isEmpty &&
          DateTime.now().isBefore(deadline)) {
        await Future<void>.delayed(const Duration(milliseconds: 50));
        await tester.pump(const Duration(milliseconds: 50));
        final proceed = find.text('Continue anyway');
        if (proceed.evaluate().isNotEmpty) await tester.tap(proceed);
      }
    });
    await tester.pump(const Duration(milliseconds: 350));

    expect(find.byType(ScanResultScreen), findsOneWidget);
    final saved = AppData.instance.history.first;
    expect(saved.analysis['model_type'], classifier);
    expect(saved.fabricModelName, modelNames[classifier]);
    expect(saved.hasFabricScore, classifier != 'svm');
    expect(api.fabricLabels, contains(saved.fabricType));
    expect(saved.analysis['health_score'], isNull);
    expect(saved.analysis['estimated_remaining_months'], isNull);
    expect(saved.userInputs['care_label']['wash'], 'dry_clean_only');
    expect(saved.care['laundry_mode'], 'Dry clean only');
    expect(saved.care['max_temperature_c'], isNull);
    expect(
        saved.section('six_month_outlook')['status'], 'insufficient_evidence');
    expect(File(saved.images.single.path).existsSync(), isTrue);
    if (expectedVerification.isNotEmpty) {
      expect(saved.analysis['fabric']['verification_status'],
          expectedVerification);
    }
    await tester.scrollUntilVisible(find.text('Fabric Identification'), 300,
        scrollable: find.descendant(
            of: find.byType(ScanResultScreen),
            matching: find.byType(Scrollable)));
    final fabricCard =
        find.widgetWithText(SectionCard, 'Fabric Identification');
    expect(
        find.descendant(
          of: fabricCard,
          matching: find.text(classifier == 'svm'
              ? saved.fabricType
              : '${saved.fabricType} — ${(saved.analysis['fabric_confidence'] as num).toStringAsFixed(1).replaceFirst(RegExp(r'\.0$'), '')}% model score'),
        ),
        findsOneWidget);
    final resultsScroller = find.descendant(
        of: find.byType(ScanResultScreen), matching: find.byType(Scrollable));
    await tester.scrollUntilVisible(
        find.text('Six-Month Condition Outlook'), 260,
        scrollable: resultsScroller);
    expect(find.text('Care scenario'), findsOneWidget);
    final compare = find.text('Compare with an earlier scan');
    await tester.scrollUntilVisible(compare, 200, scrollable: resultsScroller);
    await tester.ensureVisible(compare);
    await tester.pumpAndSettle();
    final comparisonPhotos = find.descendant(
        of: find.byType(AlertDialog), matching: find.byType(Image));
    await tester.runAsync(() async {
      await tester.tap(compare);
      await tester.pumpAndSettle();
      await tester.tap(find.textContaining('Earlier Garment'));
      final deadline = DateTime.now().add(const Duration(seconds: 5));
      while (comparisonPhotos.evaluate().length != 2 &&
          DateTime.now().isBefore(deadline)) {
        await Future<void>.delayed(const Duration(milliseconds: 50));
        await tester.pump(const Duration(milliseconds: 50));
      }
    });
    expect(find.text('Compare front photos'), findsOneWidget);
    expect(comparisonPhotos, findsNWidgets(2));
    await tester.tap(find.text('Close'));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Laundry Suggestions'), 260,
        scrollable: resultsScroller);
    expect(find.text('Wash · Dry clean only'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Scientific Basis'), 260,
        scrollable: resultsScroller);
    final why = find.text('Why this result?');
    await tester.ensureVisible(why);
    await tester.pumpAndSettle();
    await tester.tap(why);
    await tester.pumpAndSettle();
    final iso = find.textContaining('ISO 15487:2018');
    await tester.scrollUntilVisible(iso, 200, scrollable: resultsScroller);
    await tester.ensureVisible(iso);
    await tester.pumpAndSettle();
    await tester.tap(iso);
    await tester.pumpAndSettle();
    expect(openedReference, 'https://www.iso.org/standard/67602.html');
    if (reviewDamage) {
      final findings = saved.damageDetection.allItems;
      expect(findings, isNotEmpty, reason: 'Use a photo with detected damage.');
      final scrollable = find.descendant(
          of: find.byType(ScanResultScreen), matching: find.byType(Scrollable));
      final dismiss = find.widgetWithText(ChoiceChip, 'Dismiss');
      await tester.scrollUntilVisible(find.text('Review Damage'), -300,
          scrollable: scrollable);
      if (dismiss.evaluate().isEmpty) {
        await tester.ensureVisible(find.text('Review Damage'));
        await tester.pumpAndSettle();
        await tester.tap(find.text('Review Damage'));
        await tester.pumpAndSettle();
      }
      for (var i = 0; i < findings.length; i++) {
        await tester.ensureVisible(dismiss.at(i));
        await tester.pump(const Duration(milliseconds: 200));
        await tester.tap(dismiss.at(i));
        await tester.pump();
      }
      final save = find.widgetWithText(FilledButton, 'Save damage review');
      await tester.ensureVisible(save);
      await tester.pump(const Duration(milliseconds: 200));
      await tester.runAsync(() async {
        await tester.tap(save);
        final deadline = DateTime.now().add(const Duration(seconds: 20));
        while (find.text('Damage review saved.').evaluate().isEmpty &&
            DateTime.now().isBefore(deadline)) {
          await Future<void>.delayed(const Duration(milliseconds: 50));
          await tester.pump(const Duration(milliseconds: 50));
        }
      });
      final reviewed = AppData.instance.history.first;
      expect(
          reviewed.damageReview, {for (final item in findings) item.id: false});
      expect(reviewed.section('condition_assessment')['observations'], isEmpty);
      expect(reviewed.damageDetection.allItems.length, findings.length);
    }
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pumpAndSettle();
    final cache = PaintingBinding.instance.imageCache;
    cache.clearLiveImages();
    cache.clear();
  }, skip: photoPath.isEmpty);
}
