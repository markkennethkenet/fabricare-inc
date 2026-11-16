import 'dart:io';

import 'package:fabricare/data/mock_data.dart';
import 'package:fabricare/screens/history/history_screen.dart';
import 'package:fabricare/services/fabricare_api_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hive/hive.dart';

void main() {
  testWidgets('history helps start a scan and find saved reviews',
      (tester) async {
    final data = AppData.instance;
    final storage =
        Directory.systemTemp.createTempSync('fabricare-history-ui-');
    const pathProvider = MethodChannel('plugins.flutter.io/path_provider');
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(pathProvider, (_) async => storage.path);
    addTearDown(() async {
      data.setUserId(null);
      await Hive.close();
      messenger.setMockMethodCallHandler(pathProvider, null);
      await storage.delete(recursive: true);
    });

    var started = false;
    await tester.pumpWidget(
      MaterialApp(home: HistoryScreen(onScan: () => started = true)),
    );
    await tester.tap(find.text('Start a scan'));
    expect(started, isTrue);

    await tester.runAsync(() async {
      await data.init();
      data.setUserId('1');
      final scan = await data.addAnalysis(
        [],
        const AnalyzeResponse({
          'fabric_type': 'Linen',
          'health_score': 90,
          'condition_rating': 'Excellent',
          'damages': [
            {'id': '0:1', 'type': 'stain', 'confidence': 80}
          ],
        }),
      );
      await data.saveDamageReview(scan, {'health_score': 100}, {'0:1': false});
      await data.addAnalysis(
        [],
        const AnalyzeResponse({
          'fabric_type': 'Denim',
          'damages': [
            {'id': '0:2', 'type': 'hole', 'confidence': 80}
          ],
        }),
      );
    });
    await tester.pumpAndSettle();

    expect(find.text('2 saved scans'), findsOneWidget);
    expect(find.text('Linen Garment'), findsOneWidget);
    expect(find.text('Denim Garment'), findsOneWidget);
    await tester.tap(find.text('1 scan needs review'));
    await tester.pumpAndSettle();
    expect(find.text('Denim Garment'), findsOneWidget);
    expect(find.text('Linen Garment'), findsNothing);

    await tester.tap(find.widgetWithText(ChoiceChip, 'Reviewed'));
    await tester.pumpAndSettle();
    expect(find.text('Linen Garment'), findsOneWidget);
    await tester.enterText(find.byType(TextField), 'denim');
    await tester.pumpAndSettle();
    expect(find.text('No matching scans'), findsOneWidget);

    await tester.tap(find.text('Show all scans'));
    await tester.pumpAndSettle();
    expect(find.text('Linen Garment'), findsOneWidget);
    await tester.tap(find.text('Linen Garment'));
    await tester.pumpAndSettle();
    expect(find.text('Saved scan'), findsOneWidget);
  });
}
