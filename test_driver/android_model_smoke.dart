import 'package:fabricare/models/scan_result.dart';
import 'package:fabricare/services/fabricare_api_service.dart';
import 'package:fabricare/services/history_store.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

// Run on Android with API_BASE_URL set to the USB forwarded local backend.
// Stage a garment image in the app cache as fabricare-model-test.jpg.
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(
      home: Scaffold(
          body: Center(
    child: Text('Checking all three fabric models…'),
  ))));

  const expectedNames = {
    'cnn': 'MobileNetV2 (six fabrics)',
    'svm': 'GLCM + SVM',
    'glcm_dt': 'GLCM + Decision Tree',
  };
  final results = <String>[];
  final failures = <String>[];
  try {
    final cache = await getTemporaryDirectory();
    final image = ScanImage(
      view: 'front',
      path: '${cache.path}/fabricare-model-test.jpg',
    );
    final api = FabricCareApiService.instance;
    await api.init();
    if (!api.isReady) throw StateError(api.loadError ?? 'API unavailable');
    final quality = await api.checkQuality([image]);
    final passed = (quality.single['quality'] as Map)['passed'] == true;
    final directory = await cache.createTemp('fabricare-smoke-');
    final store = HistoryStore(directory);
    await store.open();
    try {
      for (final entry in expectedNames.entries) {
        try {
          final model = entry.key;
          final response = await api.analyzeImages(
            [image],
            modelType: model,
            allowLowQuality: !passed,
          );
          final saved = await store.add(response.toScanResult([image], '1'));
          final restored = store.forUser('1').singleWhere(
                (scan) => scan.id == saved.id,
              );
          if (restored.analysis['model_type'] != model ||
              restored.fabricModelName != entry.value ||
              !api.fabricLabels.contains(restored.fabricType) ||
              restored.hasFabricScore != (model != 'svm') ||
              !restored.hasDamageDetection ||
              !restored.hasHealthAssessment ||
              restored.care.isEmpty ||
              restored.analysis['estimated_remaining_months'] != null ||
              !await restored.images.single.file.exists()) {
            throw StateError('Unexpected model result or saved scan');
          }
          final score = restored.hasFabricScore
              ? restored.analysis['fabric_confidence']
              : 'unavailable';
          final result = '$model: ${restored.fabricType}, score $score, '
              '${restored.damageDetection.allItems.length} damage findings, '
              '${restored.healthRating}';
          results.add(result);
          debugPrint('FABRICARE_ANDROID_SMOKE PASS $result');
        } catch (error) {
          failures.add(entry.key);
          debugPrint('FABRICARE_ANDROID_SMOKE FAIL ${entry.key}: $error');
        }
      }
    } finally {
      await store.close();
      await directory.delete(recursive: true);
    }
  } catch (error) {
    failures.add('setup');
    debugPrint('FABRICARE_ANDROID_SMOKE FAIL setup: $error');
  }
  debugPrint('FABRICARE_ANDROID_SMOKE DONE ${results.length}/3 passed');
  runApp(MaterialApp(
      home: Scaffold(
          body: SafeArea(
              child: ListView(
    padding: const EdgeInsets.all(20),
    children: [
      Text('${results.length}/3 Android model checks passed'),
      for (final result in results) Text(result),
      for (final failure in failures) Text('Failed: $failure'),
    ],
  )))));
}
