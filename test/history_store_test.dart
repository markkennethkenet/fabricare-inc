import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:fabricare/models/scan_result.dart';
import 'package:fabricare/services/history_store.dart';

void main() {
  test(
      'history survives reopen, keeps every detection, and scopes delete/clear by account',
      () async {
    final directory =
        await Directory.systemTemp.createTemp('fabricare_history_test');
    final source = File('${directory.path}/source.jpg');
    await source.writeAsBytes([1, 2, 3]);
    final store = HistoryStore(Directory('${directory.path}/store'));
    await store.open();
    ScanResult scan(String id, String userId) => ScanResult(
        id: id,
        userId: userId,
        garmentName: 'Test garment',
        images: [ScanImage(view: 'front', path: source.path)],
        dateScanned: DateTime(2026, 10, 20),
        userInputs: const {'condition': 'new'},
        analysis: {
          'fabric_type': 'future_fabric',
          'model_type': 'glcm_dt',
          'fabric_model': 'GLCM + Decision Tree',
          'fabric_confidence': 91,
          'health_score': 70,
          'condition_rating': 'Fair',
          'estimated_remaining_months': 12,
          'damages': [
            for (var i = 0; i < 2; i++)
              {
                'id': '0:$i',
                'type': 'future_damage',
                'confidence': 85,
                'severity': 'minor',
                'bbox': [.1, .2, .1, .1],
                'area_ratio': .01,
                'image_index': 0,
                'view': 'front'
              }
          ]
        });
    final saved = await store.add(scan('1', '10'));
    await store.add(scan('2', '10'));
    await store.add(scan('3', '20'));
    expect(saved.image!.path, isNot(source.path));
    await store.close();
    final reopened = HistoryStore(Directory('${directory.path}/store'));
    await reopened.open();
    expect(reopened.forUser('10').length, 2);
    final restored = reopened.forUser('10').first;
    expect(restored.analysis['model_type'], 'glcm_dt');
    expect(restored.fabricModelName, 'GLCM + Decision Tree');
    expect(restored.confidencePercent, 91);
    expect(restored.dateScanned, DateTime(2026, 10, 20));
    expect(restored.damageDetection.items.length, 2);
    expect(restored.userInputs['condition'], 'new');
    expect(restored.analysis['health_score'], 70); // Preserve the raw record.
    expect(restored.healthRating, 'Assessment unavailable');
    expect(restored.lifespanRange, startsWith('Insufficient evidence'));
    expect(await restored.image!.exists(), isTrue);
    await reopened.delete('10', '1');
    expect(await saved.image!.exists(), isFalse);
    await reopened.clear('10');
    expect(reopened.forUser('10'), isEmpty);
    expect(reopened.forUser('20').length, 1);
    await reopened.close();
    await directory.delete(recursive: true);
  });

  test('replacing a scan keeps its saved photo and copies only the new view',
      () async {
    final directory =
        await Directory.systemTemp.createTemp('fabricare_replace_test');
    final front = File('${directory.path}/front.jpg');
    final back = File('${directory.path}/back.jpg');
    await front.writeAsBytes([1, 2, 3]);
    await back.writeAsBytes([4, 5, 6]);
    final store = HistoryStore(Directory('${directory.path}/store'));
    await store.open();
    final saved = await store.add(ScanResult(
        id: '5',
        userId: '10',
        garmentName: 'denim Garment',
        images: [ScanImage(view: 'front', path: front.path)],
        dateScanned: DateTime(2026, 11, 17),
        analysis: const {'fabric_type': 'denim'}));
    final updated = await store.replace(ScanResult(
        id: saved.id,
        userId: saved.userId,
        garmentName: 'denim Garment',
        images: [
          saved.images.single,
          ScanImage(view: 'back', path: back.path),
        ],
        dateScanned: saved.dateScanned,
        analysis: const {'fabric_type': 'denim', 'fabric_confidence': 80}));
    expect(updated.images.first.path, saved.images.single.path);
    expect(await File(updated.images.first.path).readAsBytes(), [1, 2, 3]);
    expect(updated.images.last.view, 'back');
    expect(updated.images.last.path, isNot(back.path));
    expect(await File(updated.images.last.path).readAsBytes(), [4, 5, 6]);
    final scans = store.forUser('10');
    expect(scans.length, 1); // the same scan, not a second one
    expect(scans.single.images.length, 2);
    expect(scans.single.analysis['fabric_confidence'], 80);
    await store.close();
    await directory.delete(recursive: true);
  });
}
