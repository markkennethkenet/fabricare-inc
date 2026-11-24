import 'package:fabricare/services/fabricare_api_service.dart';
import 'package:flutter_test/flutter_test.dart';

AnalyzeResponse scan(String fabric, double? score, {bool review = false}) =>
    AnalyzeResponse({
      'fabric_type': fabric,
      'fabric_confidence': score,
      'fabric': {'review_required': review, 'threshold_percent': 60},
    });

void main() {
  test('a fabric below the threshold is rejected and not saved', () {
    final low = scan('silk', 59.6, review: true);
    expect(low.fabricRejected, isTrue);
    // 59.6% must not read as meeting the 60% threshold.
    expect(low.rejectionMessage,
        'Best guess: Silk at 59%, below the 60% needed to identify it.');
  });

  test('an uncertain fabric is rejected without a guess', () {
    final uncertain = scan('uncertain', null, review: true);
    expect(uncertain.fabricRejected, isTrue);
    expect(uncertain.rejectionMessage, contains('could not identify'));
  });

  test('a fabric at or above the threshold is kept', () {
    expect(scan('cotton', 60).fabricRejected, isFalse);
    expect(scan('cotton', 92.4).fabricRejected, isFalse);
    // An older API response without the fabric section is only rejected when uncertain.
    expect(
        const AnalyzeResponse({'fabric_type': 'cotton', 'fabric_confidence': 40})
            .fabricRejected,
        isFalse);
  });
}
