import 'dart:io';
import 'dart:typed_data';
import 'package:image_picker/image_picker.dart' show XFile;
import 'damage_detection.dart';

class ScanImage {
  final String view;

  /// A file path on phones; a blob or data URL on the web.
  final String path;
  const ScanImage({required this.view, required this.path});
  File get file => File(path);

  /// The photo's bytes from a file path, blob URL or data URL.
  Future<Uint8List> readBytes() => XFile(path).readAsBytes();
  Map<String, dynamic> toJson() => {'view': view, 'path': path};
  factory ScanImage.fromJson(Map<String, dynamic> json) =>
      ScanImage(view: json['view'] as String, path: json['path'] as String);
}

/// Persists the full API response so future classes and fields survive restart.
/// No score, grade, severity or lifespan is computed on the client.
class ScanResult {
  final String id;
  final String userId;
  final String garmentName;
  final List<ScanImage> images;
  final DateTime dateScanned;
  final Map<String, dynamic> analysis;
  final Map<String, dynamic> userInputs;
  String notes;

  ScanResult(
      {required this.id,
      required this.userId,
      required this.garmentName,
      required this.images,
      required this.dateScanned,
      required this.analysis,
      this.userInputs = const {},
      this.notes = ''});

  File? get image => images.isEmpty ? null : images.first.file;
  String? get imagePath => images.isEmpty ? null : images.first.path;
  String get imageAssetFallback => fabricType;
  String get fabricType => analysis['fabric_type'] as String? ?? 'Unknown';
  bool get fabricUncertain => fabricType == 'uncertain';
  String get fabricLabel => fabricUncertain ? 'Fabric uncertain' : fabricType;
  int get confidencePercent =>
      (analysis['fabric_confidence'] as num? ?? 0).round();
  bool get hasFabricScore => analysis['fabric_confidence'] is num;
  String? get fabricModelName => analysis['fabric_model'] as String?;
  DamageDetectionResult get damageDetection =>
      DamageDetectionResult.fromJson(analysis['damages'] as List? ?? []);
  Map<String, bool> get damageReview =>
      Map<String, bool>.from(analysis['damage_review'] as Map? ?? {});
  bool get hasScientificAssessment => analysis['methodology_version'] is String;
  Map<String, dynamic> section(String key) =>
      Map<String, dynamic>.from(analysis[key] as Map? ?? const {});
  bool get hasHealthAssessment => section('condition_assessment').isNotEmpty;
  bool get hasDamageDetection => analysis.containsKey('damages');
  bool get hasConditionAssessment => hasHealthAssessment;
  String get healthRating =>
      section('condition_assessment')['category'] as String? ??
      'Assessment unavailable';
  String get healthStatusNote =>
      section('condition_assessment')['explanation'] as String? ??
      'Rescan the garment to get an assessment using the current method.';
  String get lifespanRange =>
      'Insufficient evidence for a reliable remaining lifespan estimate.';
  String get usageStatus => hasScientificAssessment
      ? analysis['usage_status'] as String? ?? 'Usage history not supplied.'
      : 'Usage history has no validated lifespan model.';
  Map<String, dynamic> get care => section('laundry_settings');
  String _care(String key) =>
      care[key] as String? ?? 'Check the garment care label';
  String get washMethod => laundryMode;
  String get dryMethod => _care('dry_method');
  String get ironLevel => _care('iron_level');
  String get bleachWarning => _care('bleach_warning');
  String get detergentSuggestion => _care('detergent');
  String get laundryMode => _care('laundry_mode');

  ScanResult withImages(List<ScanImage> value) => ScanResult(
      id: id,
      userId: userId,
      garmentName: garmentName,
      images: value,
      dateScanned: dateScanned,
      analysis: analysis,
      userInputs: userInputs,
      notes: notes);

  ScanResult withDamageReview(
          Map<String, dynamic> assessment, Map<String, bool> decisions) =>
      ScanResult(
          id: id,
          userId: userId,
          garmentName: garmentName,
          images: images,
          dateScanned: dateScanned,
          analysis: {...analysis, ...assessment, 'damage_review': decisions},
          userInputs: userInputs,
          notes: notes);

  Map<String, dynamic> toJson() => {
        'id': id,
        'user_id': userId,
        'garment_name': garmentName,
        'images': images.map((i) => i.toJson()).toList(),
        'timestamp': dateScanned.toIso8601String(),
        'analysis': analysis,
        'user_inputs': userInputs,
        'notes': notes
      };
  factory ScanResult.fromJson(Map<String, dynamic> json) => ScanResult(
        id: json['id'] as String,
        userId: json['user_id'] as String,
        garmentName: json['garment_name'] as String,
        images: (json['images'] as List)
            .map((i) => ScanImage.fromJson(Map<String, dynamic>.from(i as Map)))
            .toList(),
        dateScanned: DateTime.parse(json['timestamp'] as String),
        analysis: Map<String, dynamic>.from(json['analysis'] as Map),
        userInputs:
            Map<String, dynamic>.from(json['user_inputs'] as Map? ?? {}),
        notes: json['notes'] as String? ?? '',
      );
}
