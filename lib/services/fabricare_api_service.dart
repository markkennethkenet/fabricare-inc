import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/damage_detection.dart';
import '../models/scan_result.dart';
import 'api_config.dart';

class ScanApiException implements Exception {
  final String message;
  const ScanApiException(this.message);

  @override
  String toString() => message;
}

class AnalyzeResponse {
  final Map<String, dynamic> json;
  const AnalyzeResponse(this.json);
  factory AnalyzeResponse.fromJson(Map<String, dynamic> json) =>
      AnalyzeResponse(json);
  String get fabricType => json['fabric_type'] as String;
  double? get fabricConfidence =>
      (json['fabric_confidence'] as num?)?.toDouble();
  DamageDetectionResult toDamageDetectionResult() =>
      DamageDetectionResult.fromJson(json['damages'] as List? ?? []);
  ScanResult toScanResult(List<ScanImage> images, String userId,
          {Map<String, dynamic> userInputs = const {}}) =>
      ScanResult(
          id: DateTime.now().microsecondsSinceEpoch.toString(),
          userId: userId,
          garmentName: '$fabricType Garment',
          images: images,
          dateScanned: DateTime.now(),
          analysis: json,
          userInputs: Map<String, dynamic>.from(
              json['user_inputs'] as Map? ?? userInputs));
}

class FabricCareApiService {
  FabricCareApiService._();
  static final FabricCareApiService instance = FabricCareApiService._();
  bool isReady = false;
  String? loadError;
  List<String> fabricLabels = [];
  List<String> damageLabels = [];

  Future<void> init() async {
    debugPrint('FabriCare API: ${ApiConfig.baseUrl}');
    try {
      final response = await http
          .get(Uri.parse('${ApiConfig.baseUrl}/'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception('API returned ${response.statusCode}');
      }
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      fabricLabels = List<String>.from(body['fabric_labels'] as List? ?? []);
      damageLabels = List<String>.from(body['damage_labels'] as List? ?? []);
      isReady = true;
      loadError = null;
    } catch (e) {
      isReady = false;
      loadError = e.toString();
    }
  }

  // Retain the original single-image API contract for existing callers.
  Future<AnalyzeResponse> analyze(File image) async => AnalyzeResponse.fromJson(
      await _upload('/analyze', [ScanImage(view: 'front', path: image.path)],
          legacy: true));

  Future<List<Map<String, dynamic>>> checkQuality(
      List<ScanImage> images) async {
    final body = await _upload('/quality', images);
    final entries = body['per_image'];
    if (entries is! List ||
        entries.length != images.length ||
        entries.any((entry) =>
            entry is! Map<String, dynamic> ||
            entry['view'] is! String ||
            entry['quality'] is! Map<String, dynamic> ||
            entry['quality']['passed'] is! bool ||
            entry['quality']['reason'] is! String)) {
      throw const ScanApiException(
          'The server returned an invalid photo quality result.');
    }
    return entries.cast<Map<String, dynamic>>();
  }

  Future<AnalyzeResponse> analyzeImages(List<ScanImage> images,
      {bool allowLowQuality = false,
      String? modelType,
      Map<String, dynamic> userInputs = const {}}) async {
    final body = await _upload('/predict/hybrid', images, fields: {
      'allow_low_quality': allowLowQuality.toString(),
      if (modelType != null) 'model_type': modelType,
      'user_inputs': jsonEncode(userInputs)
    });
    if (body['fabric_type'] is! String ||
        body['damages'] is! List ||
        body['user_inputs'] is! Map<String, dynamic>) {
      throw const ScanApiException(
          'The server returned an incomplete scan result.');
    }
    if (modelType != null &&
        (body['model_type'] != modelType || body['fabric_model'] is! String)) {
      throw const ScanApiException(
          'The server did not confirm the selected fabric model.');
    }
    return AnalyzeResponse.fromJson(body);
  }

  Future<Map<String, dynamic>> reviewDamage(
      ScanResult scan, Map<String, bool> decisions) async {
    final response = await http
        .post(
          Uri.parse('${ApiConfig.baseUrl}/review-damage'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'fabric_type': scan.fabricType,
            'fabric_confidence': scan.analysis['fabric_confidence'],
            'detector_available':
                scan.section('condition_assessment')['detector_available'] ??
                    true,
            'coverage_limited':
                scan.section('condition_assessment')['coverage_limited'] ??
                    false,
            'quality_limited':
                scan.section('condition_assessment')['quality_limited'] ??
                    scan.analysis['quality_override'] == true,
            'user_inputs': scan.userInputs,
            'findings': [
              for (final finding in scan.damageDetection.allItems)
                {
                  'id': finding.id,
                  'view': finding.view,
                  'type': finding.label,
                  'area_ratio': finding.areaRatio,
                  if (finding.bbox.length == 4) 'bbox': finding.bbox,
                  'confirmed': decisions[finding.id],
                }
            ],
          }),
        )
        .timeout(const Duration(seconds: 20));
    final result = _readResponse(response);
    if (result['condition_assessment'] is! Map ||
        result['care_guidance'] is! Map ||
        result['laundry_settings'] is! Map ||
        result['lifespan'] is! Map ||
        result['six_month_outlook'] is! Map ||
        result['risk_alerts'] is! List ||
        result['references'] is! List) {
      throw const ScanApiException(
          'The server returned an incomplete damage review.');
    }
    return result;
  }

  Future<Map<String, dynamic>> _upload(String endpoint, List<ScanImage> images,
      {bool legacy = false, Map<String, String> fields = const {}}) async {
    final request = http.MultipartRequest(
        'POST', Uri.parse('${ApiConfig.baseUrl}$endpoint'));
    request.fields.addAll(fields);
    if (!legacy) {
      request.fields['views'] = jsonEncode(images.map((i) => i.view).toList());
    }
    try {
      for (final image in images) {
        request.files.add(await http.MultipartFile.fromPath(
            legacy ? 'image' : 'images', image.path));
      }
    } on FileSystemException {
      throw const ScanApiException(
          'A selected photo could not be read. Select it again.');
    }
    final client = http.Client();
    final timer = Stopwatch()..start();
    final step =
        endpoint == '/quality' ? 'Photo quality check' : 'Scan analysis';
    debugPrint(
        'FabriCare POST ${request.url} photos=${images.length} bytes=${request.contentLength}');
    try {
      final response = await client
          .send(request)
          .then(http.Response.fromStream)
          .timeout(Duration(seconds: endpoint == '/quality' ? 30 : 120));
      debugPrint(
          'FabriCare POST $endpoint HTTP ${response.statusCode} in ${timer.elapsedMilliseconds} ms');
      return _readResponse(response);
    } on TimeoutException {
      throw ScanApiException('$step timed out. Try again with fewer photos.');
    } on http.ClientException catch (error) {
      final details = error.message.toLowerCase();
      if (details.contains('cleartext') || details.contains('insecure http')) {
        throw const ScanApiException(
            'Android blocked the HTTP connection to the scan server.');
      }
      throw ScanApiException(
          'Could not reach the scan server at ${request.url.authority}.');
    } on SocketException {
      throw ScanApiException(
          'Could not reach the scan server at ${request.url.authority}.');
    } on FileSystemException {
      throw const ScanApiException(
          'A selected photo could not be read. Select it again.');
    } finally {
      client.close();
    }
  }

  Map<String, dynamic> _readResponse(http.Response response) {
    Object? body;
    try {
      body = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      // A proxy or older deployment may return HTML for an HTTP error.
    }
    final status = response.statusCode;
    if (status == 413) {
      throw const ScanApiException(
          'Photos exceed the upload limit. Use smaller or fewer photos.');
    }
    if (status == 404 || status == 405) {
      throw ScanApiException(
          'This server does not support the requested scan operation (HTTP $status). Check the API address and backend version.');
    }
    if (status != 200) {
      final message = body is Map<String, dynamic> ? body['error'] : null;
      throw ScanApiException(message is String && message.isNotEmpty
          ? '$message (HTTP $status)'
          : 'The scan server returned HTTP $status. Try again.');
    }
    if (body is! Map<String, dynamic>) {
      throw const ScanApiException(
          'The server returned an unexpected response. Check the API address.');
    }
    return body;
  }
}
