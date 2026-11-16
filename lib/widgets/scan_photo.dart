import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// A scan photo from a file path on phones, or a blob/data URL on the web.
class ScanPhoto extends StatelessWidget {
  final String path;
  final BoxFit fit;
  final double? width;
  final double? height;
  final Widget? fallback;

  const ScanPhoto(
    this.path, {
    super.key,
    this.fit = BoxFit.cover,
    this.width,
    this.height,
    this.fallback,
  });

  /// True for photos that live at a URL rather than in the device file system.
  static bool isUrl(String path) =>
      kIsWeb ||
      path.startsWith('data:') ||
      path.startsWith('blob:') ||
      path.startsWith('http');

  @override
  Widget build(BuildContext context) {
    Widget error(BuildContext context, Object _, StackTrace? __) =>
        fallback ?? const SizedBox.shrink();
    if (isUrl(path)) {
      return Image.network(path,
          fit: fit, width: width, height: height, errorBuilder: error);
    }
    return Image.file(File(path),
        fit: fit, width: width, height: height, errorBuilder: error);
  }
}
