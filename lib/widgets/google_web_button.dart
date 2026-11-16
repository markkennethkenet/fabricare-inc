// Google's own sign-in button on the web; nothing on other platforms.
export 'google_web_button_stub.dart'
    if (dart.library.js_interop) 'google_web_button_web.dart';
