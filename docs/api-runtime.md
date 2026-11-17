# Model compatible API runtime

Use the isolated CPython 3.12 environment at `.venv-api` for the API. The old
parent `.venv` uses Python 3.14 and scikit-learn 1.9.0; it is preserved, but it
must not be used to serve the existing classifier.

The saved Decision Tree requires scikit-learn 1.6.1. This matches the model's
embedded version without re-saving, converting, modifying, or retraining it.
Python 3.12 has an official 1.6.1 Windows wheel in the
[release metadata](https://pypi.org/project/scikit-learn/1.6.1/#files).
Scikit-learn recommends loading persisted models with the same package version;
cross-version loading is unsupported. See its
[model persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

## Setup and start

Run these PowerShell commands from the `fabricare` repository:

```powershell
.\scripts\setup_api.ps1
.\.venv-api\Scripts\python.exe .\fabricare_api.py
```

For a USB-connected Android phone, the existing startup command now uses the
same environment:

```powershell
.\scripts\start_dev.ps1
```

The setup script accepts `-Python312 'C:\path\to\python.exe'` if Python 3.12 is
installed elsewhere. It validates the interpreter, installs binary wheels, and
runs `pip check`. It uses `requirements-api.lock.txt` when present, with
`requirements.txt` as the initial-resolution fallback. The environment is ignored
by Git. The existing `.env` remains unchanged.

`requirements.txt` pins scikit-learn 1.6.1, NumPy 2.2.6, and SciPy 1.15.3 for
the compatible numerical stack. The core Flask, YOLO, PyTorch, image-processing,
and database-client versions are pinned to the versions already used by the
previous API smoke test. The lock file also pins the resolved transitive
dependencies for repeatable Windows/Python 3.12 installs.

## Choose a fabric model on Android

The scanner defaults to CNN — MobileNetV2. Open **Choose Fabric Identification
Model** before adding the front photo to select CNN, GLCM + SVM, or
GLCM + Decision Tree. See the [scientific basis](FABRICARE_SCIENTIFIC_BASIS.md)
for the evidence behind the result sections.

From the FabriCare repository, start XAMPP MySQL on the port in the local
`.env` (3307 in the development setup). Connect the Android phone over USB
and enable USB debugging. Run these commands in separate PowerShell windows.

```powershell
# Window 1, keep open
Set-Location C:\flutter-project\fabricare
$env:Path = 'C:\Android\Sdk\platform-tools;' + $env:Path
.\scripts\start_dev.ps1
```

```powershell
# Window 2
Set-Location C:\flutter-project\fabricare
& C:\Android\Sdk\platform-tools\adb.exe devices
flutter run -d <device-id> --no-pub --dart-define=API_BASE_URL=http://127.0.0.1:5000
```

Use the serial shown by `adb devices` in place of `<device-id>`. Use an account
in the local database. Scan the same clear garment photo once with each model.
Check the model name, fabric, damage findings, condition, and care on each
result. Reopen the saved scans from History and check each model name under
Overview. The SVM should show that prediction confidence is unavailable.

`POST /predict/hybrid` and `POST /analyze` accept the optional multipart
`model_type` field (`cnn`, `svm`, or `glcm_dt`). The response records the
actual `model_type` and `fabric_model` along with the existing assessment.
Unknown values receive HTTP 400. Unavailable or failed selected models receive
HTTP 503 with a model-specific error. Requests that omit the field continue to
use the server's `FABRIC_CLASSIFIER` setting. The previous artifacts and
server modes remain available for clients that use that setting.

The new artifacts are `model/glcm_models/glcm_svm.joblib` and
`model/glcm_models/glcm_decision_tree.joblib`. They share 72 GLCM features from
EXIF-corrected, padded RGB images. The SVM pipeline applies its saved scaler
internally. The CNN uses `model/mobilenetv2_6class/mobilenetv2_v3_refit.keras`
and its adjacent `class_names.json`. Scan history saves the response and
timestamp in Hive. MySQL continues to store accounts and sessions.

## Trained model fabric and damage results

### Android LAN scans

Start the local database and run `.\scripts\start_dev.ps1 -WiFi` on the PC.
Keep both devices on the same trusted Wi-Fi. Check the PC's Wi-Fi IPv4 address
with `ipconfig` before building the APK. On the phone, open
`http://<PC Wi-Fi IPv4>:5000/health/ready` and check for `"ready":true`.
If the PC's address changes, rebuild the APK with the new address.

Build the APK with the same address the phone uses to reach the backend.
For the current LAN address, run:

```powershell
.\scripts\build_android_release.ps1 -ApiBaseUrl 'http://192.168.1.48:5000'
```

The default release build uses the hosted API. Opening the LAN URL in a phone
browser does not change the address compiled into an installed APK.
`adb logcat -s flutter` shows the effective API address and the scan request
route, upload bytes, HTTP status, and timing. The backend logs the same scan
routes and HTTP outcomes. A scan sends the photos to `/quality` and then to
`/predict/hybrid`. Both accept JPEG or PNG, with a combined multipart request
limit of 20 MiB. The scan endpoint returns trained-model results.

Flutter submits garment images to `POST /predict/hybrid`. The endpoint retains
the `/analyze` response fields and adds trained-model `fabric` and `damage`
results, `care_recommendations`, and `analysis_status`. The endpoint runs only
the configured local fabric classifier and damage detector. Fabric and damage
results are decided independently, and scan analysis does not require a
third-party service or API key.

Set these variables in the backend environment or local `.env` (never in Flutter):

```dotenv
HYBRID_ML_CONFIDENCE_THRESHOLD=85
# Optional: replaces the built-in Google client IDs; empty turns Google sign-in off.
# GOOGLE_CLIENT_IDS=<comma-separated OAuth client IDs>
```

The V4 CNN sets its own `review_threshold_percent` (60) in
`model/fabric_cnn/model_info.json`. It was trained with label smoothing 0.1, which
caps a six-class score near 92%, so 85% flagged almost every confident scan for
another photo. Calibrating that value on the validation photos is still recommended.

### Care advice

`ai_care` holds the trained care model's laundry mode plus, written for the scan,
`care_tips` and `laundry_tips` (at most three sentences each, with reference ids),
a `headline` (the care label's wash method when the user entered one) and
`provided_by`. `care_model.care_tips` and `care_model.laundry_tips` choose the
sentences from the detected damage, fabric, colour, care label and wear and washing
habits. `/review-damage` refreshes `ai_care` when the app sends the scan's saved
`care_features`. The local Qwen `/vision-opinion` route was removed, and the app no
longer asks for the GLCM SVM second opinion.

With `GEMINI_API_KEY` set, `gemini_care.py` asks Gemini (`GEMINI_MODEL`, default
`gemini-3.5-flash-lite`, free tier) to reword those tips for the garment from the scan results
and, unless `GEMINI_SEND_PHOTO=false`, a 768-pixel copy of the front photo. The card's headline (the care model's mode after
the care label and the unconfirmed-fabric check), the care label and the referenced tips are sent
as facts it must follow; it may not suggest a different wash method. `ai_care` then
carries Gemini's sentences, `writer`, a `provided_by` ending in "· worded by Gemini", and the
original tips under `reference_tips`. No key, an 8-second timeout, a used-up quota (HTTP 429)
or a malformed answer keeps the referenced tips. Google may use free-tier requests to improve
its products.

### Google sign-in

`POST /auth/google` takes `{"id_token": ...}` from Google Sign-In, verifies its
signature, expiry, issuer and audience (FabriCare's web and Android client IDs, or
`GOOGLE_CLIENT_IDS`) with `google-auth`, and
requires a verified email. A new email creates an account named after the Google
profile with an unusable random password hash; an existing email signs in to that
account, so its saved scans stay. The response matches `/auth/login` plus `created`
and `picture`. The app offers Google sign-in only; `/auth/register` and `/auth/login`
remain for older app builds. The app has the web client ID built in
(`ApiConfig.googleWebClientId`; `--dart-define=GOOGLE_WEB_CLIENT_ID` overrides it).
The Android OAuth client is registered for package `com.example.fabricare` and the
SHA-1 of the debug key that signs the APKs; a different signing key needs its SHA-1
added. The client secret is not used: the app only needs ID tokens.

The 85% threshold applies to the GLCM models and is an experimental decision rule. The Decision Tree's class
vote scores are uncalibrated and must not be read as probabilities. The legacy
12-feature Decision Tree supports five fabrics. All three scanner choices support
cotton, denim, linen, polyester, silk, and wool. Model
labels come from the active model. Denim names a construction, and a photograph
cannot reliably establish fiber composition.

The existing `best.pt` YOLO model supplies trained damage boxes for `hole`,
`needlemark`, `pilling`, and `stain`. Damage detections are suggestions for user
review. Existing scan history and images are stored locally by Flutter in Hive;
MySQL currently stores accounts and auth tokens, so no scan-history database
migration is performed.

The API accepts the same multipart `image` or `images`/`views` input and
`user_inputs` fields as `/analyze`. Use `allow_low_quality=true` to explicitly
continue past a failed quality check. Both analysis routes return
`schema_version: 3`, scientific provenance, and the versioned condition and care
sections. Read the [scientific basis](FABRICARE_SCIENTIFIC_BASIS.md) for
legacy field compatibility and label input options.

For a defense rehearsal with no internet, keep the phone connected to the
local API (for USB, use `adb reverse tcp:5000 tcp:5000`). Scans use only the
configured local trained models and appear in History.
`GET /` serves model metadata without waiting for MySQL. Use
`GET /health/ready` to verify MySQL availability before rehearsing sign-in.

## Runtime protection and verification

The default classifier is the original-data V3 MobileNetV2 refit in
`model/mobilenetv2_6class/`. Keep `mobilenetv2_v3_refit.keras` with its
`class_names.json` on the backend. Each photo is corrected for EXIF orientation,
converted to RGB, and resized with its aspect ratio preserved and gray (127)
padding to 224 × 224. Pixel values stay in the 0–255 range for the model's
internal rescaling layer. API scores describe the model output for the scan.
The refit reached 72.85% validation accuracy, 81.87% test accuracy, and
0.8221 test macro-F1 in the V3 comparison. See the [dataset report](SIX_CLASS_DATASET.md)
for the test split and the polyester results.

### V4 models

`model/fabric_cnn/` holds the V4 ConvNeXt-Tiny CNN (86.69% test accuracy, macro-F1 0.863) as an architecture file
with 16-bit weights, which keeps it under GitHub's 100 MB file limit. The API loads it in place of
`model/mobilenetv2_6class/` whenever `model/fabric_cnn/model_info.json` exists; delete that folder to fall back to
the V3 MobileNetV2. `model/glcm_models/` holds the V4 GLCM SVM (now with confidence scores) and Decision Tree, with
their test results in `metrics.json`; the V3 versions are in `v3_backup/`. `model/care/care_recommender.joblib` is the
selected care model (Decision Tree). Install a new export with
`.venv-api\Scripts\python.exe scripts\install_v4_models.py <export.zip>`. Results for every model are in
[OBJECTIVES.md](OBJECTIVES.md) and `model/v4_training_report.json`.

### Mirror averaging and retake stability

The CNN scores each photo and its horizontal mirror, then averages the two
scores. The V3 model trained with `RandomFlip("horizontal")`, so the mirror is
an in-distribution view. Averaging reduces score swings between retakes of the
same garment. It adds no crops or zooms that the model never saw in training.
Each `per_image` entry reports `inference_views`. Set `FABRIC_TTA=off` and
restart to score a single view as before.

Measure accuracy and retake stability with `scripts/evaluate_fabric_scans.py`.
It runs the scanner's own preprocessing and reports single-view and
mirror-averaged accuracy, macro-F1, and per-class recall. It also simulates
retakes of each photo with a new crop, light level, tilt, and JPEG compression,
and reports how often the label stays the same.

```powershell
# V3 held-out split
.\.venv-api\Scripts\python.exe .\scripts\evaluate_fabric_scans.py `
  --raw "$dataset\raw" --manifest "$dataset\reports\split_manifest_70_20_10.csv" `
  --output research\output\scan_stability.json

# Your own phone scans in phone_scans\<fabric>\*.jpg
.\.venv-api\Scripts\python.exe .\scripts\evaluate_fabric_scans.py --folder C:\path\to\phone_scans
```

Judge the 80% target on independent phone scans of real garments. Simulated
retakes check stability; they do not replace a real phone test set.

The app's smart camera supports the same goal. It checks light, motion, focus,
and framing live on the phone, guides the user closer or back, auto-captures
when the frame is ready, and crops the photo to the on-screen guide box before
upload.

Each scan accepts views of one garment. When clear photos are available, fabric
identification uses those photos. A close-up that disagrees with the clear
whole-garment views is excluded from fabric identification. Clear front and back
photos with different predicted fabrics produce an uncertain result.

Set `FABRIC_CLASSIFIER=decision_tree` and restart to roll back to
`model/best_decision_tree.joblib`. The other selectable classifiers are
`mobilenet_legacy` (the previous 160 × 160 export), `legacy`, and `ensemble`.
Both `/analyze` and `/predict/hybrid` use the selected classifier. Flutter uses
`/predict/hybrid` for scans.

To run the Colab ensemble locally, keep `inference_config.json` and both
`.keras` files from the validation development export in
`model/validation_development/`. The configuration supplies the class order
and the 75% EfficientNetB0 / 25% MobileNetV2 weights.

```powershell
.\scripts\setup_api.ps1 -Ensemble
.\scripts\start_dev.ps1 -Ensemble
```

For a direct API launch or HTTP smoke check, select the ensemble in the same
PowerShell session before running the command.

```powershell
$env:FABRIC_CLASSIFIER = 'ensemble'
.\.venv-api\Scripts\python.exe .\scripts\check_api.py
.\.venv-api\Scripts\python.exe .\fabricare_api.py
```

Build or run the USB Android app with
`--dart-define=API_BASE_URL=http://127.0.0.1:5000`. The API root reports the
active classifier name and labels. Restart the backend after switching models.

With the API running, set `SCAN_TEST_PHOTO` to an absolute path to
a sharp, well-lit image through Flutter's `--dart-define` argument, and pass
`--dart-define=API_BASE_URL=http://127.0.0.1:5000` to
`flutter test test/scan_live_api_test.dart`. This check supplies the photo at
the picker boundary and verifies the actual HTTP calls, local history write,
and rendered result. For a Decision Tree rollback, also pass
`--dart-define=SCAN_TEST_CLASSIFIER=decision_tree`. Use independent garment
photos to measure accuracy.

`model_runtime.py` treats `InconsistentVersionWarning` as a startup error and
points to the correct setup command. It neither suppresses the warning nor
changes the serialized estimator to claim another version. `GET /` includes
`runtime.python` and `runtime.scikit_learn` so the running service can be checked.

```powershell
.\.venv-api\Scripts\python.exe -m pip check
.\.venv-api\Scripts\python.exe -m unittest test_assessment test_image_processing test_api_contract test_model_runtime -v
.\.venv-api\Scripts\python.exe .\scripts\check_api.py
```

For a predeployment gate that also requires the configured database to be
reachable, add `--require-database`. The script exits with an error if the
analysis checks pass but MySQL health is false.

The HTTP check starts a temporary loopback server with the real unchanged
models, then checks health/runtime metadata, legacy single-image analysis,
three-view analysis and garment inputs, quality warnings, explicit override,
and missing-image rejection. It verifies model hashes did not change and shuts
down the server afterward. It only reads database connectivity through the
health route; it does not create users, update garments, or initialize schemas.

Synthetic smoke-test photos verify runtime wiring and response contracts. They
do not establish model accuracy on garments or replace physical-phone testing.

## Verified results on October 20 2026

- Python 3.12.6, scikit-learn 1.6.1, NumPy 2.2.6, and SciPy 1.15.3.
- `pip check`: no broken requirements.
- All 19 backend tests passed, including the new model-version guard tests.
- Real HTTP checks with the existing models passed: health returned 200,
  legacy single-image and three-view analysis returned 200, quality preflight
  returned 200, unconfirmed low-quality input returned 422, explicit override
  returned 200, and missing images returned 400.
- No sklearn version warning occurred during real model loading or inference.
- Model and label hashes matched before and after inference; no retraining or
  model serialization was performed.
- `scripts/setup_api.ps1` was rerun against the generated lock and completed
  successfully with no dependency changes needed.
- `scripts/check_api.py --require-database` completed the same successful model
  and HTTP checks, then exited with status 1 because MySQL health was false.

Full predeployment readiness remains blocked by the local MySQL connection.
The configured loopback port is 3307 and connection fails with error 2003.
The running MySQL80 service listens on port 3306, but a read-only attempt there
with the existing credentials is rejected with error 1045. The configuration and
credentials were not changed. Confirm the intended instance and update its local
`.env` connection details; do not put passwords in chat, reports, or Git. The HTTP
test reports `database_connected: false` and `predeployment_ready: false` until
that is resolved. No database data or schema was modified by these checks.
