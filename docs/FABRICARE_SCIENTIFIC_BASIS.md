# FabricCare AI scientific basis and validation

Methodology version `visible-condition-and-label-care/2.0`

Scan API schema `3`

## Method 2.0 changes (study objectives 2, 4 and 5)

Version 2.0 adds three outputs that version 1.2 withheld. Each is labelled with its basis, and every number
can be traced to a source or a named assumption.

- **Health score.** FHS = 100 × (1 − Σ weights of the damage types present) (`assessment.health_assessment`). The
  weights come from an expert AHP survey (Saaty, 1980; `ahp.py`) and live in `assessment_config.json`; grades use the
  health index bands of Jahromi et al. (2009). Candidate findings below the detector thresholds in
  `care_model.DAMAGE_MIN_CONFIDENCE` are listed for review but do not lower the score.
- **Remaining lifespan.** A range in months (`assessment.lifespan_estimate`), with status `experimental_estimate`. It
  starts from the garment-lifespan regression in Laitala and Klepp (2020), reference D, §4.1: an expected total
  lifespan of about 5.2 years; fibre offsets relative to cotton (silk +2.2 years, wool +1.2 years, synthetics
  +5.5 months); −0.018 years per extra wear per year; and +0.792 years per laundering bracket (up to +4.8 years for
  washing less often). The garment's age is subtracted, and the result is scaled by the Fabric Health Score, whose
  damage weights come from the expert AHP survey (`ahp.py`).
  That regression explains about 22% of the variation, so the app shows a ±40% range. This estimate is not a
  validated prediction for an individual garment.
- **AI laundry-mode suggestion and alerts.** A Decision Tree, chosen against a Random Forest and an SVM, reproduces
  the referenced care rules in `care_model.py` from the fabric, the detected condition and the image features. Its
  agreement with those rules is reported in [OBJECTIVES.md](OBJECTIVES.md). An entered care label always takes
  precedence.

The sections below describe the version 1.2 audit and remain the basis for condition and label-based care.

## Repository audit before implementation

- **Fabric identification was supported.** The Flutter scan flow uploaded a front photo and optional back and close-up views. Flask ran one of three selectable classifiers, using MobileNetV2, GLCM + SVM, or GLCM + Decision Tree. The previous classifier modes and bundled inactive TFLite artifact also existed. These classifiers identify fabric appearance classes. Denim is a construction and the classifier cannot establish blend percentages, coatings, finishes, linings or trim.
- **Visible defect detection was partially supported.** The deployed `model/best.pt` YOLO artifact has four labels, `hole`, `needlemark`, `pilling` and `stain`. Its SHA-256 is `83f828fb58f6aebd0e1d3e7f9f4ff31022ecc4cf8bce3800542068393d51bcc7`. The six-class experimental damage dataset also includes missing yarn and wrinkles. The deployed artifact has no such outputs. Independent phone-photo validation of the deployed detector was unavailable. See [damage dataset provenance](damage-yolov8-dataset.md) and [fabric dataset provenance](SIX_CLASS_DATASET.md).
- **Whole-garment health and severity were unsupported.** The prior backend treated bounding boxes under 2% of the photo as minor and boxes over 8% as severe. It subtracted type and severity deductions from 100 and mapped the result to four grades. The values lacked expert calibration and depended on crop, zoom, detector errors and duplicate views.
- **Remaining lifespan and six-month condition were unsupported.** The previous rules chose assumed lifespans for each fabric, scaled them by the health score, subtracted a fraction of calendar age and adjusted by researcher-chosen wear multipliers. The same assumed baseline drove a six-month score decline. The scan path was deterministic. No measured service-life outcomes, longitudinal forecast validation or evaluated lifespan model were found in the active repo, model artifacts or prepared dataset descriptions.
- **Care recommendations, laundry and risks were partially implemented.** Previous fabric-class tables and defect addenda included wash modes, temperature ranges, drying and ironing without garment-specific restrictions. The separate OCR screen returned fixed example text for any photo. The active scan path used local inference and local rules. No independently labeled care dataset or evaluated care model was found.
- **Persistence and endpoints.** Flutter screens cover onboarding, auth, scan, results, history, garment profile and general tips. The existing Flask endpoints are `GET /`, `GET /health/ready`, `POST /quality`, `POST /analyze`, `POST /predict/hybrid`, `POST /review-damage` and the `/auth` routes for registration, sign-in, sign-out and account data. Hive stores complete scan responses plus dated image paths on the device, scoped by account. The configured MySQL or PostgreSQL service stores accounts and auth tokens. There are no scan-history rows in the server database.

## Revision audit

Git commit `023f0ce` stores the previous scoring and projection functions and their configuration. The old score was `clamp(round(100 − Σ(type deduction × box severity multiplier)), 0, 100)`. It used a default deduction of 10 and multipliers of 0.5, 1, and 1.5 from photo-area bands of under 2%, 2–8%, and over 8%. The old remaining time was `round(min(1.5 × baseline, max(0, baseline × score / 100 − 0.5 × age in months) / wear), 1)`. The old forecast was `max(0, round(score − 100 × wear × 6 / baseline))`. The configuration assumed 24 months as a general baseline and 36 for denim, and multiplied use, washing, condition, and maximum severity factors into `wear`. These formulas appear here for traceability. The project has no evaluated health rubric or longitudinal end-of-use data to justify their parameters. Relabeling them experimental would still anchor users to unsupported garment timelines.

Version 1.0 suppressed the numeric fields, withheld class advice below the score cutoff, and often returned a label check plus unknown wash settings. It also displayed evidence codes and long reasoning inline. Version 1.1 adds finding locations, qualitative significance, durability guidance, partial review, conditional advice for uncertain classes, and shorter summaries with expandable reasoning. The numeric fields stay null.

## Implemented method

### Photo and model inputs

JPEG and PNG uploads are decoded and checked for dimensions and size. The API accepts a legacy `image` upload or repeated `images` with a matching `views` array. The existing quality checks use grayscale mean brightness and variance of the grayscale Laplacian. Warning cutoffs are brightness below 45 or above 220 and blur variance below 60. These are operational photo checks. Users can continue past a warning and the limitations are carried into the result.

Fabric classification preserves the original artifacts and preprocessing. When models return scores, the existing view aggregation averages each class's included view scores and selects the highest, reporting its mean on a 0–100 scale. When no scores are supported, the app reports the class from the selected model or a multiview vote with confidence unavailable. Front/back disagreement yields an uncertain result. Two usable agreeing front/back views take precedence over a dissenting close-up. Scores describe the model's output and are uncalibrated. The configurable 85% review cutoff is an experimental application policy. When the score is unavailable or below the cutoff or photo quality is limited, class-specific precautions start with an explicit material condition. Recognized user-reported fiber content can supply the basis for general guidance. Garment-label instructions govern all treatment permissions.

The YOLO detector retains its previous box preprocessing and multiscale and tile rules. Inference uses a base confidence floor of 0.25 and a tile floor of 0.40. Extra stain and hole passes may be skipped when an existing box for the same class reaches 80%. Oversized hole proposals and tile-spanning proposals are filtered as documented in `assessment_config.json`. Same-class detections in the same image with intersection-over-union at least 0.50 are deduplicated by retaining the higher-scoring box. Front, back and close-up are different projections and can contain the same physical finding. The UI treats all boxes as candidates for review. If the damage artifact or an inference pass fails, classification can continue and the result records unavailable or partial detector coverage.

### Visible condition

The category follows the saved evidence. A user-confirmed opening takes priority, followed by confirmed pilling, then confirmed surface marks. Pending observations remain candidates even after another finding is reviewed. With no accepted or pending findings, the assessment distinguishes a completed review, a scan with no candidates, and unavailable detector coverage. A confirmed mark can be cosmetic. An opening prompts a repair check. The physical severity of each finding needs a separate assessment [A, B, I, L].

Box coverage is `(box width ÷ photo width) × (box height ÷ photo height)`. The location uses the box center. With normalized coordinates, column = `min(2, floor(3 × (x + width/2)))` and row = `min(2, floor(3 × (y + height/2)))`. The resulting region describes the photo, including the named view. The calculation does not measure garment-wide damage, material strength, or severity.

### Remaining lifespan and six-month outlook

A confirmed opening prompts a repair check. Confirmed pilling prompts friction management and repeat photos. Surface marks prompt monitoring, candidates prompt review, and scans without observations prompt a baseline photo and seam check. Reliable remaining months require dated wear, cleaning and repair records, service-life outcomes, and an independently evaluated model [C, D, I].

The six-month care scenario uses confirmed findings and the reported wear frequency, wash frequency, and optional cleaning practice. Repeated rubbing can make pilling more visible. Catching or stretching a confirmed opening can enlarge it. With frequent washing or reported rubbing or wringing, the scenario recommends checking the same areas across scans and using label-compatible handling. It reports its assumptions and allows manual comparison with a dated front photo from the user's history. Forecast validation requires dated repeat observations and a held-out evaluation [A, D, F, G, J, L].

### Care, laundry and risk

Each recommendation records an action, reason, evidence that selected the rule, restrictions where needed, and source IDs. The Flutter results display all recommendation actions and show their references by title in **Why this result?**. Advanced details display the method version and technical evidence codes. Material guidance uses the exact supported user-reported fiber when recognized, otherwise the predicted class. A weak or missing score and limited photo quality trigger **Needs confirmation** and conditional material wording. Material rules include sorting colored cotton, reducing dye transfer from dark denim, checking ordinary linen creases, avoiding excessive heat on polyester, protecting silk from rubbing or wet spot treatment, and reducing abrasion on wool [E, G, J, K, M, P].

A possible stain can trigger a precaution to hold off strong treatment while it is checked. A user-confirmed opening or pilling can trigger a more targeted repair or friction alert. Saving review decisions replaces both structured and compatibility care outputs. Dismissed findings are excluded from the assessment and unresolved findings remain marked as candidates [A, B, I, J, L, N].

Laundry guidance separates entered label instructions, general precautions, and unknown or restricted settings. Dry-clean-only, professional-only, do-not-wash, and conflicting entries withhold home-washing guidance. Mild machine wash, hand wash and other processes require the corresponding label entry. An entered Celsius maximum is treated as an upper limit. Unknown wash permission cannot authorize machine washing. Detergent for washable pure wool is suggested when pure wool is reported. A predicted wool garment with wash permission receives conditional advice to verify the fiber before selecting a wool detergent. Explicit prohibitions remain visible. The rules assign no heat setting or dryer permission from fabric-class predictions [E, G, H].

When a washing label is missing or unreadable, FabricCare gives precautions based on the garment and its findings. For a stain, it advises avoiding hot water and strong treatments while checking what the mark is and what cleaning method is suitable. A confirmed opening adds a repair check. When the material suggests cotton, detector coverage is complete, and no opening is flagged, the app provides a conditional note for plain cotton T-shirts whose machine washability has been established independently. Cotton et al. compared a 25 °C, 30-minute machine program with a 40 °C, 85-minute program and observed less colour loss and dye transfer with the cooler, shorter program. Those temperatures and times describe experimental conditions. They are not app settings or validated limits for other garments. Temperature and program duration changed together [N, Q, R].

A proposed local validation study should start with defined garment groups, such as plain cotton T-shirts without linings or embellishments. Record the verified composition and construction, then withhold label details from the model. Test multiple independent garments using recorded wash and drying procedures. Compare dimensional change, colour change or transfer, visible wear, and cleaning effectiveness. Agree on acceptable limits before testing and include garments with damage in a separately assessed group. Use ISO 6330 to define the wash and dry procedure, ISO 5077 for dimensional change, and ISO 15487 for appearance where the study can follow those methods. A wash trial can support a conditional cleaning recommendation within the tested group. Remaining calendar life still requires follow-up on garments in use [A, C, F, Q].

## Traceability

| Feature | Input | Calculation, rule or model | Supporting source | Output | Validation status |
| --- | --- | --- | --- | --- | --- |
| Fabric classification | Photo views, model selection | Existing classifier and view aggregation. Experimental score cutoff for review | Saved model artifacts and dataset documentation | Fabric class, model score when available, uncertainty | Inference smoke tested with saved artifacts. App-wide phone performance remains unverified |
| Fabric health | Detector boxes, quality, user review | Retain supported candidates, deduplicate, and assign qualitative category by evidence | A, B, I and deployed artifact | Observations, photo extent, provenance, severity unavailable | Rule tested. Expert photo rubric and independent detector validation pending |
| Remaining lifespan | Optional history, photo condition, evidence availability | Withhold numeric estimate without observed service outcomes and evaluated prediction model | C, D, I | Explicit unavailable status and evidence needed | Missing-data behavior tested. No lifespan prediction validated |
| Six-month outlook | Photo condition, wear and wash frequency, earlier dated scan when selected | Conditional guidance and manual comparison of earlier front photos | A, D, F, J, L | Assumptions, possible concerns and forecast unavailable | Qualitative rules tested. No quantitative forecast validated |
| Care recommendations | Fabric class and score, label, composition, reviewed findings | Label first. Conditional material and finding guidance | B, E, G, H, J, K | Action, reason, restrictions and references | Rule tested. Expert evaluation and labeled ML comparison pending |
| Laundry suggestions | User-transcribed restrictions and optional composition | Map label entries to process and limits. Withhold unknown or conflicting settings | E, G, H | Wash mode, maximum if known, detergent, drying, ironing, bleach and professional-care guidance | Conflict and rule behavior tested. Correct transcription depends on the user |
| Risk alerts | Reviewed findings, material uncertainty, label gaps or limits | Trigger relevant precautions only | B, E, G, I, J, L | Risk reason and precaution | Rule tested. No risk probabilities estimated |

## References and scope

The ISO references below were used through their public abstracts or scope previews. The app does not perform controlled ISO testing.

- **A** describes the scope of appearance assessment after controlled domestic washing and drying. [ISO 15487:2018](https://www.iso.org/standard/67602.html)
- **B** describes standardized visual assessment of pilling, fuzzing and matting. [ISO 12945-4:2020](https://www.iso.org/standard/75377.html)
- **C** explains service duration, wears, cleaning cycles and other lifespan measures. [Klepp, Laitala and Wiedemann (2020)](https://doi.org/10.3390/su12156219), [publisher full text](https://www.mdpi.com/2071-1050/12/15/6219)
- **D** relates garment characteristics and behavior to lifespan using wardrobe survey data. [Laitala and Klepp (2020)](https://doi.org/10.3390/su12219151), [publisher full text](https://www.mdpi.com/2071-1050/12/21/9151)
- **E** defines garment care labelling symbols for treatment limits. [ISO 3758:2023](https://www.iso.org/standard/74401.html)
- **F** describes controlled domestic washing and drying procedures for textile testing and informs the design of repeat-observation studies. [ISO 6330:2021](https://www.iso.org/standard/75934.html)
- **G** explains wash, bleach, dry, iron and professional care symbols and restrictions. [GINETEX care labelling](https://www.ginetex.net/gb/labelling/care-labelling/), [care-symbol definitions](https://www.ginetex.net/gb/labelling/care-symbols.asp)
- **H** explains care-label permission and detergent guidance for washable wool. [Woolmark washing guidance](https://www.woolmark.com/care/how-to-wash-wool/)
- **I** describes strip testing for tensile maximum force and elongation. It establishes the need for physical measurements for strength claims. [ISO 13934-1:2013](https://www.iso.org/standard/60676.html)
- **J** explains how rubbing contributes to pilling in wool garments. [Woolmark pilling guidance](https://www.woolmark.com/care/pilling/)
- **K** gives manufacturer advice for preserving the fit and colour of its denim. [Levi’s denim care](https://help.levi.com/hc/en-us/articles/15463553339917-Denim-care-and-washing)

- **L** describes a laboratory method for measuring the force needed to extend a tear started at a cut. It provides a physical basis for the precaution about pulling on openings. [ISO 13937-1:2000](https://www.iso.org/standard/23369.html)

- **M** describes fiber properties including linen creasing, polyester heat sensitivity, and wet-care risks for silk. [UGA Extension](https://site.extension.uga.edu/textiles/textile-basics/understand-your-fibers/)
- **N** describes using stain and fiber information before deciding on a treatment. [Oregon State University Extension](https://extension.oregonstate.edu/sites/extd8/files/2023-07/Oh%2C%20No%21%20Stains%21.pdf)
- **P** discusses dye transfer from dark and raw denim. [Levi’s](https://help.levi.com/hc/en-us/articles/34797976189709-Dark-denim-and-dye-transfer)
- **Q** compares colour loss and dye transfer under two machine programs on retail T-shirts. [Cotton et al. (2020)](https://eprints.whiterose.ac.uk/id/eprint/154480/)
- **R** describes water-related risks to dyes, weak textiles and attachments in a conservation context. It supports caution about unknown components. [Canadian Conservation Institute](https://www.canada.ca/en/conservation-institute/services/preventive-conservation/guidelines-collections/textiles-costumes.html)

## Compatibility and verification

The API returns structured condition, lifespan, outlook, recommendations, laundry settings, alerts and reference records. Every scan response includes a methodology version. Existing numeric score and projection keys return null. The old categorical and care fields are mapped to the new method where applicable. `/review-damage` recomputes the assessment from accepted and pending findings and returns their separate provenance. Dismissed findings are excluded from the assessment. The Flutter client sends the original care inputs, photo limitations and fabric score on review. It stores the full response in Hive. Legacy data remain available as raw records, while the current client withholds old numeric claims and legacy treatment advice. The server database schema is unchanged.

From the Flutter project directory, install the pinned API environment with `scripts/setup_api.ps1` if it does not already exist. Run the Flask API with the existing environment and start Flutter on the device using the local API address. The [API runtime guide](api-runtime.md) covers USB forwarding and LAN setup.

```powershell
.\.venv-api\Scripts\python.exe .\fabricare_api.py
flutter pub get
flutter run --dart-define=USE_LOCAL_API=true
```

Confirm database readiness at `GET /health/ready`. Sign in, select a fabric model and scan a clear front photo. Run the flow with an unknown label, then with **Dry clean only**. Check the absence of remaining months and numeric projections, the presence of source links and the dry-clean restriction. Review a marked finding and revisit the saved scan in History.

```powershell
.\.venv-api\Scripts\python.exe -m unittest discover -v
flutter analyze --no-pub
flutter test --no-pub
flutter test --no-pub test/scan_live_api_test.dart --dart-define=USE_LOCAL_API=true --dart-define=SCAN_TEST_PHOTO=model/2b2a6f82-68e1-4d0f-ad22-a011dd8df367.jpg --dart-define=SCAN_TEST_REVIEW_DAMAGE=true
```

### Version 1.1 verification

- Python unittest discovery passed 56 tests, including the four requested scenarios and partial review. Confirming an opening created the repair alert. Dismissing it cleared the alert while an unresolved stain remained a candidate. Dismissing the stain removed the stain precaution.
- Flutter passed 11 regular tests and a separate live scan against Flask and the trained artifacts. The checks covered the 62.5% score display, candidate location and status, conditional care, dry-clean-only instructions, reference-link dispatch, earlier-photo comparison, and saved damage review. Static analysis reported the existing informational lint in the profile screen.
- An updated USB Android APK was built and its signature verified. Android device interaction was outside this revision's verification.

### Version 1.0 verification

- Python unittest discovery passed 51 tests, including real inference through the three selectable fabric classifiers, request validation, reviewed and unsupported findings, label restrictions, low confidence and absent numeric outputs.
- The HTTP smoke script passed its upload, multiview, quality and compatibility checks and confirmed the model files were unchanged. It reported the configured database unavailable.
- A real garment upload returned HTTP 200 through the saved MobileNetV2 classifier and four-label YOLO detector. Model inference ran successfully on the supplied photo. This smoke check did not establish correctness against independent ground truth.
- The regular Flutter test suite passed 10 tests with the opt-in live test skipped. The live test passed separately against Flask and the saved models, including manual Dry clean only entry, results, reference-link dispatch, rendering both earlier and current photos, a damage review, and account-scoped Hive history.
- A disposable loopback MariaDB instance passed readiness, registration, login, authenticated profile, a scan with the actual SVM and YOLO artifacts, logout, and token rejection after logout. The configured development database returned HTTP 503 because its loopback MySQL connection was unavailable. No configured account data was used for the verification.
- Flutter static analysis reported one existing informational lint in the profile screen. No new issues were reported. Tests used the Windows Flutter SDK, and a physical Android or iOS run was outside this verification.

## Remaining research requirements

- Collect expert-reviewed phone photos with garment IDs, controlled references, repeat captures and explicit negatives such as prints, natural creases and textured fabric. Evaluate the deployed detector on a garment-level holdout using per-class precision, recall and false-positive analysis. Measure reviewer agreement and calibrate severity if a future grade is needed.
- Collect dated use, cleaning, repairs, repeated observations and disposal or end-of-use reasons. Define a garment-level endpoint and handle garments still in service at study end as censored outcomes. For a six-month prediction, evaluate against held-out future observations and report uncertainty for the intended user population.
- Independently label care recommendations against complete garment labels and expert decisions. Compare **Decision Tree**, **Random Forest** and **Support Vector Machine** on identical garment-grouped splits with comparable input variables and tuning effort. Report held-out accuracy, per-class precision and recall, macro F1 and confusion matrices. Measure violations of label restrictions separately. Select a care model after independent evaluation and always enforce label restrictions after inference. Existing fabric-classifier metrics describe a different task.

## Implementation files

- Backend rules and integration updated in `assessment.py`, `assessment_config.json`, `fabricare_api.py`, `image_processing.py` and `model_analysis.py`.
- Flutter scan and history updated in `lib/models/scan_result.dart`, `lib/screens/scan/scan_screen.dart`, `lib/screens/scan/ocr_label_screen.dart`, `lib/screens/scan/scan_result_screen.dart`, `lib/screens/history/garment_profile_screen.dart`, `lib/services/fabricare_api_service.dart` and `lib/services/health_score_calculator.dart`. The old OCR file now contains the manual care-label form.
- Reference-link support updated `pubspec.yaml`, `pubspec.lock`, `linux/flutter/generated_plugin_registrant.cc`, `linux/flutter/generated_plugins.cmake`, `macos/Flutter/GeneratedPluginRegistrant.swift`, `windows/flutter/generated_plugin_registrant.cc` and `windows/flutter/generated_plugins.cmake`.
- Tests and verification updated in `test_assessment.py`, `test_api_contract.py`, `scripts/check_api.py`, `test/history_store_test.dart`, `test/widget_test.dart`, `test/scan_live_api_test.dart` and `test_driver/android_model_smoke.dart`.
- Created `docs/FABRICARE_SCIENTIFIC_BASIS.md` and updated `docs/api-runtime.md` and `docs/scan-flow-changes.md`.
