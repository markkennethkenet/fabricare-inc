# FabriCare Models, Formulas and Objectives

17 November 2026 · Copy of the Claude Doc report.

## Summary

All five study objectives are implemented in the current system on `main`. Each one has trained models or formulas behind it, measured results, and a screen in the app that shows it.

- **Fabric identification (objective 1):** ConvNeXt-Tiny, a CNN, scored 86.69% on 353 held-out test photos. GLCM + SVM scored 59.49% and GLCM + Decision Tree 55.52%.
- **Care recommendation (objective 2):** the Decision Tree reproduced the referenced care rules on 99.15% of test photos, and 90.65% end to end, where the CNN's fabric mistakes are included.
- **Health and lifespan (objective 4):** every scan gets a 0–100 fabric health score and a remaining-lifespan range in months.
- **App and AI care (objectives 3 and 5):** the Flutter app shows the fabric with its confidence score, the condition, the laundry mode the model predicts, and automated care-risk alerts.

The laundry suggestion is machine-learning AI, not generative AI. The trained Decision Tree predicts the wash mode; the three tips under it are chosen from referenced guidance. Gemini can write those tips freely at no cost (section 7).

One thing outside the code needs fixing before the defense. The online database (Aiven MySQL) is not answering, so Google sign-in fails online. Scanning works.

## Why the Decision Tree

The Decision Tree was chosen because it had the best validation score, the rule chosen before the test photos were opened. The Random Forest scored 0.28 points higher on test, which is 1 photo out of 353. That gain costs 300 times the model size and over 300 times the time per scan.

| Care model | Validation macro-F1 | Test accuracy, true fabric | Test accuracy, CNN fabric (end to end) | Model size | Time per scan |
| --- | --- | --- | --- | --- | --- |
| **Decision Tree (deployed)** | **0.999** | 99.15% | 90.65% | 7.3 KB | 0.16 ms |
| Random Forest | 0.997 | 99.43% | 90.93% | 2,183 KB (300 trees) | 53.6 ms |
| SVM (RBF) | 0.918 | 95.75% | 87.82% | 117 KB (443 support vectors) | 0.55 ms |

Times were measured on this laptop's CPU. All three models were trained on the same garment-grouped split and the same 24 features.

- **Selected by a fixed rule.** The model with the highest validation macro-F1 was picked. Choosing by test score would make the test result optimistic.
- **Explainable.** The tree has 23 leaves and depth 11. Each laundry mode it gives traces to one readable if-then path, such as "wool and no hole → hand wash cold". A Random Forest averages 300 trees and an SVM uses a curved boundary, so neither gives one reason a panel can follow.
- **Matches the labels.** The care labels come from referenced if-then rules (Woolmark, Levi's, UGA Extension, ISO 3758). A tree has the same form, so it learns them almost exactly: 99.15% on test.
- **Cheap to serve.** The Vercel container loads 7.3 KB instead of 2.1 MB, and each scan adds 0.16 ms.

The 9-point drop from 99.15% to 90.65% end to end comes from the fabric step, not the care step. When the CNN names the wrong fabric, every care model inherits the error. The same drop appears for all three models.

## How a care recommendation is made

The trained Decision Tree picks the laundry mode from 24 scan features. The app then keeps the 3 most important care tips by fixed priority, and writes 3 laundry steps within any care label the user entered.

**1. Features (24 per scan)**

- Fabric from the CNN, one-hot (6): cotton, denim, linen, polyester, silk, wool.
- Damage (8): for hole, needle mark, pilling and stain, the count and the largest box share of the photo. A finding counts only if its detector score reaches the threshold (hole 90%, stain 70%, pilling 70%, needle mark 60%) or the user confirmed it.
- Colour of the central 80% of the front photo (4): mean luminance (0–255), luminance spread, mean saturation (0–1), share of dark pixels (luminance under 70).
- GLCM texture (6): contrast, dissimilarity, homogeneity, energy, correlation and ASM, each averaged over 3 distances and 4 angles.

**2. Laundry mode.** The tree returns one of 5 modes: machine normal warm, machine gentle cold, hand wash cold, professional cleaning, repair first. Its confidence is the share of training photos of that mode in the leaf the scan lands in.

```latex
\text{mode} = \arg\max_{k} \; P_{\text{tree}}(k \mid x), \qquad \text{confidence} = \max_{k} P_{\text{tree}}(k \mid x)
```

**3. Training labels (the rules the tree learned).** Each training photo was labelled by the first rule that applies, from the sources the app cites:

1. A hole → repair first (ISO 13934-1, ISO 13937-1).
2. Silk with a stain → professional cleaning; other silk → hand wash cold (UGA Extension, Canadian Conservation Institute).
3. Wool → hand wash cold (Woolmark).
4. Denim or polyester → machine gentle cold (Levi's, UGA Extension).
5. Cotton or linen that is dark or strongly coloured, or has pilling → machine gentle cold (Cotton et al. 2020, ISO 12945-4).
6. Other cotton or linen → machine normal warm; unknown fabric → hand wash cold.

```latex
\text{dark or strong} \iff \bar{L} < 90 \;\lor\; \text{dark share} > 0.5 \;\lor\; \bar{S} > 0.45
```

**4. Care label limits.** A wash symbol the user entered replaces the mode's method. A cold suggestion stays at min(label maximum, 30 °C), and drying and ironing follow the label's symbols.

**5. Care tips: the top 3 by priority.** Every rule whose condition holds adds one sentence; the 3 highest priorities are shown.

| Condition on the scan | Priority | Tip (shortened) |
| --- | --- | --- |
| Hole found | 100 | Mend the opening before washing or wearing |
| Stain found | 95 | Treat it first; silk goes to a cleaner |
| Fabric below the 60% threshold | 90 | Check the fibre label first |
| Pilling found | 85 | Shave pills when dry, wash inside out |
| Needle mark found | 80 | Secure loose threads |
| User wrings or rubs | 78 | Press water out gently |
| Dark or strong colour (not denim) | 75 | Wash with dark colours in cool water |
| Fabric-specific (wool, silk, denim, polyester, cotton, linen) | 72 to 56 | E.g. wool: air it between wears |
| Washed several times a week | 62 | Wash only when dirty |
| Worn daily or several times a week | 57 | Rotate with similar pieces |
| Light colour | 50 | Keep away from dark or new items |
| Always | 10 | Follow the care label |

**6. Laundry steps (3 sentences):** how to wash (mode or label), what to wash with (fabric, colour, the label's bleach symbol), and how to dry and iron (label symbols, otherwise the fabric's default).

**7. Automated care-risk alerts** are added for an opening, stain, pilling, needle mark, colour bleeding, felting (wool), water marks (silk), heat (polyester) and dye transfer (denim).

## Fabric health score

The health score starts at 100 and loses points once for each kind of damage found, more for damage the user confirmed. Small marks under 2% of the photo cost 30% less.

```latex
\text{health} = \max\Big(0,\; \operatorname{round}\big(100 - \sum_{d \in \text{found}} p_d \times a_d\big)\Big), \qquad a_d = \begin{cases} 1.0 & \text{largest box} \ge 2\% \text{ of the photo} \\ 0.7 & \text{otherwise} \end{cases}
```

A damage type is found when the user confirmed it, or the detector's score reaches its threshold (hole 90%, stain 70%, pilling 70%, needle mark 60%). Weaker findings stay on screen for review and remove nothing.

| Damage type | Points if detected (p) | Points if confirmed (p) | Detector threshold |
| --- | --- | --- | --- |
| Hole | 20 | 40 | 90% |
| Pilling | 10 | 20 | 70% |
| Stain | 8 | 15 | 70% |
| Needle mark | 5 | 10 | 60% |

Grades: **Excellent** 85–100 · **Good** 70–84 · **Fair** 50–69 · **Poor** 0–49.

**Worked example.** Denim with possible pilling at an 80% detector score, covering 3% of the photo: 100 − 10 × 1.0 = **90, Excellent**. If the user confirms the pilling: 100 − 20 = 80, Good. A confirmed hole as well: 80 − 40 = 40, Poor.

The point weights are FabriCare's own design, not taken from a study. The score describes only what the photos show, not fibre strength. The detector thresholds come from the V4 test run: at those levels the detector flagged holes on 3.4% of mostly undamaged photos, against 35.1% at a 40% score.

## Remaining lifespan estimate

The lifespan estimate starts from how long garments last in a survey of 46,857 garments in five countries ([Laitala and Klepp 2020](https://doi.org/10.3390/su12219151), section 4.1). It adjusts for fibre, wear and washing, subtracts the garment's age, shortens the result for visible damage, and shows it as a range.

```latex
R = \max\big(0,\; T_{\text{fibre}} + \tfrac{\Delta_{\text{wear}}}{12} + \Delta_{\text{wash}} - \text{age}\big) \times 12 \times \prod_{d \in \text{found}} f_d \quad \text{(months)}
```

```latex
\text{shown range} = [\,0.6R,\; 1.4R\,], \qquad \text{confirmed hole} \Rightarrow R = \min(R,\, 6)
```

| Term | How it is set | Source |
| --- | --- | --- |
| T fibre (years) | Cotton, denim, linen 4.91 · polyester 5.37 (+5.5 months) · wool 6.11 (+1.2 years) · silk 7.11 (+2.2 years) · unconfirmed fabric 5.2 (survey mean) | Survey |
| Δ wear (months) | −2 × (wears per year − 52) / 10, kept between −24 and +12. Wears per year: rarely 12, weekly 52, several a week 156, daily 312 | Survey: 10 more wears a year ≈ 2 months shorter |
| Δ wash (years) | B(n) − B(3), kept between −2 and +2, where n = wears between washes and B(n) = 4.8 × ln(min(n, 30)) / ln 30 | Survey: washing less often adds up to 4.8 years |
| Age (years) | From the purchase date; 0 if new; 2.6 (survey average) if unknown | User / survey |
| f damage | Hole 0.75 · pilling 0.9 (confirmed 0.8) · stain 0.95 (0.9) · needle mark 0.98 (0.95) | FabriCare assumption |
| Range 0.6R–1.4R | The survey's model explains about 22% of the variation between garments | Survey |

**Worked example.** Denim worn weekly, washed after every wear, bought a year ago, possible pilling:

1. Fibre baseline: 4.91 years = **59 months**.
2. Wear weekly (52 a year): **0 months**.
3. Washing after every wear (n = 1) instead of every 3 wears: 0 − 1.55 years = **−19 months** (40 months left).
4. Age 1 year: **−12 months** (28.3 months left).
5. Possible pilling × 0.9: **−3 months**.

Result: about **25 months**, shown as **15 to 36 months**.

The survey figures describe garments in general, so this is an estimate for planning, not a prediction for one garment. The damage factors and the 6-month cap for a confirmed hole are FabriCare's assumptions.

## Objectives checklist

All five objectives are met in the code on `main`. The one open item is operational, not missing work: the online database must be powered on before Google sign-in works on the hosted app.

| Objective | Met? | Where it lives | Evidence |
| --- | --- | --- | --- |
| 1. Train, test and compare CNN, GLCM + Decision Tree and SVM for fabric identification | Yes | Training notebook `docs/fabricare_v4_training_colab.ipynb`; app model picker | Test accuracy on 353 photos: ConvNeXt-Tiny 86.69%, GLCM + SVM 59.49%, GLCM + Decision Tree 55.52%. Also macro-F1, per-class scores, simulated retakes and 11 phone photos. |
| 2. Train, test and compare Decision Tree, Random Forest and SVM for care recommendations from fabric, condition and image features | Yes | `care_model.py`; `model/care/` | Validation macro-F1 0.999 / 0.997 / 0.918; end-to-end test 90.65% / 90.93% / 87.82%. Decision Tree deployed (section 2). |
| 3. Mobile-responsive app: upload images, get the fabric with a confidence score and a condition assessment | Yes | Flutter app (Android APK and web on Vercel) | Smart camera or gallery upload, the fabric in large type with its model score, other likely fabrics, damage review. All three fabric models report a score. |
| 4. Fabric health assessment and lifespan estimation | Yes | `assessment.py`: `health_assessment`, `lifespan_estimate` | 0–100 health score with grade (section 4); remaining months with a range (section 5). |
| 5. AI laundry-mode suggestions and automated care-risk alerts | Yes | `fabricare_api.ai_care_for`; result screen | The Decision Tree predicts the laundry mode; 9 alert types fire automatically. Optional Gemini wording (section 7). |

What each objective looks like in the app, on one scan: the fabric card (objectives 1 and 3), Care recommendations and Laundry suggestion (2 and 5), Care alerts (5), Fabric condition and Remaining lifespan (4).

## Gemini wording (generative AI)

Gemini can reword the care and laundry tips for each garment at no cost, using Google's free tier for `gemini-3.5-flash-lite`. It is built and tested, and turns on once a `GEMINI_API_KEY` is set on the server.

- **Cost.** Google lists input and output for this model as "Free of charge" on the free tier ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). Without a billing account linked to the project, nothing can be charged. Paid use would be $0.30 per million input tokens and $2.50 per million output tokens.
- **Daily limit.** Google shows each project's limits in AI Studio rather than in its docs ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)). Limits apply per project and reset at midnight Pacific time.
- **Privacy.** On the free tier Google may use requests to improve its products. Each request carries the scan results and a 768-pixel copy of the front photo. Setting `GEMINI_SEND_PHOTO=false` sends the results only.
- **What stays in charge.** The Decision Tree's laundry mode, the care label's limits and the referenced tips are all sent as facts. Gemini is told to follow them, add nothing that wasn't given, and write at most 3 sentences per list.
- **Fallback.** No key, a timeout over 8 seconds, a used-up quota or a malformed answer → the referenced tips are shown, as now. A scan never waits on or fails because of Gemini.
- **Labelling.** Gemini's wording is marked "… · worded by Gemini" under the card. The referenced tips stay saved with the scan, and their sources still appear under Sources and method.

To turn it on:

1. In [Google AI Studio](https://aistudio.google.com/apikey), create an API key with the FabriCare Google account.
2. In Vercel → fabricare1 → Settings → Environment Variables, add `GEMINI_API_KEY` for Production.
3. Redeploy.

Sources: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), opened 17 November 2026.

## Limitations and how to say it in the defense

State these limits before the panel finds them. Each has a short, honest answer.

| Limitation | What to say |
| --- | --- |
| Care accuracy measures agreement with referenced rules, not laundry lab results. | "The care model learned published care guidance; 99% means it reproduces that guidance, and the care label always overrides it." |
| End-to-end care accuracy (90.65%) inherits the CNN's fabric errors. | "Most care mistakes start as fabric mistakes, so better fabric photos improve both." |
| The 60% fabric review threshold follows from training with label smoothing 0.1, which caps scores near 92%. It is not yet calibrated. | "The next step is to choose it on validation photos by the accuracy of scans above it." |
| Health point weights and lifespan damage factors are FabriCare's assumptions. | "The structure is referenced; the weights are ours and are labelled as assumptions in the app." |
| The lifespan survey model explains about 22% of the variation between garments. | "That is why the app shows a wide range, not one number." |
| The YOLOv8 damage detector is a short baseline. | "Thresholds keep false alarms low, and every finding can be confirmed or dismissed by the user." |
| A photo cannot prove fibre content; denim is a weave, not a fibre; blends are not classes. | "FabriCare suggests a fabric and asks for the fibre label when unsure." |
| Gemini's free tier may use requests to improve Google's products, and its sentences carry no reference of their own. | "Gemini only rewords referenced tips; the decision comes from our trained model." |

Calling it AI: say the laundry mode is **predicted by a trained machine-learning model**. With Gemini on, add that **a generative model words the advice**. Without Gemini, avoid the term "generative AI".

Before the defense: power on the Aiven MySQL service, so Google sign-in works on the hosted app, and run one full scan on the phone without the laptop.
