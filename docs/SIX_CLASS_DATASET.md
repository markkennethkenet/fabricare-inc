# FabricCare V3 dataset and training report

The main Colab notebook uses [FabriCare_Dataset_V3](https://drive.google.com/drive/folders/1cHixSkWAvL3gK1Mhuw8s6G0E7RMYx0-8). The six supported classes are **cotton, denim, linen, polyester, silk, and wool**. Adding another class requires verified labeled examples, a new evaluation split, and an updated model and scanner label configuration.

## Run the training in Colab

1. Open the shared V3 folder. Add a shortcut directly under **My Drive** named `FabriCare_Dataset_V3`, matching the Drive folder's capitalization.
2. Upload [the main training notebook](fabric_identification_mobilenet_colab.ipynb) to [Google Colab](https://colab.research.google.com/).
3. Select **Runtime → Change runtime type → GPU**, then choose **Run all** and authorize the Drive mount when prompted.
4. Open **My Drive → Fabricare_training_v3 → the new timestamp**. Use `training_report.json` to compare all three models on the same test images. The folder also contains the model exports, ordered class labels, and the exact split manifest used in that run.

The notebook trains a MobileNetV2 CNN and two classifiers that use the scanner's 72 GLCM texture features, a Decision Tree and an SVM. It selects settings with validation data and opens the test images in the final evaluation cell. The three models perform fabric identification. Their metrics do not measure damage detection, care recommendations, or lifespan estimates.

## Dataset checks and fixes

- **Stale training inputs.** The older notebooks and a local comparison helper referenced the earlier dataset. The training notebooks now use the V3 split manifest and load the corresponding files from `raw/<fabric>/`. The validation experiment uses the published V3 model for its reference scores.
- **Comparable splits.** I matched all 3,462 local image files and 2,538 recorded groups against the V3 manifest. There are 2,398 training, 711 validation, and 353 test images, with all six classes in every split. No duplicate image hashes or recorded garment groups cross splits. The notebook enforces these checks before training. Group checks depend on the accuracy of the original group labels. [V3 split manifest](https://drive.google.com/file/d/1Hb1q5qwEFzE2CKHiP__bAd1JTZKeh8P7/view)
- **Local filenames and large originals.** The local copy uses slightly different names for 26 images, all verified by SHA-256. Of the 3,462 archived images, 972 exceed the API's direct upload pixel limit. Offline staging verifies the original file hash and applies the scanner's EXIF orientation, bilinear resize, and gray padding to produce a 224 × 224 input.
- **Scanner provenance.** The MobileNetV2, GLCM Decision Tree, and GLCM SVM files in the project matched the corresponding V3 Drive exports byte for byte. A saved scanner model can classify photos without a live Drive connection. The notebook mounts Drive to read training images and save new artifacts. [V3 model exports](https://drive.google.com/drive/folders/1w9sYuv_XmUwJJxkiRGdWXWhcWxhmq0b7)
- **Colab access.** I could read the shared Drive dataset and compare the published exports. The desktop browser URL check stopped before I could start the signed-in Colab session. The GPU training run still needs to be launched from the notebook with Drive access.

## Reproduced test results

I reran the three existing V3 model exports on all 353 held-out photos. The accuracy and macro F1 matched the published results.

- **MobileNetV2** had 78.75% accuracy and 0.7821 macro F1.
- **GLCM + SVM** had 58.92% accuracy and 0.5781 macro F1.
- **GLCM + Decision Tree** had 50.42% accuracy and 0.5109 macro F1.

I also ran the updated notebook's training, export, and reporting cells locally on 36 V3 photos, two per class per split, with one epoch per CNN stage and no ImageNet download. That run verified that all three training and export paths work on a sample. The scores above are from the full held-out evaluation of the existing V3 exports.

## Confidence and accuracy improvements

- Collect more **independent, verified polyester and silk garments**. These had the weakest MobileNetV2 recall in the published V3 test set, at 51.1% and 67.3%. Record material from a reliable source during collection, including verified composition for blends, and capture each garment with the range of phones, angles, distances, and lighting expected in use. [Published MobileNetV2 class report](https://drive.google.com/file/d/157RvoF_MgYHURnC6ihjoPWc24wUT-CpP/view)
- Keep every photo of a garment in one split and review near-duplicate crops and label conflicts. Reserve a separate batch of real phone scans for the final deployment check and report per-class recall and macro F1 alongside overall accuracy.
- Calibrate the CNN scores on a separate garment-grouped validation set. On the final test set, report how often high scores are correct and what fraction of photos are above the chosen threshold. Use those results to choose when the app should request another photo or show an uncertain result. Include blends and unsupported fabrics when evaluating that behavior.
- As an example, at an uncalibrated score cutoff of 0.85, the current CNN covered 172 of 353 test photos (48.7%). It classified 160 of those 172 correctly (93.0%). Choose the application threshold on validation images and verify it on a new phone test set.

## Polyester retraining experiment

I found 600 additional filenames in the V3 polyester folder. I used 292 whose titles explicitly named polyester and did not name another fiber or indicate a blend or backing. All 292 passed decoding, SHA-256, and perceptual duplicate checks. Titles are provisional fiber labels. The remaining 308 need composition verification before inclusion. [V3 polyester folder](https://drive.google.com/drive/folders/1UNSiVspMziCEbovM8CcVdKzDlPF9jIJg)

I trained the final six-class layer locally, starting from the scanner's existing MobileNetV2 weights and freezing its image feature extractor. One run used the original 2,398 training photos, and the other added the 292 new photos. Both used the original 711 validation photos to select epochs. All three results below use the same 353 held-out images, including 45 labeled polyester. No new silk images were required.

- **Current scanner.** Correctly found 23 of 45 polyester photos (51.1% recall). Polyester precision was 88.5% (23 of 26 calls), and overall accuracy was 78.75%. The mean polyester score on labeled polyester was 44.1%.
- **Refit on original photos.** Correctly found 33 of 45 polyester photos (73.3% recall). Polyester precision was 86.8% (33 of 38 calls), and overall accuracy was 81.87%. The mean polyester score on labeled polyester was 64.0%.
- **Refit with the 292 additions.** Correctly found 35 of 45 polyester photos (77.8% recall). Polyester precision was 76.1% (35 of 46 calls), and overall accuracy was 79.89%. The mean polyester score on labeled polyester was 69.1%.

The additions corresponded to two more correct polyester predictions and six more false polyester calls than the original-only refit. At a score cutoff of 0.85, the candidate made 22 polyester predictions and got 21 right, compared with 14 predictions and 13 correct for the original-only refit. These are test-set observations of raw model scores. A separate batch of phone photos would help verify whether either retrain should be used by the app.

The [experiment directory](../research/polyester_v3) contains the candidate Keras model, the 292-image selection with SHA-256 hashes, and the full results. The [retraining script](../scripts/retrain_polyester.py) reads the original V3 split manifest and published MobileNetV2 export. Use the project Python environment with TensorFlow, Pillow, NumPy, SciPy, and scikit-learn installed. With a local copy of the V3 folder, run from the repository root in PowerShell.

```powershell
$dataset = 'C:\path\to\FabriCare_Dataset_V3'
python scripts/retrain_polyester.py `
  --raw "$dataset/raw" `
  --manifest "$dataset/reports/split_manifest_70_20_10.csv" `
  --baseline "$dataset/models_v3/mobilenetv2_stage2_best.keras" `
  --output research/output/polyester_run
```
