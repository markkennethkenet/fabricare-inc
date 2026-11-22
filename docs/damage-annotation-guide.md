# Labelling guide: FabriCare damage detector (V7)

The app's damage detector draws a box around each **hole**, **needle mark**, **pilling** patch, and **stain**.
It was trained mostly on factory close-ups, so on phone photos of whole garments it boxes floor beside sleeves,
reads prints as stains, and misses small slits. Retraining on our own labelled phone photos fixes this.

## 1. Collect photos (Google Drive)

Put new phone photos in `My Drive/FabriCare_Damage_Dataset/Phone_Scans/<garment>/`, one folder per garment
(for example `Phone_Scans/yellow_dress/`). Photos already in `Own_Collected_Dataset/Tshirt_00N/` are used too.

| What | How many | Notes |
|---|---|---|
| Whole garment, front and back, on the floor or a table | 100+ photos, 30+ garments | Shoot like the app does: garment fills most of the frame |
| Close-ups of damage | 100+ | Holes, slits, needle marks, pilling, stains |
| Garments with prints, logos, embroidery and **no damage** | 30+ | Teaches that prints are not stains |
| Plain undamaged garments | 20+ | Teaches that clean fabric has no boxes |

Use different garments, colours, floors and lighting. Several photos of one garment are fine: the notebook keeps
each garment in a single split so test results are honest.

## 2. Pre-label (Colab, Part A)

Run Part A of `docs/fabricare_v7_damage_detector_colab.ipynb`. It runs the current detector on every photo, keeps
only boxes on the garment, and writes `FabriCare_Damage_Dataset/v7_prelabels.zip`. You correct these boxes
instead of drawing every box from scratch.

## 3. Correct the boxes (Roboflow)

**Send the photos without using your laptop's storage (recommended).** In Colab, open the key icon (Secrets),
add `ROBOFLOW_API_KEY` with your key from Roboflow → Settings → API Keys, and turn on notebook access. Then run
cell 3b: it uploads every pre-labelled photo from Drive into the Roboflow project `fabricare-damage-v7`
(created if missing, as an Object Detection project). Set `ROBOFLOW_WORKSPACE` if you use a team workspace.
Roboflow's free plan makes projects **public**; use a paid or education plan if the photos must stay private.

**Or upload by hand:** create an **Object Detection** project, download and unzip `v7_prelabels.zip`, and upload
the folder (images plus YOLO `.txt` labels and `data.yaml`).

1. Check the project has exactly these classes: `hole`, `needlemark`, `pilling`, `stain`. Keep the file names.
2. Open **Annotate** and fix every photo:

| Class | Box it | Do not box |
|---|---|---|
| **hole** | Every opening, tear or slit where fabric is missing or cut. **One box per hole**: a cluster of 8 small holes is 8 boxes. Box tightly around the opening. | Seams, folds, shadows, the gap between a sleeve and the floor, buttonholes |
| **needlemark** | Small rows of needle punctures or pulled threads | Fabric texture, weave pattern |
| **pilling** | Patches of fuzz balls; one box per patch | Normal fabric texture, lint you can brush off |
| **stain** | Discoloured marks: food, oil, rust, sweat yellowing, dye bleeding | **Prints, logos, embroidery, patterns, shadows, wet spots, reflections** |

3. Delete every wrong pre-label box (most often: floor, prints, folds). Leave a photo with **no boxes** if it has no damage: these background photos matter.
4. Every photo must be checked. An unchecked pre-label teaches the detector its own mistakes.

## 4. Export

Generate a dataset version with **no augmentation** and only *Auto-Orient* preprocessing (no resize; the notebook
resizes). Note its version number. Then either set `ROBOFLOW_VERSION` to that number in Colab cell 3c, which fetches
the **YOLOv8** export straight into `My Drive/FabriCare_Damage_Dataset/Annotated_Dataset/`, or download the YOLOv8
zip yourself and put it there. Roboflow's own train/valid/test split is ignored; the notebook re-splits by garment.

## 5. Train (Colab, Part B)

Run Part B. It fine-tunes the current detector on your labels (plus the public dataset if
`My Drive/FabriCare_Damage_YOLOv8_6Class.zip` exists), then tests the old and new detector on the **same held-out
garments** and reports precision, recall, mAP@0.5 and mAP@0.5:0.95 per class. Install the new detector only if it
beats the old one on your own photos.

## Quick checklist

- [ ] Classes named exactly `hole`, `needlemark`, `pilling`, `stain`
- [ ] Each hole in a cluster has its own box
- [ ] No boxes on prints, logos, floor, folds or shadows
- [ ] Every photo reviewed, including those left empty
- [ ] YOLOv8 export, no augmentation, file names unchanged
