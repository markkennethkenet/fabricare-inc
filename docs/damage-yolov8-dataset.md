# FabriCare YOLOv8 damage dataset

The combined dataset is in `dataset_sources/prepared_damage_yolov8/`. Its
`data.yaml` names the six detection classes and points to the train,
validation, and test image folders. Each image has a matching YOLO text file
in its split's `labels/` folder. Empty labels represent background images.
The class IDs are:

| ID | Class |
| ---: | --- |
| 0 | hole |
| 1 | needlemark |
| 2 | pilling |
| 3 | stain |
| 4 | missing_yarn |
| 5 | wrinkle |

## Sources and annotation changes

- [FabriD v2](https://data.mendeley.com/datasets/vncdwps24r/2), Sabbir Ahmed,
  Ismam Ahmed, Nishat Mahzabin, and Tahmina Islam, DOI
  `10.17632/vncdwps24r.2`. The public download is CC BY 4.0. The source
  archive was verified against Mendeley's SHA-256
  `4944cec7f54ce3f9120ed243ea8c33f3ff3a3d4c39556fd06e7cfe3a5b3a05c3`.
  Its segmentation polygons were converted to their enclosing detection
  boxes. The original empty annotations were preserved.
- [FabricSpotDefect](https://data.mendeley.com/datasets/6574nhzm8x/1), Sumaya,
  Farzana Islam, Md. Fahad Monir, and Ashraful Islam, DOI
  `10.17632/6574nhzm8x.1`. CC BY 4.0. The single `spot` class was mapped
  to `stain`. Original COCO boxes were used when their image dimensions
  matched the file. For the 228 dimension mismatches, one paired image and
  label from the authors' YOLO export was used in the same source split.
  Training fallback images may include the publisher's augmentations.
- [Fabric Wrinkles v4](https://universe.roboflow.com/fabric-wrinkles/fabric-wrinkles/dataset/4),
  CC BY 4.0. One version per Roboflow source filename was retained. Half of the
  unique source validation photos were assigned to test with seed 42. The
  source images are grayscale, and retained training versions may be augmented.
- [FabriCare damage archive](https://universe.roboflow.com/markkenneths-workspace/my-first-project-4dbrt-qbpgj/dataset/1),
  CC BY 4.0 per the supplied archive. All four classes were included so that
  the pilling images' other annotations are kept.

The build keeps source splits where possible. If an original filename within a
source or an exact decoded image appears more than once, test takes precedence
over validation, then training. Original garment IDs are unavailable, so this
does not establish a garment-level split. A held-out set of new phone photos
is still needed to measure performance on FabriCare's camera inputs.

`manifest.csv` identifies each image's archive, member, source split,
destination split, source photo group, pixel hash, annotation conversion, and
box count. `summary.json` reports output counts and duplicates removed. The
archives and existing trained model remain in place. This dataset includes
general stain labels and visible missing yarn; it has no separate classes for
stain causes, wear-related fraying, discoloration, or fading.

## Rebuild

From the FabriCare repository, use the Python environment with Pillow and
PyYAML. The output argument must name a directory that does not exist.

```powershell
$downloads = Join-Path $env:USERPROFILE 'Downloads'
$spot = Join-Path $downloads 'FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types.zip'
.\.venv-api\Scripts\python.exe scripts\assemble_damage_dataset.py `
  --fabrid dataset_sources\raw\FabriD-v2.zip `
  --spot "$spot" `
  --wrinkles "$downloads\Fabric Wrinkles.v4i.yolov8.zip" `
  --fabricare "$downloads\fabric_damage_dataset.zip" `
  --output dataset_sources\prepared_damage_yolov8
```

For Ultralytics, pass the absolute path to the generated `data.yaml` as the
`data` argument. Treat this as a new six-class detector experiment. The
current FabriCare damage model and its app integration use the original four
classes.
