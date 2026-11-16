# FabriCare

FabriCare is a Flutter app with a Python API for identifying fabrics from photos, checking visible damage, and suggesting garment care.

The scanner supports cotton, denim, linen, polyester, silk, and wool. It offers MobileNetV2, GLCM + SVM, and GLCM + Decision Tree for fabric identification.

- [Set up the API and test on Android](docs/api-runtime.md)
- [Understand the fabric assessment and care recommendations](docs/FABRICARE_SCIENTIFIC_BASIS.md)
- [Review the V3 dataset, training notebook, and polyester experiment](docs/SIX_CLASS_DATASET.md)

Copy `.env.example` to `.env` and enter your local database settings before running the API.
