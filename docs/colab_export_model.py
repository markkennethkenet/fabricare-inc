from google.colab import files
OUTPUT_DIR = Path('/content/drive/MyDrive/Fabricare_training') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
model_path = OUTPUT_DIR / 'fabric_glcm_decision_tree.pkl'
joblib.dump(best_model, model_path)
(OUTPUT_DIR / 'fabric_labels.json').write_text(json.dumps(CLASS_NAMES, indent=2), encoding='utf-8')
metadata = {
    'dataset': str(DATASET_DIR), 'seed': SEED, 'counts': counts,
    'source_counts': source_counts, 'skipped_identical_copies': len(duplicate_report),
    'duplicate_policy': 'keep first: train, valid, test; fail on conflicting labels',
    'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
    'numpy': np.__version__, 'opencv': cv2.__version__, 'scikit_image': skimage.__version__,
    'joblib': joblib.__version__, 'feature_count': 74,
    'selected_parameters': best_model.get_params(), 'validation_trials': trials,
    'evaluation': metrics,
    'preliminary_model': True,
    'missing_evaluation_classes': {split: [label for label in CLASS_NAMES if counts[split][label] == 0] for split in ['valid', 'test']},
    'limitations': ['Linen has no validation/test images in this run.', 'Exact duplicates filtered; garment/burst grouping not verified.', 'Low held-out accuracy; review before application deployment.'],
}
(OUTPUT_DIR / 'training_report.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
(OUTPUT_DIR / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(OUTPUT_DIR / 'duplicate_report.json').write_text(json.dumps(duplicate_report, indent=2), encoding='utf-8')
reloaded = joblib.load(model_path)
np.testing.assert_allclose(reloaded.predict_proba(X_test), best_model.predict_proba(X_test))
print('Export reload check passed:', OUTPUT_DIR)
archive = shutil.make_archive('/content/' + OUTPUT_DIR.name + '_fabricare_model', 'zip', OUTPUT_DIR)
files.download(archive)