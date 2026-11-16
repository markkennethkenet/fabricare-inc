"""Run after the corrected dataset-check cell; preserves the active API model."""
import re
import time
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

assert 'data' in globals() and 'manifest' in globals(), 'Run the corrected dataset-check cell first.'
assert sklearn.__version__ == '1.6.1', 'Use the backend-compatible scikit-learn version.'

# Verify each feature row has its own matching provenance record.
improvement_records = {}
for split in ['train', 'valid', 'test']:
    records = [r for r in manifest if r['split'] == split]
    x, y = data[split]
    assert len(records) == len(x) == len(y), f'{split}: provenance length mismatch'
    assert [r['label'] for r in records] == list(y), f'{split}: provenance order mismatch'
    improvement_records[split] = records

def burst_group(record):
    # A timestamped burst is a detectable photo session, not proof of garment identity.
    stem = Path(record['path']).stem
    stem = re.sub(r'\s*\(\d+\)$', '', stem)
    match = re.match(r'(?i)^(.*?)[_-]burst\d+', stem)
    return (record['label'], match.group(1).lower()) if match else None

test_groups = {g for r in improvement_records['test'] if (g := burst_group(r)) is not None}
valid_groups = {g for r in improvement_records['valid'] if (g := burst_group(r)) is not None}
improved_data, improved_manifest, burst_exclusions = {}, [], []
for split in ['train', 'valid', 'test']:
    blocked = (test_groups | valid_groups) if split == 'train' else test_groups if split == 'valid' else set()
    keep = []
    for record in improvement_records[split]:
        group = burst_group(record)
        retain = group is None or group not in blocked
        keep.append(retain)
        if retain:
            improved_manifest.append(record)
        else:
            burst_exclusions.append({**record, 'reason': 'burst group also present in a held-out split'})
    mask = np.asarray(keep, dtype=bool)
    improved_data[split] = (data[split][0][mask], data[split][1][mask])
    assert mask.any(), f'{split}: burst filtering emptied the split; collect independent photos.'
    print(split, 'retained:', int(mask.sum()), 'excluded burst overlap:', int((~mask).sum()), flush=True)

train_x, train_y = improved_data['train']
valid_x, valid_y = improved_data['valid']
assert set(train_y) == set(CLASS_NAMES), 'Burst filtering removed a training class; review the dataset.'
supported_valid_labels = sorted(set(valid_y))
print('Untested validation classes:', sorted(set(CLASS_NAMES) - set(valid_y)), flush=True)
print('Selection uses validation macro F1 over labels actually represented in validation.', flush=True)

candidate_specs = [('decision_tree_baseline', DecisionTreeClassifier(
    class_weight='balanced', max_depth=12, min_samples_leaf=10, random_state=SEED))]
for leaf in [1, 3]:
    for features in ['sqrt', 0.5]:
        candidate_specs.append((f'extra_trees_leaf{leaf}_features{features}', ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features=features,
            class_weight='balanced', n_jobs=2, random_state=SEED)))
    candidate_specs.append((f'random_forest_leaf{leaf}', RandomForestClassifier(
        n_estimators=400, min_samples_leaf=leaf, max_features='sqrt',
        class_weight='balanced_subsample', n_jobs=2, random_state=SEED)))
for c in [1, 10, 100]:
    candidate_specs.append((f'scaled_rbf_svm_C{c}', make_pipeline(StandardScaler(), SVC(
        C=c, gamma='scale', class_weight='balanced', probability=True,
        random_state=SEED, cache_size=512))))

improvement_trials = []
selected_model, selected_name, selected_score = None, None, -1
baseline_model = None
for name, spec in candidate_specs:
    started = time.monotonic()
    model = clone(spec).fit(train_x, train_y)
    prediction = model.predict(valid_x)
    score = float(f1_score(valid_y, prediction, labels=supported_valid_labels, average='macro', zero_division=0))
    result = {'name': name, 'validation_macro_f1_supported_classes': score,
        'validation_accuracy': float(accuracy_score(valid_y, prediction)),
        'training_seconds': round(time.monotonic() - started, 2)}
    improvement_trials.append(result)
    print(result, flush=True)
    if name == 'decision_tree_baseline':
        baseline_model = model
    if score > selected_score:
        selected_model, selected_name, selected_score = model, name, score

print('SELECTED USING VALIDATION ONLY:', selected_name, 'macro F1:', selected_score, flush=True)

# Evaluate the chosen model and the retrained baseline once on the same test set.
# The test labels below do not participate in model selection.
test_x, test_y = improved_data['test']
supported_test_labels = sorted(set(test_y))
def evaluation(model):
    prediction = model.predict(test_x)
    return {
        'accuracy': float(accuracy_score(test_y, prediction)),
        'macro_f1_supported_classes': float(f1_score(test_y, prediction,
            labels=supported_test_labels, average='macro', zero_division=0)),
        'supported_classes': supported_test_labels,
        'missing_classes': sorted(set(CLASS_NAMES) - set(test_y)),
        'classification_report': classification_report(test_y, prediction,
            labels=CLASS_NAMES, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(test_y, prediction, labels=CLASS_NAMES).tolist(),
        'confusion_matrix_labels': CLASS_NAMES,
    }

selected_evaluation = evaluation(selected_model)
baseline_evaluation = evaluation(baseline_model)
print('TEST selected:', selected_evaluation['accuracy'], 'baseline:', baseline_evaluation['accuracy'], flush=True)
print('TEST macro F1 (supported classes):', selected_evaluation['macro_f1_supported_classes'], flush=True)
print('UNTESTED classes:', selected_evaluation['missing_classes'], flush=True)
print(classification_report(test_y, selected_model.predict(test_x), labels=CLASS_NAMES, zero_division=0), flush=True)

improvement_dir = Path('/content/drive/MyDrive/Fabricare_training') / (
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_classifier_comparison')
improvement_dir.mkdir(parents=True, exist_ok=False)
joblib.dump(selected_model, improvement_dir / 'fabric_glcm_decision_tree.pkl', compress=3)
(improvement_dir / 'fabric_labels.json').write_text(json.dumps(CLASS_NAMES, indent=2))
improvement_report = {
    'selected_model': selected_name, 'seed': SEED, 'feature_count': 74,
    'selection_metric': 'validation macro F1 on supported classes',
    'validation_supported_classes': supported_valid_labels,
    'candidates': improvement_trials,
    'selected_test': selected_evaluation, 'baseline_test_same_split': baseline_evaluation,
    'counts': {split: {label: int(np.sum(y == label)) for label in CLASS_NAMES}
        for split, (x, y) in improved_data.items()},
    'excluded_burst_images': len(burst_exclusions),
    'preliminary_model': True,
    'limitations': ['Garment identity and non-burst related photos have not been verified.',
        'Linen has no validation or test images.',
        'Test split was evaluated in a previous run; collect a new independent phone/garment test set before claiming final accuracy.'],
    'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'opencv': cv2.__version__, 'scikit_image': skimage.__version__, 'joblib': joblib.__version__},
}
for name, payload in [('training_report.json', improvement_report),
        ('dataset_manifest.json', improved_manifest), ('burst_exclusions.json', burst_exclusions),
        ('duplicate_report.json', duplicate_report)]:
    (improvement_dir / name).write_text(json.dumps(payload, indent=2))
# Preserve computed features for future experiments without rereading all Drive photos.
np.savez_compressed(improvement_dir / 'glcm_features.npz',
    **{f'{split}_{kind}': value for split, pair in improved_data.items()
       for kind, value in zip(['x', 'y'], pair)})
loaded = joblib.load(improvement_dir / 'fabric_glcm_decision_tree.pkl')
np.testing.assert_array_equal(loaded.predict(test_x), selected_model.predict(test_x))
np.testing.assert_allclose(loaded.predict_proba(test_x), selected_model.predict_proba(test_x))
improvement_archive = shutil.make_archive('/content/' + improvement_dir.name, 'zip', improvement_dir)
print('EXPORT VERIFIED:', improvement_dir, flush=True)
print('Archive:', improvement_archive, flush=True)
from google.colab import files
files.download(improvement_archive)
