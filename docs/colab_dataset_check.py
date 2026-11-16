# Reuse the feature snapshot from this session when available.
# Restart the session if source images have changed since extraction.
_feature_cache = {}
if 'data' in globals() and 'manifest' in globals():
    for _split, (_x, _y) in data.items():
        _records = [r for r in manifest if r['split'] == _split]
        if len(_records) == len(_x) == len(_y):
            for _r, _v, _label in zip(_records, _x, _y):
                if _r['label'] == str(_label) and np.asarray(_v).shape == (74,):
                    _feature_cache[_r['path']] = (_r['decoded_sha256'], np.asarray(_v).copy())
print('Reusable feature rows:', len(_feature_cache))

def extract_glcm_features(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot read image: {image_path}')
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray / 32).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.extend(graycoprops(glcm, prop).flatten())
    features.extend([np.mean(gray), np.std(gray)])
    return np.array(features).reshape(1, -1)

# Keep one copy per decoded image, preferring train, then valid, then test.
# Source images are never edited, moved, or deleted.
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
data, counts, source_counts, manifest, seen = {}, {}, {}, [], {}
duplicate_report, errors = [], []
for split in ['train', 'valid', 'test']:
    split_dir = DATASET_DIR / split
    if not split_dir.is_dir():
        raise ValueError(f'Missing split: {split_dir}')
    actual_classes = {p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith('.')}
    if actual_classes != set(CLASS_NAMES):
        raise ValueError(f'{split}: expected {CLASS_NAMES}, found {sorted(actual_classes)}')
    features, labels = [], []
    counts[split], source_counts[split] = {}, {}
    for label in CLASS_NAMES:
        paths = sorted(p for p in (split_dir / label).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        source_counts[split][label] = len(paths)
        counts[split][label] = 0
        for path in paths:
            relative_path = str(path.relative_to(DATASET_DIR))
            cached = _feature_cache.get(relative_path)
            if cached is not None:
                digest, vector = cached
            else:
                image = cv2.imread(str(path))
                if image is None:
                    errors.append(f'Unreadable: {path}')
                    continue
                digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
                vector = extract_glcm_features(path)[0]
            previous = seen.get(digest)
            if previous:
                if previous['label'] != label:
                    errors.append(f'Conflicting labels for identical image: {previous["path"]} and {relative_path}')
                else:
                    duplicate_report.append({'skipped_path': relative_path, 'skipped_split': split,
                        'kept_path': previous['path'], 'kept_split': previous['split'],
                        'label': label, 'decoded_sha256': digest})
                continue
            record = {'path': relative_path, 'split': split, 'label': label, 'decoded_sha256': digest}
            seen[digest] = record
            features.append(vector)
            labels.append(label)
            manifest.append(record)
            counts[split][label] += 1
        if counts[split][label] == 0:
            if split == 'train' or source_counts[split][label] > 0:
                errors.append(f'No unique readable images remain: {split}/{label}. Add independent photos to this split.')
            else:
                print(f'WARNING: {split}/{label} has no images. This class cannot be evaluated in this split.')
    if not features:
        errors.append(f'No images retained in {split}')
    data[split] = (np.asarray(features), np.asarray(labels))
    print(split, 'source:', source_counts[split], 'retained:', counts[split])
print(f'Skipped {len(duplicate_report)} identical same-label copies. No Drive images were modified.')
for item in duplicate_report[:10]:
    print('Skipped:', item['skipped_path'], '| Kept:', item['kept_path'])
if errors:
    print('\n'.join(errors[:30]))
    raise ValueError(f'{len(errors)} unresolved dataset issue(s). Review conflicting labels, unreadable files, or empty classes before training.')
assert all(x.shape[1] == 74 for x, y in data.values())
print('Exact-image duplicate checks passed. Remaining burst shots/crops of the same garment must still stay in one split.')
print('Validation and test results apply to the retained unique images, not the original unfiltered splits.')
