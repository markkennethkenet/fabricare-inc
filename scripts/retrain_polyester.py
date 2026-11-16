"""Compare a polyester retrain against the original V3 scanner and an original-only refit."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tensorflow as tf

LABELS = ['cotton', 'denim', 'linen', 'polyester', 'silk', 'wool']
POLY = LABELS.index('polyester')
SEED = 42
THRESHOLD = 0.85
V3_MANIFEST_SHA256 = 'f164042dcb4a0b4876fc5474532df1bb7b25403703e4a70852f19ada64c287ea'
V3_MODEL_SHA256 = '8ad5859f340c3fdd8544537aff1c2cbaad4625b380185e73f21750ac6a138325'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True, help='V3 raw folder with the six fabric subfolders')
    parser.add_argument('--manifest', type=Path, required=True, help='Original V3 split manifest')
    parser.add_argument('--baseline', type=Path, required=True, help='Published V3 MobileNetV2 Keras file')
    parser.add_argument('--selection', type=Path, default=Path(__file__).resolve().parents[1] / 'research/polyester_v3/selected_images.json')
    parser.add_argument('--additions', type=Path, help='Override the folder containing new photos, default is raw/polyester')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the candidate and report')
    parser.add_argument('--cache', type=Path, help='Optional local feature cache for repeated runs')
    return parser.parse_args()


def fingerprint(rgb):
    gray = ImageOps.grayscale(rgb).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), 'big')

def prepare(content):
    with Image.open(io.BytesIO(content)) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
        key = fingerprint(image)
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR, color=(127, 127, 127))
        return np.asarray(image), key

def original(record, raw):
    filename = record['filename']
    if not filename or Path(filename).name != filename:
        raise ValueError(f'Invalid original image filename: {filename}')
    p = raw / record['fabric'] / filename
    if not p.is_file():
        p = p.with_name(p.name.replace(' (1)', ''))
    content = p.read_bytes()
    if hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError(f'Original differs from V3 manifest: {p}')
    return prepare(content)

def baseline_features(baseline, manifest, rows, raw, cache):
    model = tf.keras.models.load_model(baseline, compile=False)
    feature_model = tf.keras.Model(model.inputs, model.get_layer('global_average_pooling2d').output)
    provenance = hashlib.sha256(baseline.read_bytes() + manifest.read_bytes()).hexdigest()
    if cache and cache.exists():
        with np.load(cache) as saved:
            if saved['provenance'].item() != provenance:
                raise ValueError('Feature cache belongs to a different baseline model or manifest.')
            return model, feature_model, saved['features'], saved['keys']
    features, keys = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i in range(0, len(rows), 24):
            pairs = list(pool.map(partial(original, raw=raw), rows[i:i+24]))
            keys.extend(pair[1] for pair in pairs)
            features.append(np.asarray(feature_model(np.stack([pair[0] for pair in pairs]), training=False)))
            if i % 480 == 0:
                print(f'Prepared {i+len(pairs)} of {len(rows)} original V3 images', flush=True)
    features = np.concatenate(features)
    keys = np.asarray(keys, np.uint64)
    if cache:
        with cache.open('wb') as handle:
            np.savez(handle, features=features, keys=keys, provenance=provenance)
    return model, feature_model, features, keys

def extra_images(feature_model, original_keys, rows, selected, extra_dir, output):
    original_digests = {row['sha256'] for row in rows}
    images, added, excluded = [], [], []
    known_keys = [int(key) for key in original_keys]
    for row in selected:
        filename, file_id = row['name'], row['id']
        if not filename or Path(filename).name != filename or not file_id or Path(file_id).name != file_id:
            raise ValueError(f'Invalid added image entry: {row}')
        path = extra_dir / filename
        if not path.is_file():
            path = extra_dir / (file_id + '.image')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row['sha256']:
            raise ValueError(f'Added image differs from the reviewed manifest: {path}')
        if digest in original_digests:
            excluded.append({**row, 'reason':'identical image'})
            continue
        try:
            image, key = prepare(data)
        except (ValueError, OSError) as err:
            excluded.append({**row, 'reason':str(err)})
            continue
        if any((key ^ original).bit_count() <= 4 for original in known_keys):
            excluded.append({**row, 'reason':'visually similar to original or accepted image by dHash'})
            continue
        original_digests.add(digest)
        known_keys.append(key)
        images.append(image)
        added.append({**row, 'sha256':digest})
    if not images:
        raise ValueError('No novel polyester images passed the preflight.')
    print(f'Accepted {len(images)} of {len(selected)} after hash and image checks', flush=True)
    (output/'added_images.json').write_text(json.dumps({'accepted':added,'excluded':excluded},indent=2), encoding='utf-8')
    vectors = [np.asarray(feature_model(np.stack(images[i:i+24]), training=False)) for i in range(0, len(images), 24)]
    return np.concatenate(vectors), len(added)

def score_summary(y, scores):
    guess = scores.argmax(axis=1)
    is_poly = y == POLY
    high = scores.max(axis=1) >= THRESHOLD
    high_poly = high & (guess == POLY)
    report = classification_report(y, guess, labels=np.arange(len(LABELS)), target_names=LABELS, zero_division=0, output_dict=True)
    return {
        'accuracy': accuracy_score(y, guess),
        'macro_f1': f1_score(y, guess, average='macro'),
        'per_class': report,
        'polyester_correct': int(((guess == POLY) & is_poly).sum()),
        'polyester_total': int(is_poly.sum()),
        'polyester_predictions': int((guess == POLY).sum()),
        'mean_polyester_score_on_polyester': float(scores[is_poly,POLY].mean()),
        'mean_polyester_score_on_other_fabrics': float(scores[~is_poly,POLY].mean()),
        'polyester_brier_score': float(np.mean(np.square(scores[:, POLY] - is_poly))),
        'high_score': {'count':int(high.sum()),'correct':int((high & (guess == y)).sum())},
        'high_score_polyester': {'count':int(high_poly.sum()),'correct':int((high_poly & is_poly).sum())},
    }

def refit(name, train_x, train_y, valid_x, valid_y, initial_weights):
    rng = np.random.default_rng(SEED)
    head = tf.keras.layers.Dense(len(LABELS),activation='softmax')
    head.build((None,train_x.shape[1]))
    head.set_weights(initial_weights)
    class_weight = len(train_y)/(len(LABELS)*np.bincount(train_y))
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function(reduce_retracing=True)
    def train_step(x, y, weight):
        with tf.GradientTape() as tape:
            objective = loss(y, head(x, training=True), sample_weight=weight)
        optimizer.apply_gradients(zip(tape.gradient(objective, head.trainable_variables), head.trainable_variables))

    best = None
    for epoch in range(1,31):
        order = rng.permutation(len(train_y))
        for start in range(0,len(order),128):
            ix = order[start:start+128]
            ys = train_y[ix]
            train_step(train_x[ix], ys, class_weight[ys].astype(np.float32))
        metrics = score_summary(valid_y, np.asarray(head(valid_x, training=False)))
        criterion = 0.5*(metrics['macro_f1']+metrics['per_class']['polyester']['f1-score'])
        if best is None or criterion > best[0]:
            best = (criterion, epoch, head.get_weights(), metrics)
    print(name,'selected epoch',best[1], 'validation macro F1',round(best[3]['macro_f1'],4),
          'validation polyester F1',round(best[3]['per_class']['polyester']['f1-score'],4),flush=True)
    return best

def main():
    tf.keras.utils.set_random_seed(SEED)
    args = arguments()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != V3_MANIFEST_SHA256:
        raise ValueError('The split manifest differs from the published V3 benchmark.')
    if hashlib.sha256(args.baseline.read_bytes()).hexdigest() != V3_MODEL_SHA256:
        raise ValueError('The baseline model differs from the published V3 scanner.')
    args.output.mkdir(parents=True, exist_ok=False)
    with args.manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    selected = json.loads(args.selection.read_text(encoding='utf-8'))
    baseline, feature_model, features, keys = baseline_features(args.baseline, args.manifest, rows, args.raw, args.cache)
    extra_dir = args.additions or args.raw / 'polyester'
    extra, extra_count = extra_images(feature_model, keys, rows, selected, extra_dir, args.output)
    y = np.asarray([LABELS.index(row['fabric']) for row in rows], dtype=np.int32)
    split = np.asarray([row['split'] for row in rows])
    train_x, train_y = features[split=='train'], y[split=='train']
    valid_x, valid_y = features[split=='valid'], y[split=='valid']
    test_x, test_y = features[split=='test'], y[split=='test']
    classifier = baseline.get_layer('dense')
    initial = classifier.get_weights()
    baseline_validation = score_summary(valid_y,np.asarray(classifier(valid_x)))
    control = refit('original-only refit',train_x,train_y,valid_x,valid_y,initial)
    candidate = refit('with new polyester',np.concatenate([train_x,extra]),
                      np.concatenate([train_y,np.full(extra_count,POLY,dtype=np.int32)]),valid_x,valid_y,initial)
    scores = {}
    for name, weights in [('current_scanner',initial),('original_only_refit',control[2]),('with_new_polyester',candidate[2])]:
        classifier.set_weights(weights)
        scores[name] = np.asarray(classifier(test_x))
    model_path=args.output/'mobilenetv2_polyester_candidate.keras'
    baseline.save(model_path)
    reopened=tf.keras.models.load_model(model_path,compile=False)
    np.testing.assert_allclose(np.asarray(reopened.get_layer('dense')(test_x)),scores['with_new_polyester'],atol=1e-7)
    mask=test_y==POLY
    old_correct=scores['current_scanner'][mask].argmax(axis=1)==POLY
    new_correct=scores['with_new_polyester'][mask].argmax(axis=1)==POLY
    gained=int((new_correct & ~old_correct).sum())
    lost=int((old_correct & ~new_correct).sum())
    report={
        'method':'Frozen deployed MobileNetV2 feature extractor and retrained six-class output layer',
        'extra_training_photos':extra_count,
        'old_training_photos':len(train_y),'validation_photos':len(valid_x),'test_photos':len(test_x),
        'selection':'Mean of six-class macro F1 and polyester F1 on the original validation set',
        'score_threshold':THRESHOLD,
        'validation':{'current_scanner':baseline_validation,'original_only_refit':control[3],'with_new_polyester':candidate[3]},
        'epochs':{'original_only_refit':control[1],'with_new_polyester':candidate[1]},
        'test':{name:score_summary(test_y,result) for name,result in scores.items()},
        'paired_polyester_test':{'gained':gained,'lost':lost,'exact_p':binomtest(gained,gained+lost,0.5).pvalue if gained+lost else 1.0},
        'candidate_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,r in report['test'].items():
        print(name, 'accuracy',round(r['accuracy'],4),'macro F1',round(r['macro_f1'],4),
              'polyester',r['polyester_correct'], '/',r['polyester_total'],
              'polyester precision',round(r['per_class']['polyester']['precision'],4),
              'mean polyester score',round(r['mean_polyester_score_on_polyester'],4),
              'high-score polyester',r['high_score_polyester'],flush=True)
    print('Paired polyester results',report['paired_polyester_test'],flush=True)

if __name__=='__main__':
    main()
