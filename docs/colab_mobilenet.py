"""MobileNetV2 experiment; run after the classifier comparison in Colab."""
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

assert 'improved_manifest' in globals(), 'Run the dataset checks and burst-filtered comparison first.'
assert tf.config.list_physical_devices('GPU'), 'Connect a GPU runtime before this training run.'
tf.keras.utils.set_random_seed(SEED)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
MOBILE_SIZE, MOBILE_BATCH = 224, 32
HEAD_EPOCHS, FINE_EPOCHS = 6, 8
mobile_records = [dict(r) for r in improved_manifest]
mobile_classes = list(CLASS_NAMES)
mobile_label_to_id = {name: i for i, name in enumerate(mobile_classes)}
mobile_run = Path('/content/drive/MyDrive/Fabricare_training') / (
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_mobilenetv2')
mobile_run.mkdir(parents=True, exist_ok=False)
mobile_cache = Path('/content/fabricare_mobilenet_224_png')
mobile_cache.mkdir(exist_ok=True)

# Stage resized images on Colab's local disk; original Drive photos are untouched.
# Cache keys include the decoded hash produced by the dataset audit.
def stage_mobile_image(record):
    cached = mobile_cache / (record['decoded_sha256'] + '.png')
    if not cached.exists():
        source = DATASET_DIR / record['path']
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f'Unreadable source photo: {source}')
        actual_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if actual_hash != record['decoded_sha256']:
            raise ValueError(f'Photo changed since dataset audit: {source}; rerun the audit.')
        image = cv2.resize(image, (MOBILE_SIZE, MOBILE_SIZE), interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(cached), image):
            raise IOError(f'Could not stage image: {source}')
    return str(cached)

stage_started = time.monotonic()
print('STAGING', len(mobile_records), 'images to local runtime disk', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    staged = []
    for i, path in enumerate(pool.map(stage_mobile_image, mobile_records), 1):
        staged.append(path)
        if i % 250 == 0 or i == len(mobile_records):
            print('STAGED', i, '/', len(mobile_records), 'elapsed seconds:', round(time.monotonic()-stage_started), flush=True)

mobile_paths, mobile_labels = {}, {}
for split in ['train', 'valid', 'test']:
    indexes = [i for i, r in enumerate(mobile_records) if r['split'] == split]
    mobile_paths[split] = [staged[i] for i in indexes]
    mobile_labels[split] = np.asarray([mobile_label_to_id[mobile_records[i]['label']] for i in indexes], dtype=np.int32)
    assert indexes, f'Empty {split} split.'
    print(split, 'samples:', len(indexes), flush=True)
assert set(mobile_labels['train']) == set(range(len(mobile_classes))), 'Missing training class.'

def mobile_decode(path, label):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.ensure_shape(image, [MOBILE_SIZE, MOBILE_SIZE, 3])
    return tf.cast(image, tf.float32), label

def mobile_dataset(split, training=False):
    ds = tf.data.Dataset.from_tensor_slices((mobile_paths[split], mobile_labels[split]))
    if training:
        ds = ds.shuffle(len(mobile_paths[split]), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    ds = ds.with_options(options)
    return ds.map(mobile_decode, num_parallel_calls=4).batch(MOBILE_BATCH).prefetch(2)

mobile_train_ds = mobile_dataset('train', True)
mobile_valid_ds = mobile_dataset('valid')
mobile_test_ds = mobile_dataset('test')
class_sizes = np.bincount(mobile_labels['train'], minlength=len(mobile_classes))
# Square-root weighting, capped at 3, avoids letting 28 linen photos dominate.
mobile_class_weight = {i: float(min(3.0, np.sqrt(len(mobile_labels['train']) / (len(mobile_classes)*n))))
    for i, n in enumerate(class_sizes)}
print('Class weights:', dict(zip(mobile_classes, mobile_class_weight.values())), flush=True)

mobile_backbone = tf.keras.applications.MobileNetV2(input_shape=(MOBILE_SIZE, MOBILE_SIZE, 3),
    include_top=False, weights='imagenet')
mobile_backbone.trainable = False
mobile_backbone_name = mobile_backbone.name
inputs = tf.keras.Input(shape=(MOBILE_SIZE, MOBILE_SIZE, 3), dtype='float32', name='rgb_0_to_255')
x = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=SEED),
    tf.keras.layers.RandomRotation(0.1, fill_mode='reflect', seed=SEED+1),
    tf.keras.layers.RandomZoom(0.1, fill_mode='reflect', seed=SEED+2),
    tf.keras.layers.RandomContrast(0.1, seed=SEED+3),
], name='training_augmentation')(inputs)
# MobileNetV2 preprocessing embedded in the saved model: pixel / 127.5 - 1.
x = tf.keras.layers.Rescaling(1.0/127.5, offset=-1.0, name='mobilenet_preprocessing')(x)
x = mobile_backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35, seed=SEED)(x)
outputs = tf.keras.layers.Dense(len(mobile_classes), activation='softmax', dtype='float32', name='fabric_probabilities')(x)
mobile_model = tf.keras.Model(inputs, outputs, name='fabricare_mobilenetv2')
mobile_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class MobileValidationCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best = -1.0
        self.phase = 'head'
        self.rows = []
        self.wait = 0
    def on_train_begin(self, logs=None):
        self.wait = 0
    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(mobile_valid_ds, verbose=0)
        prediction = np.argmax(probabilities, axis=1)
        score = float(f1_score(mobile_labels['valid'], prediction,
            labels=sorted(set(mobile_labels['valid'])), average='macro', zero_division=0))
        row = {'phase': self.phase, 'epoch': epoch+1,
            **{k: float(v) for k, v in (logs or {}).items()},
            'validation_macro_f1_supported_classes': score}
        self.rows.append(row)
        print('MOBILE_EPOCH', json.dumps(row), flush=True)
        if score > self.best:
            self.best, self.wait = score, 0
            self.model.save(mobile_run / 'fabric_mobilenetv2.keras')
            print('MOBILE_BEST_CHECKPOINT', self.best, flush=True)
        else:
            self.wait += 1
            if self.wait >= 4:
                self.model.stop_training = True
        (mobile_run / 'epoch_history.json').write_text(json.dumps(self.rows, indent=2))

mobile_checkpoint = MobileValidationCheckpoint()
print('MOBILE_PHASE head training; epochs:', HEAD_EPOCHS, flush=True)
mobile_model.fit(mobile_train_ds, validation_data=mobile_valid_ds,
    epochs=HEAD_EPOCHS, class_weight=mobile_class_weight,
    callbacks=[mobile_checkpoint, tf.keras.callbacks.TerminateOnNaN()], verbose=2)

# Fine-tune from the strongest validation checkpoint, not the last head epoch.
mobile_model = tf.keras.models.load_model(mobile_run / 'fabric_mobilenetv2.keras', compile=False)
mobile_backbone = mobile_model.get_layer(mobile_backbone_name)
mobile_backbone.trainable = True
for layer in mobile_backbone.layers[:-30]:
    layer.trainable = False
for layer in mobile_backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
mobile_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mobile_checkpoint.phase = 'fine_tune'
print('MOBILE_PHASE fine tuning; epochs:', FINE_EPOCHS, flush=True)
mobile_model.fit(mobile_train_ds, validation_data=mobile_valid_ds,
    epochs=FINE_EPOCHS, class_weight=mobile_class_weight,
    callbacks=[mobile_checkpoint, tf.keras.callbacks.TerminateOnNaN()], verbose=2)

mobile_model = tf.keras.models.load_model(mobile_run / 'fabric_mobilenetv2.keras', compile=False)
# Test labels are first used for scoring only after checkpoint selection ends.
mobile_test_probs = mobile_model.predict(mobile_test_ds, verbose=0)
mobile_test_prediction = np.argmax(mobile_test_probs, axis=1)
mobile_test_truth = mobile_labels['test']
mobile_metrics = {
    'accuracy': float(accuracy_score(mobile_test_truth, mobile_test_prediction)),
    'macro_f1_supported_classes': float(f1_score(mobile_test_truth, mobile_test_prediction,
        labels=sorted(set(mobile_test_truth)), average='macro', zero_division=0)),
    'classification_report': classification_report(mobile_test_truth, mobile_test_prediction,
        labels=list(range(len(mobile_classes))), target_names=mobile_classes,
        output_dict=True, zero_division=0),
    'confusion_matrix': confusion_matrix(mobile_test_truth, mobile_test_prediction,
        labels=list(range(len(mobile_classes)))).tolist(),
}
mobile_report = {
    'architecture': 'ImageNet-pretrained MobileNetV2, 224x224 RGB', 'seed': SEED,
    'tensorflow': tf.__version__, 'numpy': np.__version__, 'classes': mobile_classes,
    'epochs': mobile_checkpoint.rows, 'best_validation_macro_f1_supported_classes': mobile_checkpoint.best,
    'test': mobile_metrics, 'test_count': len(mobile_test_truth),
    'class_counts': {s: {label: int(np.sum(mobile_labels[s] == i)) for i, label in enumerate(mobile_classes)}
        for s in ['train', 'valid', 'test']},
    'preprocessing': {'input_size': [224,224], 'color': 'RGB', 'input_range': [0,255],
        'resize': 'OpenCV INTER_LINEAR, stretch to 224x224', 'scaling': 'embedded: x/127.5-1',
        'augmentation': 'training only; disabled by model(..., training=False)'},
    'missing_evaluation_classes': {s: [label for i,label in enumerate(mobile_classes) if i not in mobile_labels[s]]
        for s in ['valid','test']},
    'limitations': ['Linen has no validation/test images.',
        'Garment identity and non-burst near duplicates are unverified.',
        'This test set was evaluated in prior experiments; a fresh independent test set is required for final claims.',
        'This Keras model needs a new backend inference path; it cannot replace the existing sklearn pickle.'],
}
(mobile_run / 'fabric_labels.json').write_text(json.dumps(mobile_classes, indent=2))
(mobile_run / 'training_report.json').write_text(json.dumps(mobile_report, indent=2))
(mobile_run / 'dataset_manifest.json').write_text(json.dumps(mobile_records, indent=2))
np.savez_compressed(mobile_run / 'test_predictions.npz', truth=mobile_test_truth, probabilities=mobile_test_probs)
# Verify saved model inference against the in-memory selected checkpoint.
mobile_loaded = tf.keras.models.load_model(mobile_run / 'fabric_mobilenetv2.keras', compile=False)
probe = next(iter(mobile_test_ds.take(1)))[0]
np.testing.assert_allclose(mobile_loaded(probe, training=False).numpy(),
    mobile_model(probe, training=False).numpy(), rtol=1e-4, atol=1e-5)
print('MOBILE_TEST_ACCURACY', mobile_metrics['accuracy'], flush=True)
print('MOBILE_TEST_MACRO_F1', mobile_metrics['macro_f1_supported_classes'], flush=True)
print(classification_report(mobile_test_truth, mobile_test_prediction,
    labels=list(range(len(mobile_classes))), target_names=mobile_classes, zero_division=0), flush=True)
print('MOBILE_EXPORT_VERIFIED', mobile_run, flush=True)
mobile_archive = shutil.make_archive('/content/' + mobile_run.name, 'zip', mobile_run)
from google.colab import files
files.download(mobile_archive)
