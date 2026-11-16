"""Continue development using only train/validation images, never test images."""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

assert 'mobile_paths' in globals() and 'mobile_labels' in globals(), 'Run MobileNet preparation first.'
assert tf.config.list_physical_devices('GPU'), 'This experiment needs a GPU runtime.'
tf.keras.utils.set_random_seed(SEED)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
dev_classes = list(mobile_classes)
dev_paths = {s: list(mobile_paths[s]) for s in ['train', 'valid']}
dev_labels = {s: mobile_labels[s].copy() for s in ['train', 'valid']}
dev_records = [dict(r) for r in mobile_records if r['split'] in ['train', 'valid']]
assert all(Path(p).is_file() for s in dev_paths for p in dev_paths[s]), 'Local cache is missing; rerun image preparation.'
dev_run = Path('/content/drive/MyDrive/Fabricare_training_v3') / (
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_validation_development')
dev_run.mkdir(parents=True, exist_ok=False)

def dev_decode(path, label):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    return tf.cast(tf.ensure_shape(image, [224,224,3]), tf.float32), label

def dev_dataset(split, training=False):
    ds = tf.data.Dataset.from_tensor_slices((dev_paths[split], dev_labels[split]))
    if training:
        ds = ds.shuffle(len(dev_paths[split]), seed=SEED, reshuffle_each_iteration=True)
    opts = tf.data.Options()
    opts.threading.private_threadpool_size = 4
    return ds.with_options(opts).map(dev_decode, num_parallel_calls=4).batch(32).prefetch(2)

dev_train_ds, dev_valid_ds = dev_dataset('train', True), dev_dataset('valid')
dev_supported = sorted(set(dev_labels['valid']))
def dev_metrics(probabilities):
    prediction = np.argmax(probabilities, axis=1)
    return {
        'validation_accuracy': float(accuracy_score(dev_labels['valid'], prediction)),
        'validation_macro_f1_supported_classes': float(f1_score(dev_labels['valid'], prediction,
            labels=dev_supported, average='macro', zero_division=0)),
        'classification_report': classification_report(dev_labels['valid'], prediction,
            labels=list(range(len(dev_classes))), target_names=dev_classes, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(dev_labels['valid'], prediction,
            labels=list(range(len(dev_classes)))).tolist(),
    }

print('DEVELOPMENT_SPLITS', {s: len(dev_paths[s]) for s in dev_paths}, flush=True)
print('No test images or test labels are read in this experiment.', flush=True)
dev_mobile_path = mobile_run / 'mobilenetv2_stage2_best.keras'
dev_mobile_reference = tf.keras.models.load_model(dev_mobile_path, compile=False)
dev_mobile_probs = dev_mobile_reference.predict(dev_valid_ds, verbose=0)
dev_mobile_metrics = dev_metrics(dev_mobile_probs)
print('REFERENCE_MOBILENET_VALIDATION', dev_mobile_metrics['validation_accuracy'],
    dev_mobile_metrics['validation_macro_f1_supported_classes'], flush=True)

dev_base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
dev_base.trainable = False
dev_base_name = dev_base.name
dev_inputs = tf.keras.Input(shape=(224,224,3), dtype='float32', name='rgb_0_to_255')
dev_x = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=SEED),
    tf.keras.layers.RandomRotation(0.08, fill_mode='reflect', seed=SEED+1),
    tf.keras.layers.RandomZoom(0.1, fill_mode='reflect', seed=SEED+2),
    tf.keras.layers.RandomContrast(0.1, seed=SEED+3),
], name='training_augmentation')(dev_inputs)
# EfficientNetB0 expects RGB [0,255]; its rescaling/normalization are built in.
dev_x = dev_base(dev_x, training=False)
dev_x = tf.keras.layers.GlobalAveragePooling2D()(dev_x)
dev_x = tf.keras.layers.Dropout(0.35, seed=SEED)(dev_x)
dev_outputs = tf.keras.layers.Dense(len(dev_classes), activation='softmax', dtype='float32', name='fabric_probabilities')(dev_x)
dev_model = tf.keras.Model(dev_inputs, dev_outputs, name='fabricare_efficientnetb0')
dev_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class DevCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best, self.wait, self.phase, self.history = -1.0, 0, 'head', []
    def on_train_begin(self, logs=None):
        self.wait = 0
    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(dev_valid_ds, verbose=0)
        metrics = dev_metrics(prob)
        score = metrics['validation_macro_f1_supported_classes']
        row = {'phase': self.phase, 'epoch': epoch+1,
            **{k: float(v) for k,v in (logs or {}).items()},
            'validation_macro_f1_supported_classes': score}
        self.history.append(row)
        print('DEV_EPOCH', json.dumps(row), flush=True)
        if score > self.best:
            self.best, self.wait = score, 0
            self.model.save(dev_run / 'efficientnetb0.keras')
            print('DEV_BEST', self.best, flush=True)
        else:
            self.wait += 1
            if self.wait >= 4:
                self.model.stop_training = True
        (dev_run / 'epoch_history.json').write_text(json.dumps(self.history, indent=2))

dev_callback = DevCheckpoint()
dev_train_sizes = np.bincount(dev_labels['train'], minlength=len(dev_classes))
dev_weights = {i: float(min(3.0, np.sqrt(len(dev_labels['train'])/(len(dev_classes)*n))))
    for i,n in enumerate(dev_train_sizes)}
print('DEV_PHASE frozen backbone, up to 6 epochs', flush=True)
dev_model.fit(dev_train_ds, validation_data=dev_valid_ds, epochs=6,
    class_weight=dev_weights, callbacks=[dev_callback, tf.keras.callbacks.TerminateOnNaN()], verbose=2)

dev_model = tf.keras.models.load_model(dev_run / 'efficientnetb0.keras', compile=False)
dev_base = dev_model.get_layer(dev_base_name)
dev_base.trainable = True
for layer in dev_base.layers[:-60]:
    layer.trainable = False
for layer in dev_base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
dev_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dev_callback.phase = 'fine_tune'
print('DEV_PHASE fine tuning, up to 10 epochs', flush=True)
dev_model.fit(dev_train_ds, validation_data=dev_valid_ds, epochs=10,
    class_weight=dev_weights, callbacks=[dev_callback, tf.keras.callbacks.TerminateOnNaN()], verbose=2)

dev_model = tf.keras.models.load_model(dev_run / 'efficientnetb0.keras', compile=False)
dev_eff_probs = dev_model.predict(dev_valid_ds, verbose=0)
dev_comparisons = []
for efficientnet_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
    averaged = efficientnet_weight*dev_eff_probs + (1-efficientnet_weight)*dev_mobile_probs
    dev_comparisons.append({'efficientnet_weight': efficientnet_weight,
        'mobilenet_weight': 1-efficientnet_weight, **dev_metrics(averaged)})
dev_selected = max(dev_comparisons, key=lambda r: r['validation_macro_f1_supported_classes'])
dev_selected_weight = dev_selected['efficientnet_weight']
dev_selected_probs = dev_selected_weight*dev_eff_probs + (1-dev_selected_weight)*dev_mobile_probs
shutil.copy2(dev_mobile_path, dev_run / 'mobilenetv2_reference.keras')
dev_report = {
    'experiment': 'EfficientNetB0 and MobileNetV2 comparison using training and validation only',
    'selection_metric': 'validation macro F1 on supported classes',
    'test_evaluated': False, 'deployment_target_verified': False,
    'classes': dev_classes, 'counts': {s: {name: int(np.sum(dev_labels[s]==i))
        for i,name in enumerate(dev_classes)} for s in ['train','valid']},
    'epochs': dev_callback.history, 'comparisons': dev_comparisons, 'selected': dev_selected,
    'seed': SEED, 'tensorflow': tf.__version__, 'numpy': np.__version__,
    'missing_validation_classes': [name for i,name in enumerate(dev_classes) if i not in dev_supported],
    'limitations': ['Linen has no validation images.',
        'Repeated selection on the existing validation set can overfit this development set.',
        '95% deployment accuracy is not established; new independent garment-level test data is required.',
        'Garment identity and non-burst near-duplicates remain unverified.'],
}
dev_config = {
    'classes': dev_classes, 'input': 'RGB float32 [0,255], stretch to 224x224 using OpenCV INTER_LINEAR',
    'preprocessing': 'embedded separately in each Keras model; do not add normalization',
    'inference': 'model(rgb_batch, training=False)',
    'probabilities': 'efficientnet_weight * efficientnet_probabilities + mobilenet_weight * mobilenet_probabilities',
    'efficientnet_weight': dev_selected_weight, 'mobilenet_weight': 1-dev_selected_weight,
    'efficientnet_model': 'efficientnetb0.keras', 'mobilenet_model': 'mobilenetv2_reference.keras',
    'status': 'development only; not deployment validated',
}
for name,payload in [('training_report.json',dev_report), ('inference_config.json',dev_config),
        ('fabric_labels.json',dev_classes), ('train_validation_manifest.json',dev_records)]:
    (dev_run/name).write_text(json.dumps(payload,indent=2))
np.savez_compressed(dev_run/'validation_predictions.npz', truth=dev_labels['valid'],
    efficientnet=dev_eff_probs, mobilenet=dev_mobile_probs, selected=dev_selected_probs)
dev_reload = tf.keras.models.load_model(dev_run/'efficientnetb0.keras',compile=False)
dev_probe = next(iter(dev_valid_ds.take(1)))[0]
np.testing.assert_allclose(dev_reload(dev_probe,training=False).numpy(),
    dev_model(dev_probe,training=False).numpy(),rtol=1e-4,atol=1e-5)
print('DEV_REFERENCE_VALIDATION_ACCURACY',dev_mobile_metrics['validation_accuracy'],flush=True)
print('DEV_EFFICIENTNET_VALIDATION_ACCURACY',dev_metrics(dev_eff_probs)['validation_accuracy'],flush=True)
print('DEV_SELECTED_VALIDATION_ACCURACY',dev_selected['validation_accuracy'],flush=True)
print('DEV_SELECTED_VALIDATION_MACRO_F1',dev_selected['validation_macro_f1_supported_classes'],flush=True)
print('DEV_SELECTED_WEIGHTS',dev_selected_weight,1-dev_selected_weight,flush=True)
print('DEV_EXPORT_VERIFIED',dev_run,flush=True)
print('No test evaluation performed. 95% deployment target remains unverified.',flush=True)
dev_archive = shutil.make_archive('/content/'+dev_run.name,'zip',dev_run)
from google.colab import files
files.download(dev_archive)
