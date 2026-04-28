import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore[import]
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
import numpy as np
import pandas as pd
import os
from typing import Callable
from cnn.augmentations import apply_domain_aware_training_augmentations, apply_training_augmentations
from cnn.preprocessing import get_preprocess_fn
from cnn.split_utils import load_or_create_split_manifest

try:
    from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
except ImportError:
    convnext_preprocess = None

try:
    from transformers import TFCLIPVisionModel
except ImportError:
    TFCLIPVisionModel = None


TRANSFER_ARCHS = {"efficientnet_b0", "efficientnet_v2b0", "resnet50", "convnext_tiny", "clip_vit_b32"}


@tf.keras.utils.register_keras_serializable(package="aid")
class SobelMagnitudeLayer(layers.Layer):
    """Compute a normalized 3-channel Sobel magnitude map from RGB inputs."""

    def call(self, inputs):
        gray = tf.image.rgb_to_grayscale(inputs)
        sobel = tf.image.sobel_edges(gray)
        dy = sobel[..., 0]
        dx = sobel[..., 1]
        edges = tf.sqrt(tf.square(dx) + tf.square(dy))
        edges = tf.squeeze(edges, axis=-1)
        edges = edges / (tf.reduce_max(edges, axis=[1, 2], keepdims=True) + 1e-7)
        edges = tf.expand_dims(edges, axis=-1)
        return tf.concat([edges, edges, edges], axis=-1)


@tf.keras.utils.register_keras_serializable(package="aid")
class HaarWaveletLayer(layers.Layer):
    """Compute a simple single-level Haar-like detail representation from RGB inputs."""

    def call(self, inputs):
        gray = tf.image.rgb_to_grayscale(inputs)
        top_left = gray[:, 0::2, 0::2, :]
        top_right = gray[:, 0::2, 1::2, :]
        bottom_left = gray[:, 1::2, 0::2, :]
        bottom_right = gray[:, 1::2, 1::2, :]

        horizontal = tf.abs((top_left - top_right + bottom_left - bottom_right) / 4.0)
        vertical = tf.abs((top_left + top_right - bottom_left - bottom_right) / 4.0)
        diagonal = tf.abs((top_left - top_right - bottom_left + bottom_right) / 4.0)

        detail = tf.concat([horizontal, vertical, diagonal], axis=-1)
        detail = detail / (tf.reduce_max(detail, axis=[1, 2, 3], keepdims=True) + 1e-7)
        return tf.image.resize(detail, tf.shape(inputs)[1:3], method="bilinear", antialias=False)


def preprocess_regular(path, label, image_size):
    """Compatibility wrapper around the shared RGB preprocessing path."""
    return get_preprocess_fn("rgb")(path, label, image_size)


def preprocess_sobel_edge(path, label, image_size):
    """Compatibility wrapper around the shared Sobel preprocessing path."""
    return get_preprocess_fn("sobel")(path, label, image_size)


def build_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    # Build optimized CNN model with BatchNormalization and improved architecture - returns compiled Keras model
    model = models.Sequential(
        [
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Third convolutional block
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Fourth convolutional block
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def build_efficientnet_b0_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Build a single-backbone EfficientNetB0 classifier head for transfer learning."""
    inputs = layers.Input(shape=input_shape)
    backbone = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.2)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_classifier")
    model._backbone = backbone  # type: ignore[attr-defined]
    return model


def build_efficientnet_v2b0_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Build an EfficientNetV2B0 transfer-learning classifier head."""
    inputs = layers.Input(shape=input_shape)
    backbone = tf.keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.2)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_v2b0_classifier")
    model._backbone = backbone  # type: ignore[attr-defined]
    return model


def build_clip_vit_b32_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Build a universal-style baseline with a frozen CLIP vision encoder and a trainable classification head."""
    if TFCLIPVisionModel is None:
        raise ValueError("transformers is required for clip_vit_b32 but is not installed.")

    inputs = layers.Input(shape=input_shape)
    backbone = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    backbone.trainable = False
    pixel_values = layers.Lambda(lambda tensor: tf.transpose(tensor, perm=[0, 3, 1, 2]), name="clip_channels_first")(inputs)
    vision_outputs = backbone(pixel_values=pixel_values, training=False)
    pooled = vision_outputs.pooler_output
    pooled = layers.Dropout(0.2)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="clip_vit_b32_classifier")
    model._backbone = backbone  # type: ignore[attr-defined]
    return model


def _small_conv_branch(inputs: tf.Tensor, filters: tuple[int, int], prefix: str) -> tf.Tensor:
    x = inputs
    for index, filter_count in enumerate(filters, start=1):
        x = layers.Conv2D(filter_count, (3, 3), padding="same", name=f"{prefix}_conv_{index}")(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn_{index}")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu_{index}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{prefix}_pool_{index}")(x)
        x = layers.Dropout(0.2, name=f"{prefix}_dropout_{index}")(x)
    return layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)


def build_dual_artifact_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """A task-specific RGB plus artifact-fusion detector for synthetic-image cues."""
    inputs = layers.Input(shape=input_shape)

    rgb_features = _small_conv_branch(inputs, filters=(32, 64, 96), prefix="rgb")
    sobel_features = _small_conv_branch(SobelMagnitudeLayer(name="sobel_features")(inputs), filters=(16, 32), prefix="sobel")
    wavelet_features = _small_conv_branch(HaarWaveletLayer(name="wavelet_features")(inputs), filters=(16, 32), prefix="wavelet")

    fused = layers.Concatenate(name="artifact_fusion")([rgb_features, sobel_features, wavelet_features])
    fused = layers.Dense(256, activation="relu", name="fusion_dense")(fused)
    fused = layers.Dropout(0.4, name="fusion_dropout")(fused)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(fused)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dual_artifact_cnn")


def build_resnet50_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Build a ResNet50 transfer-learning classifier head."""
    inputs = layers.Input(shape=input_shape)
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.3)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_classifier")
    model._backbone = backbone  # type: ignore[attr-defined]
    return model


def build_convnext_tiny_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Build a ConvNeXtTiny transfer-learning classifier head when supported by the TensorFlow runtime."""
    if not hasattr(tf.keras.applications, "ConvNeXtTiny"):
        raise ValueError("ConvNeXtTiny is unavailable in this TensorFlow runtime.")

    inputs = layers.Input(shape=input_shape)
    backbone = tf.keras.applications.ConvNeXtTiny(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.3)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="convnext_tiny_classifier")
    model._backbone = backbone  # type: ignore[attr-defined]
    return model


def build_model(arch: str = "simple", input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Select the requested single-backbone architecture."""
    normalized_arch = arch.strip().lower()
    if normalized_arch == "simple":
        return build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    if normalized_arch == "dual_artifact_cnn":
        return build_dual_artifact_cnn_model(input_shape=input_shape, num_classes=num_classes)
    if normalized_arch == "efficientnet_b0":
        return build_efficientnet_b0_model(input_shape=input_shape, num_classes=num_classes)
    if normalized_arch == "efficientnet_v2b0":
        return build_efficientnet_v2b0_model(input_shape=input_shape, num_classes=num_classes)
    if normalized_arch == "clip_vit_b32":
        return build_clip_vit_b32_model(input_shape=input_shape, num_classes=num_classes)
    if normalized_arch == "resnet50":
        return build_resnet50_model(input_shape=input_shape, num_classes=num_classes)
    if normalized_arch == "convnext_tiny":
        return build_convnext_tiny_model(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unsupported architecture: {arch}")


def _apply_arch_preprocessing(img: tf.Tensor, label: tf.Tensor, arch: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply architecture-specific normalization after base preprocessing and augmentation."""
    normalized_arch = arch.strip().lower()
    img = tf.cast(img, tf.float32)
    if normalized_arch == "efficientnet_b0":
        img = efficientnet_preprocess(img * 255.0)
    elif normalized_arch == "efficientnet_v2b0":
        img = efficientnet_v2_preprocess(img * 255.0)
    elif normalized_arch == "clip_vit_b32":
        clip_mean = tf.constant([0.48145466, 0.4578275, 0.40821073], dtype=tf.float32)
        clip_std = tf.constant([0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)
        img = (img - clip_mean) / clip_std
    elif normalized_arch == "resnet50":
        img = resnet_preprocess(img * 255.0)
    elif normalized_arch == "convnext_tiny":
        if convnext_preprocess is not None:
            img = convnext_preprocess(img * 255.0)
        else:
            img = img * 255.0
    return img, label


def _supports_staged_finetune(arch: str) -> bool:
    return arch.strip().lower() in TRANSFER_ARCHS


def _resample_train_frame(train_frame: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    """Reweight manifest rows to reduce domain imbalance without changing epoch length."""
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy == "none":
        return train_frame.reset_index(drop=True)

    if train_frame.empty:
        return train_frame.reset_index(drop=True)

    if normalized_strategy not in {"domain_balanced", "domain_label_balanced"}:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    if "domain" not in train_frame.columns:
        return train_frame.reset_index(drop=True)

    if normalized_strategy == "domain_balanced":
        group_keys = train_frame["domain"].astype(str)
    else:
        group_keys = train_frame["domain"].astype(str) + "||" + train_frame["label"].astype(str)

    group_counts = group_keys.value_counts()
    if group_counts.empty:
        return train_frame.reset_index(drop=True)

    weights = group_keys.map(lambda key: 1.0 / float(group_counts[key])).astype(float).to_numpy()
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return train_frame.reset_index(drop=True)

    probabilities = weights / weight_sum
    rng = np.random.default_rng(seed)
    sampled_positions = rng.choice(
        len(train_frame),
        size=len(train_frame),
        replace=True,
        p=probabilities,
    )
    return train_frame.iloc[sampled_positions].reset_index(drop=True)


def _build_optimizer(learning_rate, weight_decay: float):
    """Build AdamW when available, otherwise fall back to Adam."""
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    except AttributeError:
        print("Warning: tf.keras.optimizers.AdamW is unavailable; falling back to Adam without weight decay.")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _compile_model(model: tf.keras.Model, optimizer) -> None:
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])


def _merge_histories(primary: callbacks.History, secondary: callbacks.History) -> callbacks.History:
    for key, values in secondary.history.items():
        primary.history.setdefault(key, [])
        primary.history[key].extend(values)
    return primary


class EpochSummaryLogger(callbacks.Callback):
    """Print concise epoch summaries that remain readable even when stderr is noisy."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        summary_parts = [f"Epoch {epoch + 1} summary"]
        for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
            value = logs.get(key)
            if value is not None:
                summary_parts.append(f"{key}={value:.4f}")
        print(" | ".join(summary_parts))


def _balanced_accuracy_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> float:
    unique_labels = sorted(set(labels.tolist()))
    recalls: list[float] = []
    for class_id in unique_labels:
        class_mask = labels == class_id
        class_total = int(np.sum(class_mask))
        if class_total == 0:
            continue
        recalls.append(float(np.sum(predictions[class_mask] == class_id)) / float(class_total))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def _build_slice_eval_dataset(
    frame: pd.DataFrame,
    preprocess_callable: Callable,
    image_size: tuple[int, int],
    batch_size: int,
    arch: str,
    parallel_calls: int,
) -> tf.data.Dataset:
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].astype(int).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: preprocess_callable(path, label, image_size),
        num_parallel_calls=parallel_calls,
        deterministic=False,
    )
    dataset = dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])
    dataset = dataset.ignore_errors()
    dataset = dataset.map(
        lambda img, label: _apply_arch_preprocessing(img, label, arch),
        num_parallel_calls=parallel_calls,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(1)
    return dataset


class SliceValidationLogger(callbacks.Callback):
    """Log dataset/domain slice metrics at the end of each epoch to expose weak generalization regions."""

    def __init__(
        self,
        val_frame: pd.DataFrame,
        preprocess_callable: Callable,
        image_size: tuple[int, int],
        batch_size: int,
        arch: str,
        parallel_calls: int,
        slice_columns: tuple[str, ...] = ("domain", "dataset_id"),
    ) -> None:
        super().__init__()
        self.val_frame = val_frame.copy()
        self.preprocess_callable = preprocess_callable
        self.image_size = image_size
        self.batch_size = batch_size
        self.arch = arch
        self.parallel_calls = parallel_calls
        self.slice_columns = slice_columns

    def on_epoch_end(self, epoch, logs=None):
        del logs
        for column in self.slice_columns:
            if column not in self.val_frame.columns:
                continue
            values = [value for value in sorted(self.val_frame[column].dropna().astype(str).unique().tolist()) if value.strip()]
            if not values:
                continue

            summaries: list[str] = []
            for value in values:
                slice_frame = self.val_frame[self.val_frame[column].astype(str) == value].copy()
                if slice_frame.empty:
                    continue
                dataset = _build_slice_eval_dataset(
                    frame=slice_frame,
                    preprocess_callable=self.preprocess_callable,
                    image_size=self.image_size,
                    batch_size=self.batch_size,
                    arch=self.arch,
                    parallel_calls=self.parallel_calls,
                )
                probabilities = np.asarray(self.model.predict(dataset, verbose=0), dtype=float)
                if probabilities.ndim == 1:
                    predictions = (probabilities >= 0.5).astype(int)
                else:
                    predictions = np.argmax(probabilities, axis=1).astype(int)
                labels = slice_frame["label"].astype(int).to_numpy()
                sample_count = min(len(labels), len(predictions))
                labels = labels[:sample_count]
                predictions = predictions[:sample_count]
                if sample_count == 0:
                    continue
                accuracy = float(np.mean(predictions == labels))
                balanced_accuracy = _balanced_accuracy_from_predictions(labels, predictions)
                summaries.append(f"{value}:acc={accuracy:.3f},bal={balanced_accuracy:.3f},n={sample_count}")

            if summaries:
                print(f"Epoch {epoch + 1} {column} slices | " + " | ".join(summaries))


def _get_pipeline_settings(has_gpu: bool, batch_size: int, enable_augmentation: bool, preprocess_mode: str) -> tuple[int, int, int]:
    """Cap tf.data parallelism to avoid excessive pinned host-memory growth on WSL/Docker."""
    parallel_calls = 2
    prefetch_batches = 1
    shuffle_buffer = 256

    if has_gpu:
        parallel_calls = 4
        prefetch_batches = 1
        shuffle_buffer = 256

    if enable_augmentation or "wavelet" in preprocess_mode:
        parallel_calls = min(parallel_calls, 2)
        prefetch_batches = 1
        shuffle_buffer = min(shuffle_buffer, 128)

    if batch_size >= 16:
        parallel_calls = min(parallel_calls, 2)
        prefetch_batches = 1
        shuffle_buffer = min(shuffle_buffer, 128)

    return parallel_calls, prefetch_batches, shuffle_buffer


def train_model(
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.2,
    use_cache: bool = True,
    cache_in_memory: bool = False,
    use_mixed_precision: bool = True,
    enable_augmentation: bool = False,
    model_save_path: str = "",
    preprocess_func: Callable | None = None,
    image_size: tuple = (224, 224),
    enable_early_stopping: bool = True,
    seed: int = 42,
    label_mapping: dict[str, int] | None = None,
    preprocess_mode: str = "rgb",
    split_manifest_path: str | None = None,
    regen_split: bool = False,
    allow_unknown: bool = False,
    arch: str = "simple",
    finetune_unfreeze: bool = False,
    finetune_freeze_epochs: int = 3,
    finetune_lr: float = 1e-5,
    finetune_weight_decay: float = 1e-5,
    early_stopping_patience: int = 1,
    sampling_strategy: str = "domain_balanced",
    enable_slice_logging: bool = True,
) -> tuple[tf.keras.Model, callbacks.History]:
    # Train the CNN model for AI-generated image detection - returns trained model and training history
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    mixed_precision_enabled = bool(use_mixed_precision and gpus)

    # Enable mixed precision only when a GPU is available.
    if mixed_precision_enabled:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled for GPU performance")

    # Configure GPU memory growth to avoid OOM errors
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    elif use_mixed_precision:
        print("No GPU detected; mixed precision disabled for this run.")

    if label_mapping is None:
        raise ValueError("train_model requires an explicit label_mapping.")
    num_classes = max(label_mapping.values()) + 1
    pipeline_parallel_calls, prefetch_batches, shuffle_buffer = _get_pipeline_settings(
        has_gpu=bool(gpus),
        batch_size=batch_size,
        enable_augmentation=enable_augmentation,
        preprocess_mode=preprocess_mode,
    )

    preprocess_callable = preprocess_func or get_preprocess_fn(preprocess_mode)
    manifest, manifest_path = load_or_create_split_manifest(
        dataset_path=dataset_path,
        label_mapping=label_mapping,
        validation_split=validation_split,
        seed=seed,
        manifest_path=split_manifest_path or None,
        regen_split=regen_split,
        allow_unknown=allow_unknown,
    )

    usable_manifest = manifest[(manifest["split"].isin(["train", "val"])) & (manifest["label"] >= 0)].copy()
    train_frame = usable_manifest[usable_manifest["split"] == "train"].copy()
    val_frame = usable_manifest[usable_manifest["split"] == "val"]
    train_frame = _resample_train_frame(train_frame, strategy=sampling_strategy, seed=seed)

    train_paths = train_frame["path"].to_numpy()
    train_labels = train_frame["label"].astype(int).to_numpy()
    train_domains = train_frame.get("domain", pd.Series([""] * len(train_frame))).fillna("").astype(str).to_numpy()
    val_paths = val_frame["path"].to_numpy()
    val_labels = val_frame["label"].astype(int).to_numpy()
    num_samples = len(usable_manifest)

    # Create train dataset with optimized pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels, train_domains))

    # Map: Load and preprocess images (GPU-accelerated, parallel)
    # Ignore errors for corrupt images (skip them instead of crashing)
    train_dataset = train_dataset.map(
        lambda path, label, domain: (*preprocess_callable(path, label, image_size), domain), num_parallel_calls=pipeline_parallel_calls, deterministic=False
    )

    # Filter out any invalid images (shape mismatches, etc.)
    # Note: Corrupt JPEG files will be caught by decode_jpeg and cause an error
    # We need to handle this at the dataset level
    train_dataset = train_dataset.filter(lambda img, label, domain: tf.shape(img)[0] == image_size[0])

    # Ignore errors for corrupt images - skip them instead of crashing
    train_dataset = train_dataset.ignore_errors()

    # Apply data augmentation if enabled (only for training)
    if enable_augmentation:
        train_dataset = train_dataset.map(
            apply_domain_aware_training_augmentations,
            num_parallel_calls=pipeline_parallel_calls,
        )

    train_dataset = train_dataset.map(
        lambda img, label, domain: _apply_arch_preprocessing(img, label, arch),
        num_parallel_calls=pipeline_parallel_calls,
    )

    # Batch: Group into batches
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)

    # Cache: Store preprocessed images (memory or disk based on parameter)
    if use_cache:
        if cache_in_memory:
            train_dataset = train_dataset.cache()  # Memory cache
        else:
            # Use disk cache only if dataset is reasonably sized (< 100K images)
            if num_samples < 100000:
                train_dataset = train_dataset.cache(os.path.join(dataset_path, "train_cache"))
            else:
                print("Warning: Dataset too large for disk cache, skipping cache for better performance")

    # Shuffle: Randomize order (after cache for efficiency)
    train_dataset = train_dataset.shuffle(
        buffer_size=min(shuffle_buffer, len(train_paths)),
        seed=seed,
        reshuffle_each_iteration=True,
    )  # TODO Change buffer size?

    # Prefetch: Prepare next batch while GPU is training
    train_dataset = train_dataset.prefetch(prefetch_batches)

    # Create val dataset with optimized pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    # Map: Load and preprocess images (GPU-accelerated, parallel)
    val_dataset = val_dataset.map(lambda path, label: preprocess_callable(path, label, image_size), num_parallel_calls=pipeline_parallel_calls, deterministic=False)

    # Filter out any invalid images
    val_dataset = val_dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])

    # Ignore errors for corrupt images
    val_dataset = val_dataset.ignore_errors()

    val_dataset = val_dataset.map(
        lambda img, label: _apply_arch_preprocessing(img, label, arch),
        num_parallel_calls=pipeline_parallel_calls,
    )

    # Batch: Group into batches (no shuffle for validation)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)

    # Cache: Store preprocessed images
    if use_cache:
        if cache_in_memory:
            val_dataset = val_dataset.cache()  # Memory cache
        else:
            if num_samples < 100000:
                val_dataset = val_dataset.cache(os.path.join(dataset_path, "val_cache"))
            else:
                print("Warning: Dataset too large for disk cache, skipping cache for better performance")

    # Prefetch: Prepare next batch while GPU is validating
    val_dataset = val_dataset.prefetch(prefetch_batches)

    # Build model
    model = build_model(
        arch=arch,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
    )

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True)
    optimizer = _build_optimizer(learning_rate=lr_schedule, weight_decay=1e-4)
    _compile_model(model, optimizer)

    # Setup callbacks
    callback_list = []
    callback_list.append(EpochSummaryLogger())
    if enable_slice_logging:
        callback_list.append(
            SliceValidationLogger(
                val_frame=val_frame,
                preprocess_callable=preprocess_callable,
                image_size=image_size,
                batch_size=batch_size,
                arch=arch,
                parallel_calls=pipeline_parallel_calls,
            )
        )

    # Early stopping to prevent overfitting
    if enable_early_stopping:
        early_stopping = callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=early_stopping_patience, restore_best_weights=True, verbose=1)
        callback_list.append(early_stopping)

    # Model checkpointing
    if model_save_path and model_save_path != "":
        checkpoint_dir = os.path.dirname(model_save_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = model_save_path
        if not checkpoint_path.endswith((".h5", ".keras")):
            checkpoint_path = f"{checkpoint_path}.keras"

        model_checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1)
        callback_list.append(model_checkpoint)

        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.csv")
        callback_list.append(callbacks.CSVLogger(metrics_log_path, append=False))

    # Train model
    print(f"\nUsing split manifest: {manifest_path}")
    print(f"Training with {len(train_paths)} training samples and {len(val_paths)} validation samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, Seed: {seed}, Preprocessing: {preprocess_mode}, Arch: {arch}\n")
    print(f"Sampling strategy: {sampling_strategy}\n")
    print(f"Pipeline settings: parallel_calls={pipeline_parallel_calls}, " f"shuffle_buffer={min(shuffle_buffer, len(train_paths))}, prefetch={prefetch_batches}\n")
    if model_save_path and model_save_path != "":
        print(f"Metrics CSV: {os.path.join(os.path.dirname(model_save_path), 'training_metrics.csv')}\n")

    normalized_arch = arch.strip().lower()
    should_finetune = _supports_staged_finetune(normalized_arch) and finetune_unfreeze and epochs > finetune_freeze_epochs
    warmup_epochs = min(epochs, finetune_freeze_epochs) if should_finetune else epochs

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=warmup_epochs, callbacks=callback_list, verbose=2)

    if should_finetune:
        backbone = getattr(model, "_backbone", None)
        if backbone is None:
            raise ValueError("EfficientNet fine-tuning requested but backbone reference is missing.")

        backbone.trainable = True
        finetune_optimizer = _build_optimizer(
            learning_rate=finetune_lr,
            weight_decay=finetune_weight_decay,
        )
        _compile_model(model, finetune_optimizer)

        fine_tune_history = model.fit(train_dataset, validation_data=val_dataset, initial_epoch=warmup_epochs, epochs=epochs, callbacks=callback_list, verbose=2)
        history = _merge_histories(history, fine_tune_history)

    # Reset mixed precision policy if it was enabled
    if mixed_precision_enabled:
        tf.keras.mixed_precision.set_global_policy("float32")

    return model, history
