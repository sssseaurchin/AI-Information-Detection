import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore[import]
import numpy as np
import pandas as pd
import os 
from typing import Callable
from preprocessing import get_preprocess_fn
from split_utils import load_or_create_split_manifest

def preprocess_regular(path, label, image_size):
    """Compatibility wrapper around the shared RGB preprocessing path."""
    return get_preprocess_fn("rgb")(path, label, image_size)

# TODO covariance matrix
def preprocess_sobel_edge(path, label, image_size):
    """Compatibility wrapper around the shared Sobel preprocessing path."""
    return get_preprocess_fn("sobel")(path, label, image_size)

def build_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    # Build optimized CNN model with BatchNormalization and improved architecture - returns compiled Keras model
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(dataset_path: str, epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2, 
                use_cache: bool = True, cache_in_memory: bool = False, 
                use_mixed_precision: bool = True, enable_augmentation: bool = False,
                model_save_path: str = "", preprocess_func: Callable | None = None,
                image_size: tuple = (224, 224), preprocess_mode: str = "rgb",
                label_mapping: dict[str, int] | None = None, seed: int = 42,
                split_manifest_path: str = "", regen_split: bool = False,
                allow_unknown: bool = False) -> tuple[tf.keras.Model, callbacks.History]:
    # Train the CNN model for AI-generated image detection - returns trained model and training history
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Enable mixed precision for faster training on GPU
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled for faster GPU performance")
    
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    if label_mapping is None:
        raise ValueError("train_model requires an explicit label_mapping.")

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
    train_frame = usable_manifest[usable_manifest["split"] == "train"]
    val_frame = usable_manifest[usable_manifest["split"] == "val"]

    train_paths = train_frame["path"].to_numpy()
    train_labels = train_frame["label"].astype(int).to_numpy()
    val_paths = val_frame["path"].to_numpy()
    val_labels = val_frame["label"].astype(int).to_numpy()
    num_samples = len(usable_manifest)
    
    def augment_image(img, label):
        # Apply data augmentation to image - random flip, brightness and contrast adjustments
        # Random horizontal flip
        img = tf.image.random_flip_left_right(img)
        
        # Random brightness adjustment
        img = tf.image.random_brightness(img, max_delta=0.1)
        
        # Random contrast adjustment
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        
        # Ensure values stay in [0, 1] range
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img, label
    
    # Create train dataset with optimized pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    
    # Map: Load and preprocess images (GPU-accelerated, parallel)
    # Ignore errors for corrupt images (skip them instead of crashing)
    train_dataset = train_dataset.map(
        lambda path, label: preprocess_callable(path, label, image_size), 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Filter out any invalid images (shape mismatches, etc.)
    # Note: Corrupt JPEG files will be caught by decode_jpeg and cause an error
    # We need to handle this at the dataset level
    train_dataset = train_dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])
    
    # Ignore errors for corrupt images - skip them instead of crashing
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    
    # Apply data augmentation if enabled (only for training)
    if enable_augmentation:
        train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch: Group into batches
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    
    # Cache: Store preprocessed images (memory or disk based on parameter)
    if use_cache:
        if cache_in_memory:
            train_dataset = train_dataset.cache()  # Memory cache
        else:
            # Use disk cache only if dataset is reasonably sized (< 100K images)
            if num_samples < 100000:
                train_dataset = train_dataset.cache(os.path.join(dataset_path, 'train_cache'))
            else:
                print("Warning: Dataset too large for disk cache, skipping cache for better performance")
                
    # Shuffle: Randomize order (after cache for efficiency)
    train_dataset = train_dataset.shuffle(
        buffer_size=min(1024, len(train_paths)),
        seed=seed,
        reshuffle_each_iteration=True,
    ) #TODO Change buffer size?
    
    # Prefetch: Prepare next batch while GPU is training
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create val dataset with optimized pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    
    # Map: Load and preprocess images (GPU-accelerated, parallel)
    val_dataset = val_dataset.map(
        lambda path, label: preprocess_callable(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Filter out any invalid images
    val_dataset = val_dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])
    
    # Ignore errors for corrupt images
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    
    # Batch: Group into batches (no shuffle for validation)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    
    # Cache: Store preprocessed images
    if use_cache:
        if cache_in_memory:
            val_dataset = val_dataset.cache()  # Memory cache
        else:
            if num_samples < 100000:
                val_dataset = val_dataset.cache(os.path.join(dataset_path, 'val_cache'))
            else:
                print("Warning: Dataset too large for disk cache, skipping cache for better performance")
    
    # Prefetch: Prepare next batch while GPU is validating
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Build model
    model = build_cnn_model()
    
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    # Mixed precision works fine with string loss, but using class is more explicit
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), # I have no idea how to change this without breaking everythin
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callback_list = []
    
    # Early stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=1,
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stopping)
    
    # Model checkpointing
    if model_save_path and model_save_path != "":
        checkpoint_dir = os.path.dirname(model_save_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=model_save_path if model_save_path.endswith('.h5') else f"{model_save_path}.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callback_list.append(model_checkpoint)
    
    # Train model
    print(f"\nUsing split manifest: {manifest_path}")
    print(f"Training with {len(train_paths)} training samples and {len(val_paths)} validation samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, Seed: {seed}, Preprocessing: {preprocess_mode}\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callback_list,
        verbose="1"
    )
    
    # Reset mixed precision policy if it was enabled
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('float32')
    
    return model, history

def predict_image(model: tf.keras.Model, image_path: str, image_size: tuple = (224, 224),
                  preprocessing_func: Callable | None = None, preprocess_mode: str = "rgb") -> float:
    # Predict if an image is AI-generated or real using TensorFlow ops (GPU-accelerated) - returns confidence score 0.0 to 1.0
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    preprocess_callable = preprocessing_func or get_preprocess_fn(preprocess_mode)
    img = preprocess_callable(image_path, label=0, image_size=image_size)[0]  # Get preprocessed image tensor

    # Ensure we have a batch dimension: model expects (batch, h, w, c)
    if len(img.shape) == 3:
        img = tf.expand_dims(img, 0)

    # Convert to numpy if possible (eager mode), otherwise pass the tensor
    try:
        img_input = img.numpy()
    except Exception:
        img_input = img

    # Make prediction (use integer verbose)
    predictions = model.predict(img_input, verbose=0)

    # Extract confidence for AI-generated class (assumes 2-class softmax)
    try:
        confidence = float(predictions[0][1])
    except Exception:
        # Fallback: if predictions shape unexpected, try a safe conversion
        preds = np.asarray(predictions)
        if preds.ndim == 1 and preds.size >= 2:
            confidence = float(preds[1])
        else:
            # As a last resort, return the max class probability
            confidence = float(np.max(preds))

    return confidence
