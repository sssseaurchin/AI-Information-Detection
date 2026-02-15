import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore[import]
import numpy as np
import pandas as pd
import os 
from typing import Callable

def preprocess_regular(path, label, image_size):
    # Load and preprocess image with error handling - reads file, decodes, resizes and normalizes to [0,1]
    # Read image file
    img_bytes = tf.io.read_file(path)
     
    # Decode image (handles JPEG, PNG, BMP, GIF)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        
    # Ensure image has 3 dimensions
    img = tf.ensure_shape(img, [None, None, 3])
        
    # Resize to target size
    img = tf.image.resize(img, image_size, method='bilinear', antialias=True)
        
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / tf.constant(255.0) 
        
    return img, label

# TODO covariance matrix
def preprocess_sobel_edge(path, label, image_size):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.cast(img, tf.float32)

    # Convert to grayscale
    img = tf.image.rgb_to_grayscale(img) # [w, h, 3] -> [w, h, 1]

    # Add batch dimension
    img = tf.expand_dims(img, 0) # [w, h, 1] -> [1, w, h, 1]

    # Sobel edges
    sobel = tf.image.sobel_edges(img) 
    
    dy = sobel[..., 0]
    dx = sobel[..., 1]

    # Gradient magnitude
    edges = tf.sqrt(tf.square(dx) + tf.square(dy))  # type: ignore
    
    # Remove batch + channel dims
    edges = tf.squeeze(edges, axis=[0, -1])  # Remove batch and channel dimensions
    
    # Normalize
    edges = edges / (tf.reduce_max(edges) + 1e-7)

    # Resize
    edges = tf.image.resize(tf.expand_dims(edges, -1),image_size,method='bilinear',antialias=False)

    edges = tf.squeeze(edges, axis=-1)
    
    # Stack the single channel 3 times to match expected input shape (224, 224, 3)
    edges = tf.stack([edges, edges, edges], axis=-1)

    return edges, label

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
                model_save_path: str = "", preprocess_func: Callable = preprocess_regular, image_size: tuple = (224, 224)) -> tuple[tf.keras.Model, callbacks.History]:
    # Train the CNN model for AI-generated image detection - returns trained model and training history
    
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
    
    csv_path = os.path.join(dataset_path, "dataset.csv")
    dataframe = pd.read_csv(csv_path)
    
    # Collect paths and labels efficiently (using list comprehension for better memory)
    paths = [
        os.path.join(dataset_path, row['category'], row['image_name'])
        for _, row in dataframe.iterrows()
    ]
    labels = [
        1 if row['category'] == 'fake' else 0
        for _, row in dataframe.iterrows()
    ]
    
    # Split into train and validation
    num_samples = len(paths)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    val_size = int(num_samples * validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Use numpy arrays for more efficient memory usage
    train_paths = np.array(paths)[train_indices]
    train_labels = np.array(labels)[train_indices]
    val_paths = np.array(paths)[val_indices]
    val_labels = np.array(labels)[val_indices]
    
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
        lambda path, label: preprocess_func(path, label, image_size), 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Filter out any invalid images (shape mismatches, etc.)
    # Note: Corrupt JPEG files will be caught by decode_jpeg and cause an error
    # We need to handle this at the dataset level
    train_dataset = train_dataset.filter(lambda img, label: tf.shape(img)[0] == 224)
    
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
    train_dataset = train_dataset.shuffle(buffer_size=min(1024, len(train_paths)), reshuffle_each_iteration=True) #TODO Change buffer size?
    
    # Prefetch: Prepare next batch while GPU is training
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create val dataset with optimized pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    
    # Map: Load and preprocess images (GPU-accelerated, parallel)
    val_dataset = val_dataset.map(
        lambda path, label: preprocess_func(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Filter out any invalid images
    val_dataset = val_dataset.filter(lambda img, label: tf.shape(img)[0] == 224)
    
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
    print(f"\nTraining with {len(train_paths)} training samples and {len(val_paths)} validation samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}\n")
    
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

def predict_image(model: tf.keras.Model, image_path: str, image_size: tuple = (224, 224), preprocessing_func: Callable = preprocess_regular) -> float:
    # Predict if an image is AI-generated or real using TensorFlow ops (GPU-accelerated) - returns confidence score 0.0 to 1.0
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = preprocessing_func(image_path, label=0, image_size=image_size)[0]  # Get preprocessed image tensor

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