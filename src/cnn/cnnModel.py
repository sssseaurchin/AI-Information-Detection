import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore[import]
import numpy as np
import pandas as pd
import os 

def build_ds_from_csv(dataset_path: str, csv_name: str, batch_size: int = 32, shuffle: bool = True, 
                     image_size: tuple = (224, 224), use_cache: bool = False) -> tf.data.Dataset: 
    # Build a TensorFlow dataset from CSV file with optimized pipeline - returns tf.data.Dataset ready for training
    csv_path = os.path.join(dataset_path, csv_name)
    dataframe = pd.read_csv(csv_path)
    
    # Collect paths and labels efficiently using list comprehension
    paths = [
        os.path.join(dataset_path, row['category'], row['image_name'])
        for _, row in dataframe.iterrows()
    ]
    labels = [
        1 if row['category'] == 'fake' else 0
        for _, row in dataframe.iterrows()
    ]
    
    # Convert to numpy arrays for better memory efficiency
    paths = np.array(paths)
    labels = np.array(labels)
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def load_and_preprocess_image(path, label):
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
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    # Map: Load and preprocess images (GPU-accelerated, parallel)
    dataset = dataset.map(
        load_and_preprocess_image, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Filter out invalid images
    dataset = dataset.filter(lambda img, label: tf.shape(img)[0] == image_size[0])
    
    # Cache in memory if requested
    if use_cache:
        dataset = dataset.cache()
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1024, len(paths)))
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

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
                model_save_path: str = None) -> tuple[tf.keras.Model, callbacks.History]:
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
    
    def load_and_preprocess_image(path, label):
        # Load and preprocess image using pure TensorFlow ops (GPU-accelerated) with error handling - reads file, decodes, resizes and normalizes
        # Read image file
        img_bytes = tf.io.read_file(path)
        
        # Decode image (handles JPEG, PNG, BMP, GIF)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        
        # Ensure image has 3 dimensions
        img = tf.ensure_shape(img, [None, None, 3])
        
        # Resize to target size
        img = tf.image.resize(img, (224, 224), method='bilinear', antialias=True)
        
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        # Ensure shape is correct
        img.set_shape((224, 224, 3))
        
        return img, label
    
    # Create train dataset with optimized pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    
    # Map: Load and preprocess images (GPU-accelerated, parallel)
    # Ignore errors for corrupt images (skip them instead of crashing)
    train_dataset = train_dataset.map(
        load_and_preprocess_image, 
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
    
    # Cache: Store preprocessed images (memory or disk based on parameter)
    if use_cache:
        if cache_in_memory:
            train_dataset = train_dataset.cache()  # Memory cache
        else:
            # Use disk cache only if dataset is reasonably sized (< 50K images)
            if num_samples < 50000:
                train_dataset = train_dataset.cache(os.path.join(dataset_path, 'train_cache'))
            else:
                print("Warning: Dataset too large for disk cache, skipping cache for better performance")
    
    # Shuffle: Randomize order (after cache for efficiency)
    train_dataset = train_dataset.shuffle(buffer_size=min(1024, len(train_paths)))
    
    # Batch: Group into batches
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    
    # Prefetch: Prepare next batch while GPU is training
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create val dataset with optimized pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    
    # Map: Load and preprocess images (GPU-accelerated, parallel)
    val_dataset = val_dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Filter out any invalid images
    val_dataset = val_dataset.filter(lambda img, label: tf.shape(img)[0] == 224)
    
    # Ignore errors for corrupt images
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    
    # Cache: Store preprocessed images
    if use_cache:
        if cache_in_memory:
            val_dataset = val_dataset.cache()  # Memory cache
        else:
            if num_samples < 50000:
                val_dataset = val_dataset.cache(os.path.join(dataset_path, 'val_cache'))
    
    # Batch: Group into batches (no shuffle for validation)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    
    # Prefetch: Prepare next batch while GPU is validating
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Build model
    model = build_cnn_model()
    
    # Configure optimizer with learning rate
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
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callback_list = []
    
    # Early stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stopping)
    
    # Model checkpointing
    if model_save_path:
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
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    callback_list.append(reduce_lr)
    
    # Train model
    print(f"\nTraining with {len(train_paths)} training samples and {len(val_paths)} validation samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callback_list,
        verbose=1
    )
    
    # Reset mixed precision policy if it was enabled
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('float32')
    
    return model, history

def predict_image(model: tf.keras.Model, image_path: str, image_size: tuple = (224, 224)) -> float:
    # Predict if an image is AI-generated or real using TensorFlow ops (GPU-accelerated) - returns confidence score 0.0 to 1.0
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load and preprocess image using TensorFlow ops
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    
    # Ensure image has correct shape
    img = tf.ensure_shape(img, [None, None, 3])
    
    # Resize to target size
    img = tf.image.resize(img, image_size, method='bilinear', antialias=True)
    
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    # Add batch dimension
    img = tf.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    confidence = float(predictions[0][1])  # Confidence for AI-generated class
    
    return confidence