import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore[import]
import numpy as np
import pandas as pd
import cv2
import os

def build_ds_from_csv(dataset_path:str ,csv_name: str, batch_size: int = 32, shuffle: bool = True, image_size: tuple = (224, 224))  -> tf.data.Dataset: 
    csv_path = os.path.join(dataset_path, csv_name)
    dataframe = pd.read_csv(csv_path)
    
    # Collect paths and labels
    paths = []
    labels = []
    for _, row in dataframe.iterrows():
        img_name = row['image_name']
        category = row['category']
        img_path = os.path.join(dataset_path, category, img_name)
        paths.append(img_path)
        labels.append(1 if category == 'fake' else 0)  # 1 for AI-generated, 0 for real
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def load_and_preprocess_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# TODO: Adjust model architecture as needed
def build_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(dataset_path: str, epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2) -> tuple[tf.keras.Model, callbacks.History]:
    """Train the CNN model for AI-generated image detection."""
    
    csv_path = os.path.join(dataset_path, "dataset.csv")
    dataframe = pd.read_csv(csv_path)
    
    # Collect paths and labels
    paths = []
    labels = []
    for _, row in dataframe.iterrows():
        img_name = row['image_name']
        category = row['category']
        img_path = os.path.join(dataset_path, category, img_name)
        paths.append(img_path)
        labels.append(1 if category == 'fake' else 0)  # 1 for AI-generated, 0 for real
    
    # Split into train and validation
    num_samples = len(paths)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    val_size = int(num_samples * validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_paths = [paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    def load_and_preprocess_image(path, label):
        def _load_py(path, label):
            try:
                img_bytes = tf.io.read_file(path)
                img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
                img = tf.image.resize(img, (224, 224))
                img = tf.cast(img, tf.float32) / 255.0
                return img, label, 1  # 1 for valid
            except:
                # For corrupt images, return zeros and mark invalid
                img = tf.zeros((224, 224, 3), dtype=tf.float32)
                return img, label, 0  # 0 for invalid
        
        result = tf.py_function(_load_py, [path, label], [tf.float32, tf.int32, tf.int32])
        img, lbl, valid = result[0], result[1], result[2] # type: ignore
        img.set_shape((224, 224, 3))
        lbl.set_shape(())
        valid.set_shape(())
        return img, lbl, valid
    
    # Create train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.cache(os.path.join(dataset_path, 'train_cache'))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create val dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache(os.path.join(dataset_path, 'val_cache'))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    model = build_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=[]
    )
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose="1"
    )
    
    return model, history

def predict_image(model: tf.keras.Model, image_path: str, image_size: tuple = (224, 224)) -> float:
    """Predict if an image is AI-generated or real."""
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img)
    confidence = predictions[0][1]  # Confidence for AI-generated class
    
    return confidence