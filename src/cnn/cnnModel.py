import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # type: ignore[import]
import numpy as np
import pandas as pd
import cv2
import os

def build_ds_from_csv(dataset_path:str ,csv_name: str, batch_size: int = 32, shuffle: bool = True, image_size: tuple = (224, 224))  -> tf.data.Dataset: 
    csv_path = os.path.join(dataset_path, csv_name)
    dataframe = pd.read_csv(csv_path)
    dataframe_length = len(dataframe)    
    
    # Load images from paths
    images = np.empty((dataframe_length, *image_size, 3), dtype=np.float32)
    labels = np.empty(dataframe_length, dtype=np.int32)
    i = 0
    valid_count = 0
    dataframe_length = len(dataframe)
    
    for _, row in dataframe.iterrows():
        i += 1
        print(f"Loading image {i}/{dataframe_length}", end='\r')
        
        img_name = row['image_name']
        category = row['category']
        img_path = os.path.join(dataset_path, category, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip if image can't be loaded
        valid_count += 1
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, image_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        images[valid_count] = img
        labels[valid_count] = 1 if category == 'fake' else 0  # 1 for AI-generated, 0 for real
        valid_count += 1
          
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
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
    
    # Load dataset
    dataset = build_ds_from_csv(dataset_path, csv_name="dataset.csv", batch_size=batch_size)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Build model
    model = build_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset, # TODO Hook the separate validation data
        epochs=epochs,
        use_multiprocessing=True,
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