import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore[import]
import numpy as np
import pandas as pd

def build_ds_from_csv(file_path: str, batch_size: int = 32, shuffle: bool = True):
    df = pd.read_csv(file_path)
    images = np.array([np.fromstring(img_str, sep=' ').reshape(28, 28) for img_str in df['image']])
    print(images.shape)
    labels = np.array(df['label'])
   
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    
    return dataset

# Server call template
# TODO
def returnConfidence(confidence: float) -> float:
    return confidence
    