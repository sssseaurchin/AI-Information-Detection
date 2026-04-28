import tensorflow as tf
import keras
from keras import layers, models, callbacks
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess
from keras.applications.resnet import preprocess_input as resnet_preprocess
import numpy as np
import pandas as pd
import os
from typing import Callable

# --- Helper logic for optional imports ---
try:
    from keras.applications.convnext import preprocess_input as convnext_preprocess
except ImportError:
    convnext_preprocess = None

try:
    from transformers import TFCLIPVisionModel
except ImportError:
    TFCLIPVisionModel = None

TRANSFER_ARCHS = {"efficientnet_b0", "efficientnet_v2b0", "resnet50", "convnext_tiny", "clip_vit_b32"}

# --- Custom Layers ---
@keras.utils.register_keras_serializable(package="aid")
class SobelMagnitudeLayer(layers.Layer):
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

@keras.utils.register_keras_serializable(package="aid")
class HaarWaveletLayer(layers.Layer):
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

# --- Model Builders (Sanitized Type Hints) ---

def build_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

def build_efficientnet_b0_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    inputs = layers.Input(shape=input_shape)
    backbone = keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.2)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = models.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_classifier")
    model._backbone = backbone
    return model

def build_efficientnet_v2b0_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    inputs = layers.Input(shape=input_shape)
    backbone = keras.applications.EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.2)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = models.Model(inputs=inputs, outputs=outputs, name="efficientnet_v2b0_classifier")
    model._backbone = backbone
    return model

def build_clip_vit_b32_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    if TFCLIPVisionModel is None:
        raise ValueError("transformers is required for clip_vit_b32 but is not installed.")
    inputs = layers.Input(shape=input_shape)
    backbone = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    backbone.trainable = False
    pixel_values = layers.Lambda(lambda tensor: tf.transpose(tensor, perm=[0, 3, 1, 2]))(inputs)
    vision_outputs = backbone(pixel_values=pixel_values, training=False)
    pooled = vision_outputs.pooler_output
    pooled = layers.Dropout(0.2)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = models.Model(inputs=inputs, outputs=outputs, name="clip_vit_b32_classifier")
    model._backbone = backbone
    return model

def build_dual_artifact_cnn_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    inputs = layers.Input(shape=input_shape)
    rgb_features = _small_conv_branch(inputs, filters=(32, 64, 96), prefix="rgb")
    sobel_features = _small_conv_branch(SobelMagnitudeLayer()(inputs), filters=(16, 32), prefix="sobel")
    wavelet_features = _small_conv_branch(HaarWaveletLayer()(inputs), filters=(16, 32), prefix="wavelet")
    fused = layers.Concatenate()([rgb_features, sobel_features, wavelet_features])
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.4)(fused)
    outputs = layers.Dense(num_classes, activation="softmax")(fused)
    return models.Model(inputs=inputs, outputs=outputs, name="dual_artifact_cnn")

def build_resnet50_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    inputs = layers.Input(shape=input_shape)
    backbone = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.3)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_classifier")
    model._backbone = backbone
    return model

def build_convnext_tiny_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    if not hasattr(keras.applications, "ConvNeXtTiny"):
        raise ValueError("ConvNeXtTiny is unavailable in this Keras version.")
    inputs = layers.Input(shape=input_shape)
    backbone = keras.applications.ConvNeXtTiny(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(0.3)(pooled)
    outputs = layers.Dense(num_classes, activation="softmax")(pooled)
    model = models.Model(inputs=inputs, outputs=outputs, name="convnext_tiny_classifier")
    model._backbone = backbone
    return model

def build_model(arch: str = "simple", input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> models.Model:
    normalized_arch = arch.strip().lower()
    mapping = {
        "simple": build_cnn_model,
        "dual_artifact_cnn": build_dual_artifact_cnn_model,
        "efficientnet_b0": build_efficientnet_b0_model,
        "efficientnet_v2b0": build_efficientnet_v2b0_model,
        "clip_vit_b32": build_clip_vit_b32_model,
        "resnet50": build_resnet50_model,
        "convnext_tiny": build_convnext_tiny_model
    }
    if normalized_arch in mapping:
        return mapping[normalized_arch](input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unsupported architecture: {arch}")

def _small_conv_branch(inputs: tf.Tensor, filters: tuple[int, int, int] | tuple[int, int], prefix: str) -> tf.Tensor:
    x = inputs
    for index, filter_count in enumerate(filters, start=1):
        x = layers.Conv2D(filter_count, (3, 3), padding="same", name=f"{prefix}_conv_{index}")(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn_{index}")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu_{index}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{prefix}_pool_{index}")(x)
        x = layers.Dropout(0.2, name=f"{prefix}_dropout_{index}")(x)
    return layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)

def _build_optimizer(learning_rate, weight_decay: float):
    try:
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    except AttributeError:
        return keras.optimizers.Adam(learning_rate=learning_rate)

def _compile_model(model: models.Model, optimizer) -> None:
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# (Rest of your helper functions: train_model, _resample_train_frame, etc. should also replace tf.keras with keras where applicable)