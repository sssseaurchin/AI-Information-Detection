import os
from typing import Callable

import tensorflow as tf


SUPPORTED_PREPROCESS_MODES = {"rgb", "sobel", "rgb+sobel", "wavelet", "rgb+wavelet"}


def get_default_preprocess_mode() -> str:
    """Read the shared preprocessing mode used by training and inference."""
    return os.environ.get("PREPROCESS_MODE", "rgb").strip().lower()


def _decode_rgb_image(path: str, image_size: tuple[int, int]) -> tf.Tensor:
    """Read, decode, resize, and normalize an image into RGB float space."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(img, image_size, method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32) / tf.constant(255.0)
    return img


def _sobel_from_rgb(img: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    """Convert an RGB tensor into a normalized 3-channel Sobel magnitude image."""
    gray = tf.image.rgb_to_grayscale(img)
    gray = tf.expand_dims(gray, 0)
    sobel = tf.image.sobel_edges(gray)
    dy = sobel[..., 0]
    dx = sobel[..., 1]
    edges = tf.sqrt(tf.square(dx) + tf.square(dy))
    edges = tf.squeeze(edges, axis=[0, -1])
    edges = edges / (tf.reduce_max(edges) + 1e-7)
    edges = tf.image.resize(tf.expand_dims(edges, -1), image_size, method="bilinear", antialias=False)
    edges = tf.squeeze(edges, axis=-1)
    return tf.stack([edges, edges, edges], axis=-1)


def _normalize_feature_map(features: tf.Tensor) -> tf.Tensor:
    """Scale feature maps into [0, 1] while remaining stable on flat inputs."""
    features = tf.cast(features, tf.float32)
    return features / (tf.reduce_max(features) + 1e-7)


def _haar_wavelet_from_rgb(img: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    """Compute a simple single-level Haar detail map and return it as 3 channels."""
    gray = tf.image.rgb_to_grayscale(img)

    top_left = gray[0::2, 0::2, :]
    top_right = gray[0::2, 1::2, :]
    bottom_left = gray[1::2, 0::2, :]
    bottom_right = gray[1::2, 1::2, :]

    horizontal = tf.abs((top_left - top_right + bottom_left - bottom_right) / 4.0)
    vertical = tf.abs((top_left + top_right - bottom_left - bottom_right) / 4.0)
    diagonal = tf.abs((top_left - top_right - bottom_left + bottom_right) / 4.0)

    detail = tf.concat([horizontal, vertical, diagonal], axis=-1)
    detail = _normalize_feature_map(detail)
    detail = tf.image.resize(detail, image_size, method="bilinear", antialias=False)
    return detail


def preprocess_image(path: str, label: int, image_size: tuple[int, int], mode: str = "rgb") -> tuple[tf.Tensor, int]:
    """Apply the repository's shared preprocessing pipeline for a chosen mode."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in SUPPORTED_PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    rgb = _decode_rgb_image(path, image_size)
    if normalized_mode == "rgb":
        return rgb, label

    if normalized_mode in {"sobel", "rgb+sobel"}:
        sobel = _sobel_from_rgb(rgb, image_size)
    if normalized_mode == "sobel":
        return sobel, label
    if normalized_mode == "rgb+sobel":
        combined = tf.clip_by_value((rgb + sobel) / 2.0, 0.0, 1.0)
        return combined, label

    wavelet = _haar_wavelet_from_rgb(rgb, image_size)
    if normalized_mode == "wavelet":
        return wavelet, label

    combined = tf.clip_by_value((rgb + wavelet) / 2.0, 0.0, 1.0)
    return combined, label


def get_preprocess_fn(mode: str | None = None) -> Callable[[str, int, tuple[int, int]], tuple[tf.Tensor, int]]:
    """Build a preprocessing callable with a frozen mode for tf.data mapping."""
    selected_mode = (mode or get_default_preprocess_mode()).strip().lower()

    def _preprocess(path: str, label: int, image_size: tuple[int, int]) -> tuple[tf.Tensor, int]:
        return preprocess_image(path, label, image_size, mode=selected_mode)

    return _preprocess
