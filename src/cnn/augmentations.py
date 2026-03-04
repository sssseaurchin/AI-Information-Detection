import tensorflow as tf


def _random_resize_crop(img: tf.Tensor) -> tf.Tensor:
    original_shape = tf.shape(img)
    scale = tf.random.uniform([], minval=0.85, maxval=1.15, dtype=tf.float32)
    resized_height = tf.maximum(8, tf.cast(tf.cast(original_shape[0], tf.float32) * scale, tf.int32))
    resized_width = tf.maximum(8, tf.cast(tf.cast(original_shape[1], tf.float32) * scale, tf.int32))
    resized = tf.image.resize(img, (resized_height, resized_width), method="bilinear", antialias=True)
    return tf.image.random_crop(resized, size=original_shape)


def _box_blur(img: tf.Tensor) -> tf.Tensor:
    kernel = tf.constant(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=tf.float32,
    ) / 9.0
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(tf.expand_dims(img, axis=0), kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred, axis=0)


def _gaussian_noise(img: tf.Tensor) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.03, dtype=tf.float32)
    return img + noise


def _random_jpeg_recompression(img: tf.Tensor) -> tf.Tensor:
    # Prefer actual JPEG encode/decode for realistic compression artifacts.
    quality = tf.random.uniform([], minval=30, maxval=91, dtype=tf.int32)
    uint8_img = tf.cast(tf.clip_by_value(img, 0.0, 1.0) * 255.0, tf.uint8)
    encoded = tf.io.encode_jpeg(uint8_img, quality=quality)
    decoded = tf.io.decode_jpeg(encoded, channels=3)
    return tf.cast(decoded, tf.float32) / 255.0


def apply_training_augmentations(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply stronger real-world augmentations only on the training pipeline."""
    img = _random_resize_crop(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.03)

    if tf.random.uniform([]) < 0.5:
        img = _box_blur(img)
    if tf.random.uniform([]) < 0.5:
        img = _gaussian_noise(img)
    if tf.random.uniform([]) < 0.5:
        img = _random_jpeg_recompression(img)

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label
