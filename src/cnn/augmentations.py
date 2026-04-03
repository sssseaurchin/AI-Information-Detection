import tensorflow as tf


def _random_resize_crop(img: tf.Tensor) -> tf.Tensor:
    original_shape = tf.shape(img)
    scale = tf.random.uniform([], minval=0.9, maxval=1.1, dtype=tf.float32)
    resized_height = tf.maximum(original_shape[0], tf.cast(tf.cast(original_shape[0], tf.float32) * scale, tf.int32))
    resized_width = tf.maximum(original_shape[1], tf.cast(tf.cast(original_shape[1], tf.float32) * scale, tf.int32))
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
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02, dtype=tf.float32)
    return img + noise


def _random_jpeg_recompression(img: tf.Tensor) -> tf.Tensor:
    # Prefer actual JPEG encode/decode for realistic compression artifacts.
    uint8_img = tf.cast(tf.clip_by_value(img, 0.0, 1.0) * 255.0, tf.uint8)
    decoded = tf.image.random_jpeg_quality(uint8_img, min_jpeg_quality=30, max_jpeg_quality=90)
    return tf.cast(decoded, tf.float32) / 255.0


def _maybe_apply(probability: float, fn, img: tf.Tensor) -> tf.Tensor:
    if probability <= 0.0:
        return img
    return tf.cond(
        tf.random.uniform([]) < probability,
        lambda: fn(img),
        lambda: img,
    )


def _augment_general_photo(img: tf.Tensor) -> tf.Tensor:
    img = _random_resize_crop(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_hue(img, max_delta=0.02)
    img = _maybe_apply(0.25, _box_blur, img)
    img = _maybe_apply(0.25, _gaussian_noise, img)
    img = _maybe_apply(0.25, _random_jpeg_recompression, img)
    return img


def _augment_document(img: tf.Tensor) -> tf.Tensor:
    img = _random_resize_crop(img)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.95, upper=1.1)
    img = _maybe_apply(0.35, _random_jpeg_recompression, img)
    img = _maybe_apply(0.1, _gaussian_noise, img)
    return img


def _augment_satellite(img: tf.Tensor) -> tf.Tensor:
    img = _random_resize_crop(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    img = tf.image.random_brightness(img, max_delta=0.06)
    img = tf.image.random_contrast(img, lower=0.95, upper=1.08)
    img = _maybe_apply(0.15, _box_blur, img)
    img = _maybe_apply(0.1, _gaussian_noise, img)
    return img


def _augment_face_like(img: tf.Tensor) -> tf.Tensor:
    img = _random_resize_crop(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.92, upper=1.08)
    img = tf.image.random_saturation(img, lower=0.92, upper=1.08)
    img = tf.image.random_hue(img, max_delta=0.015)
    img = _maybe_apply(0.15, _box_blur, img)
    img = _maybe_apply(0.15, _gaussian_noise, img)
    img = _maybe_apply(0.2, _random_jpeg_recompression, img)
    return img


def apply_training_augmentations(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply stronger real-world augmentations only on the training pipeline."""
    img = _augment_general_photo(img)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def apply_domain_aware_training_augmentations(
    img: tf.Tensor,
    label: tf.Tensor,
    domain: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply domain-specific augmentations while preserving artifact cues as much as possible."""
    normalized_domain = tf.strings.lower(tf.strings.strip(domain))

    def _doc():
        return _augment_document(img)

    def _satellite():
        return _augment_satellite(img)

    def _face():
        return _augment_face_like(img)

    def _default():
        return _augment_general_photo(img)

    augmented = tf.case(
        [
            (tf.equal(normalized_domain, "doc"), _doc),
            (tf.equal(normalized_domain, "satellite"), _satellite),
            (tf.logical_or(tf.equal(normalized_domain, "ff++"), tf.equal(normalized_domain, "chameleon")), _face),
        ],
        default=_default,
        exclusive=False,
    )
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)
    return augmented, label, domain
