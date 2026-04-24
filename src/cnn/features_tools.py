import tensorflow as tf
import numpy as np

def image_read(path: str) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.cast(img, tf.float32)
    
    return img

def fft_spectrum(path: str) -> tf.Tensor:
    img = image_read(path)
    img = tf.image.rgb_to_grayscale(img)
    fft = tf.signal.fft2d(tf.cast(img, tf.complex64))
    magnitude = tf.abs(fft)
    log_spectrum = tf.math.log(magnitude + 1e-8)
    log_spectrum = tf.signal.fftshift(log_spectrum) # Shift the zero-frequency component to the center of the spectrum

    return log_spectrum

def haar_filters():
    ll = np.array([[1, 1],
                   [1, 1]], dtype=np.float32) / 2.0

    lh = np.array([[1,  1],
                   [-1, -1]], dtype=np.float32) / 2.0

    hl = np.array([[1, -1],
                   [1, -1]], dtype=np.float32) / 2.0

    hh = np.array([[1, -1],
                   [-1, 1]], dtype=np.float32) / 2.0

    filters = np.stack([ll, lh, hl, hh], axis=-1)  # (2,2,4)
    filters = np.expand_dims(filters, axis=2)      # (2,2,1,4)

    return tf.constant(filters, dtype=tf.float32)

def dwt_haar_tf(image):
    """
    image: (H, W, 1)
    returns: (H/2, W/2, 4)
    """
    filters = haar_filters()

    image = tf.expand_dims(image, axis=0)  # (1,H,W,1)

    coeffs = tf.nn.conv2d(
        image,
        filters,
        strides=[1, 2, 2, 1],
        padding='SAME'
    )

    coeffs = tf.squeeze(coeffs, axis=0)  # (H/2, W/2, 4)
    return coeffs