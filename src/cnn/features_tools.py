import tensorflow as tf
import tensorflow_io as tfio

def image_read(path: str) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.cast(img, tf.float32)
    
    return img

def fft_spectrum(path: str) -> tf.Tensor:
    img = image_read(path)
    img = tfio.experimental.color.rgb_to_grayscale(img)
    # img = tf.image.rgb_to_grayscale(img)
    fft = tf.signal.fft2d(tf.cast(img, tf.complex64))
    magnitude = tf.abs(fft)
    log_spectrum = tf.math.log(magnitude + 1e-8)
    log_spectrum = tf.signal.fftshift(log_spectrum) # Shift the zero-frequency component to the center of the spectrum

    return log_spectrum

def wavelet_transform(path: str) -> tf.Tensor:
    img = image_read(path)

    return tf.zeros((128, 128))  