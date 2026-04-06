import tensorflow as tf
from typing import Optional
from skimage.feature import graycomatrix
from features_tools import image_read, fft_spectrum

# ----------------------------SPATIAL FEATURES----------------------------

def get_covariance_matrix(path: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Returns a [2,2] tf.Tensor of sobel edge covariants for given image.

    Args:
        path (_str_): _Path to image file_

    Returns:
        tuple[tf.Tensor, tf.Tensor]: _A [2,2] covariance matrix of the given image_
    """
    
    img = image_read(path)
    
    luminance = tf.cast(0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2], tf.float32) # [x, y]
    
    sobel = tf.image.sobel_edges(tf.expand_dims(luminance, axis=0))
    
    Gx = sobel[..., 0]
    Gy = sobel[..., 1]
    
    Gx_flat = tf.reshape(Gx, [-1])
    Gy_flat = tf.reshape(Gy, [-1])    
    
    M = tf.stack([Gx_flat, Gy_flat], axis=1)  # shape [N, 2]
    
    N = tf.cast(tf.shape(M)[0], tf.float32)
    C = tf.matmul(M, M, transpose_a=True) / N # shape [2, 2] Covariance matrix
    
    return C

# See paper "Detecting GAN generated Fake Images using Co-occurrence Matrices"[https://library.imaging.org/ei/articles/31/5/art00008]
def gray_comatrix(path: str, num_levels: int = 8, image_size: Optional[tuple[int, int]] = None) -> tf.Tensor:
    """Returns a gray level co-occurrence matrix for the given image using scikit-image.

    Args:
        path (_str_): _Path to image file_
        num_levels (int, optional): _Number of gray levels to quantize the image into. Defaults to 8._
        image_size (Optional[tuple[int, int]], optional): _If provided, the image will be resized to this size before computing the GLCM. Defaults to None._
    Returns:
        tf.Tensor: _A [num_levels, num_levels] tensor representing the GLCM of the image._
        """
        
    img = image_read(path)
    
    if image_size is not None:
        img = tf.image.resize(img, image_size)

    gray = tf.image.rgb_to_grayscale(img)
    gray = tf.cast(gray, tf.int32)
    gray = gray // tf.constant(256 // num_levels, dtype=tf.int32) # Normalize to [0, num_levels - 1]
    
    glcm_np = graycomatrix(gray.numpy().squeeze(), distances=[1], angles=[0], levels=num_levels, symmetric=True, normed=False)
    
    return tf.convert_to_tensor(glcm_np[:, :, 0, 0], dtype=tf.int32)

def noise_residual(path: str) -> tf.Tensor:
    image = image_read(path)
    
    # High pass filter kernel for noise residual extraction subject to change
    kernel = tf.constant([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=tf.float32)

    kernel = kernel[:, :, None, None]
    image = tf.expand_dims(image, 0)
    residual = tf.nn.conv2d(image, kernel, strides=1, padding="SAME")

    return residual

# ----------------------------FREQUENCY FEATURES----------------------------

def frequency_log_spectrum(path: str, size: tuple[int, int] = (224, 224)) -> tf.Tensor:
    img = image_read(path)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, size)
    fft = tf.signal.fft2d(tf.cast(img, tf.complex64))
    magnitude = tf.abs(fft)
    log_spectrum = tf.math.log(magnitude + 1e-8)
    log_spectrum = tf.signal.fftshift(log_spectrum) # Shift the zero-frequency component to the center of the spectrum

    return log_spectrum

def frequency_mean(path: str) -> tf.Tensor:
    spec = fft_spectrum(path)
    mean = tf.reduce_mean(spec)
    return mean

def frequency_variance(path: str) -> tf.Tensor:
    spec = fft_spectrum(path)
    variance = tf.math.reduce_variance(spec)
    return variance

def frequency_skewness(path: str) -> tf.Tensor:
    spec = fft_spectrum(path)
    mean = frequency_mean(path)
    variance = frequency_variance(path)
    skewness = tf.reduce_mean(((spec - mean) ** 3) / (variance ** 1.5 + 1e-8))
    return skewness

def frequency_high(path: str) -> tf.Tensor:
    spec = fft_spectrum(path)
    high_freq = tf.reduce_mean(spec[spec > tf.reduce_mean(spec)])
    return high_freq

# TODO fix math
def radial_spectrum(path: str) -> tf.Tensor:
    # Computes the radial spectrum by averaging the FFT spectrum values in concentric circles around the center of the spectrum.
    spec = fft_spectrum(path)
    h, w, d = spec.shape
    cy, cx = h//2, w//2

    y = tf.range(h)
    x = tf.range(w)

    Y, X = tf.meshgrid(y, x, indexing="ij")

    r = tf.sqrt((X-cx)**2 + (Y-cy)**2)
    r = tf.cast(r, tf.int32)
    max_r = tf.reduce_max(r)

    spectrum = []

    for i in range(max_r):
        mask = tf.where(r == i)
        values = tf.gather_nd(spec, mask)
        spectrum.append(tf.reduce_mean(values))

    return tf.stack(spectrum)