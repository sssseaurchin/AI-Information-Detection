import tensorflow as tf
from typing import Optional
from skimage.feature import graycomatrix, graycoprops

def _image_read(path: str) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.cast(img, tf.float32)
    return img

def get_covariance_matrix(path: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Returns a [2,2] tf.Tensor of sobel edge covariants for given image.

    Args:
        path (_str_): _Path to image file_

    Returns:
        tuple[tf.Tensor, tf.Tensor]: _A [2,2] covariance matrix of the given image_
    """
    
    img = _image_read(path)
    
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
        
    img = _image_read(path)
    
    if image_size is not None:
        img = tf.image.resize(img, image_size)

    gray = tf.image.rgb_to_grayscale(img)
    gray = tf.cast(gray, tf.int32)
    gray = gray // tf.constant(256 // num_levels, dtype=tf.int32) # Normalize to [0, num_levels - 1]
    
    glcm_np = graycomatrix(gray.numpy().squeeze(), distances=[1], angles=[0], levels=num_levels, symmetric=True, normed=False)
    
    return tf.convert_to_tensor(glcm_np[:, :, 0, 0], dtype=tf.int32)