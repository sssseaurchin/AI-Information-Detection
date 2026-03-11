import tensorflow as tf

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

# TODO Use scikit
# WARNING: IS VERY SLOW BEWARE
# See paper "Detecting GAN generated Fake Images using Co-occurrence Matrices"[https://library.imaging.org/ei/articles/31/5/art00008]
def gray_level_co_occurance_matrix(path: str, num_levels: int = 8) -> tf.Tensor:
    """Returns a gray level co-occurrence matrix for the given image.

    Args:
        path (_str_): _Path to image file_
        num_levels (int): _Number of gray levels_
    Returns:
        tf.Tensor: _A [num_levels, num_levels] gray level co-occurrence matrix_
    """
    
    img = _image_read(path)
    
    gray = tf.image.rgb_to_grayscale(img)
    gray = tf.cast(gray, tf.int32)
    gray = gray // tf.constant(256 // num_levels, dtype=tf.int32) # Normalize to [0, num_levels - 1]
    
    glcm = tf.zeros((num_levels, num_levels), dtype=tf.int32)
    
    for i in range(tf.shape(gray)[0] - 2):
        for j in range(tf.shape(gray)[1] - 1):
            
            left = int(gray[i, j, 0])  
            right = int(gray[i + 1, j, 0])
            
            print(f"Pixel ({i}, {j}): Left={left}, Right={right}")
            
            indices = tf.constant([[left, right]], dtype=tf.int32)
            glcm = tf.tensor_scatter_nd_add(glcm, indices, [1])
            
    return glcm

