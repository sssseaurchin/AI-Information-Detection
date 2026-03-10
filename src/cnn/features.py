import tensorflow as tf

def get_covariance_matrix(path: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Returns a [2,2] tf.Tensor of sobel edge covariants for given image.

    Args:
        path (_str_): _Path to image file_

    Returns:
        tuple[tf.Tensor, tf.Tensor]: _A [2,2] covariance matrix of the given image_
    """
    
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.cast(img, tf.float32)
    
    luminance = tf.cast(0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2], tf.float32) # [x, y]
    
    sobel = tf.image.sobel_edges(tf.expand_dims(luminance, axis=0))
    
    Gx = sobel[..., 0]
    Gy = sobel[..., 1]
    
    Gx_flat = tf.reshape(Gx, [-1])
    Gy_flat = tf.reshape(Gy, [-1])    
    
    M = tf.stack([Gx_flat, Gy_flat], axis=1)  # shape [N, 2]
    
    N = tf.cast(tf.shape(M)[0], tf.float32)
    C = tf.matmul(M, M, transpose_a=True) / N # shape [2, 2] Covariance matrix
    
    return C