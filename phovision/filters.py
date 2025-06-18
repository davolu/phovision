import numpy as np

def _validate_kernel_size(kernel_size):
    """Validate that kernel size is odd and positive."""
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd positive integer")

def _pad_image(image, pad_width):
    """Pad image with reflected borders."""
    if len(image.shape) == 2:  # Grayscale
        return np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')
    else:  # RGB/RGBA
        return np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or RGB)
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation of the Gaussian kernel
    
    Returns:
        numpy.ndarray: Blurred image
    """
    _validate_kernel_size(kernel_size)
    
    # Create Gaussian kernel
    k = (kernel_size - 1) // 2
    x, y = np.meshgrid(np.linspace(-k, k, kernel_size), np.linspace(-k, k, kernel_size))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    
    # Pad image
    padded = _pad_image(image, k)
    
    # Apply convolution
    if len(image.shape) == 2:  # Grayscale
        result = np.zeros_like(image, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
    else:  # RGB/RGBA
        result = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    result[i, j, c] = np.sum(padded[i:i+kernel_size, j:j+kernel_size, c] * kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def median_filter(image, kernel_size=3):
    """
    Apply median filter to an image for noise removal.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or RGB)
        kernel_size (int): Size of the median filter kernel (must be odd)
    
    Returns:
        numpy.ndarray: Filtered image
    """
    _validate_kernel_size(kernel_size)
    
    k = (kernel_size - 1) // 2
    padded = _pad_image(image, k)
    
    if len(image.shape) == 2:  # Grayscale
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.median(window)
    else:  # RGB/RGBA
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    window = padded[i:i+kernel_size, j:j+kernel_size, c]
                    result[i, j, c] = np.median(window)
    
    return result.astype(np.uint8)

def mean_filter(image, kernel_size=3):
    """
    Apply mean (average) filter to an image.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or RGB)
        kernel_size (int): Size of the averaging kernel (must be odd)
    
    Returns:
        numpy.ndarray: Filtered image
    """
    _validate_kernel_size(kernel_size)
    
    k = (kernel_size - 1) // 2
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    padded = _pad_image(image, k)
    
    if len(image.shape) == 2:  # Grayscale
        result = np.zeros_like(image, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
    else:  # RGB/RGBA
        result = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    result[i, j, c] = np.sum(padded[i:i+kernel_size, j:j+kernel_size, c] * kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def bilateral_filter(image, kernel_size=5, sigma_spatial=1.0, sigma_intensity=50.0):
    """
    Apply bilateral filter to an image for edge-preserving smoothing.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or RGB)
        kernel_size (int): Size of the filter kernel (must be odd)
        sigma_spatial (float): Standard deviation for spatial kernel
        sigma_intensity (float): Standard deviation for intensity kernel
    
    Returns:
        numpy.ndarray: Filtered image
    """
    _validate_kernel_size(kernel_size)
    
    k = (kernel_size - 1) // 2
    padded = _pad_image(image, k)
    
    # Create spatial kernel
    x, y = np.meshgrid(np.linspace(-k, k, kernel_size), np.linspace(-k, k, kernel_size))
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))
    
    if len(image.shape) == 2:  # Grayscale
        result = np.zeros_like(image, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                intensity_diff = window - padded[i+k, j+k]
                intensity_kernel = np.exp(-(intensity_diff**2) / (2 * sigma_intensity**2))
                kernel = spatial_kernel * intensity_kernel
                kernel = kernel / np.sum(kernel)
                result[i, j] = np.sum(window * kernel)
    else:  # RGB/RGBA
        result = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    window = padded[i:i+kernel_size, j:j+kernel_size, c]
                    intensity_diff = window - padded[i+k, j+k, c]
                    intensity_kernel = np.exp(-(intensity_diff**2) / (2 * sigma_intensity**2))
                    kernel = spatial_kernel * intensity_kernel
                    kernel = kernel / np.sum(kernel)
                    result[i, j, c] = np.sum(window * kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8) 