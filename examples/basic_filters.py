import numpy as np
from phovision.filters import (
    gaussian_blur,
    median_filter,
    mean_filter,
    bilateral_filter
)

def create_noisy_image(width=200, height=200):
    """Create a sample noisy image for testing."""
    # Create a simple pattern
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    image = np.sin(10 * x) * np.cos(10 * y) * 127 + 128
    
    # Add noise
    noise = np.random.normal(0, 25, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def main():
    # Create a test image
    image = create_noisy_image()
    
    # Apply different filters
    gaussian_result = gaussian_blur(image, kernel_size=5, sigma=1.0)
    median_result = median_filter(image, kernel_size=3)
    mean_result = mean_filter(image, kernel_size=3)
    bilateral_result = bilateral_filter(image, kernel_size=5, sigma_spatial=1.0, sigma_intensity=50.0)
    
    # Save results (requires PIL/Pillow)
    try:
        from PIL import Image
        
        Image.fromarray(image).save('original.png')
        Image.fromarray(gaussian_result).save('gaussian_blur.png')
        Image.fromarray(median_result).save('median_filter.png')
        Image.fromarray(mean_result).save('mean_filter.png')
        Image.fromarray(bilateral_result).save('bilateral_filter.png')
        
        print("Images saved successfully!")
    except ImportError:
        print("PIL/Pillow is required to save images. Please install it using:")
        print("pip install Pillow")

if __name__ == "__main__":
    main() 