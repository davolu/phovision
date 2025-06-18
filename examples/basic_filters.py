import numpy as np
from phovision import (
    gaussian_blur,
    median_filter,
    mean_filter,
    bilateral_filter,
    read_image,
    save_image
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
    
    # Save results using phovision's save_image function
    save_image(image, 'original.png')
    save_image(gaussian_result, 'gaussian_blur.png')
    save_image(median_result, 'median_filter.png')
    save_image(mean_result, 'mean_filter.png')
    save_image(bilateral_result, 'bilateral_filter.png')
    
    print("Images saved successfully!")
    
    # Example of reading an existing image
    try:
        # You can read images from:
        # 1. Local file
        img1 = read_image('original.png')
        
        # 2. URL (uncomment to test)
        # img2 = read_image('https://example.com/image.jpg')
        
        # 3. Base64 string (if you have one)
        # img3 = read_image('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...')
        
        print("Image reading examples completed successfully!")
    except Exception as e:
        print(f"Error reading image: {e}")

if __name__ == "__main__":
    main() 