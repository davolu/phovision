"""
This example demonstrates the various image I/O capabilities of phovision.
It shows how to:
1. Read images from different sources
2. Save images in different formats
3. Convert images to base64
4. Work with synthetic images
"""

import numpy as np
from phovision import read_image, save_image, to_base64, gaussian_blur

def demo_local_file():
    """Demonstrate reading and saving local files."""
    try:
        # First create a synthetic image to work with
        print("Creating synthetic image...")
        width, height = 200, 200
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        synthetic = np.sin(10 * x) * np.cos(10 * y) * 127 + 128
        synthetic = synthetic.astype(np.uint8)
        
        # Save it as a PNG
        print("Saving synthetic image...")
        save_image(synthetic, 'synthetic.png')
        
        # Read it back
        print("Reading image back...")
        loaded = read_image('synthetic.png')
        
        # Process and save in different formats
        print("Processing and saving in different formats...")
        blurred = gaussian_blur(loaded, kernel_size=5, sigma=1.0)
        save_image(blurred, 'blurred.png')  # PNG
        save_image(blurred, 'blurred.jpg')  # JPEG
        save_image(blurred, 'blurred.bmp')  # BMP
        
        print("Local file operations completed successfully!")
    except Exception as e:
        print(f"Error in local file demo: {e}")

def demo_url():
    """Demonstrate reading from URL."""
    try:
        # Read an image from a URL
        # Using a placeholder image URL - replace with a real one
        url = "https://raw.githubusercontent.com/yourusername/phovision/main/examples/test_image.jpg"
        print(f"Attempting to read image from URL: {url}")
        print("Note: This will fail if the URL is not accessible.")
        
        img = read_image(url)
        save_image(img, 'from_url.png')
        print("URL image reading completed successfully!")
    except Exception as e:
        print(f"Error in URL demo: {e}")

def demo_base64():
    """Demonstrate base64 conversion."""
    try:
        # Create a simple image
        print("Creating test image for base64 conversion...")
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255  # White square on black background
        
        # Convert to base64
        print("Converting to base64...")
        base64_str = to_base64(img, format='PNG')
        print("Base64 string (first 50 chars):", base64_str[:50], "...")
        
        # Read back from base64
        print("Reading back from base64...")
        img_from_base64 = read_image(base64_str)
        save_image(img_from_base64, 'from_base64.png')
        
        print("Base64 conversion demo completed successfully!")
    except Exception as e:
        print(f"Error in base64 demo: {e}")

def main():
    print("=== Phovision I/O Examples ===\n")
    
    print("1. Local File Operations")
    print("-----------------------")
    demo_local_file()
    print()
    
    print("2. URL Operations")
    print("----------------")
    demo_url()
    print()
    
    print("3. Base64 Operations")
    print("-------------------")
    demo_base64()
    print()
    
    print("Examples completed!")

if __name__ == "__main__":
    main() 