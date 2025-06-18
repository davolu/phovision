# Phovision

A pure Python computer vision library implementing various image processing algorithms from scratch. This library provides implementations of common image processing operations with no dependencies except NumPy.

## Features

Image Processing:
- Gaussian Blur
- Median Filter (Noise Removal)
- Mean Filter (Averaging)
- Bilateral Filter (Edge-preserving smoothing)

Image I/O (Pure Python Implementation):
- Read images from files, URLs, base64 strings, or bytes
- Save images in various formats
- Convert images to base64 strings
- Supported formats:
  - JPEG (read)
  - PNG (read)
  - BMP (read/write)
  - PPM/PGM (read/write)

## Installation

```bash
pip install phovision
```

## Usage

### Basic Image Processing

```python
from phovision import (
    read_image, save_image,
    gaussian_blur, median_filter, 
    mean_filter, bilateral_filter
)

# Read an image
image = read_image('input.jpg')  # supports jpg, png, bmp, ppm, pgm

# Apply filters
blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)
denoised = median_filter(image, kernel_size=3)
averaged = mean_filter(image, kernel_size=3)
smoothed = bilateral_filter(image, kernel_size=5, sigma_spatial=1.0, sigma_intensity=50.0)

# Save results
save_image(blurred, 'blurred.bmp')  # or .ppm, .pgm
save_image(denoised, 'denoised.bmp')
```

### Flexible Image I/O

```python
from phovision import read_image, save_image, to_base64

# Read from local file
img1 = read_image('photo.jpg')  # JPEG support
img2 = read_image('image.png')  # PNG support
img3 = read_image('data.bmp')   # BMP support

# Read from URL
img4 = read_image('https://example.com/image.jpg')

# Read from base64 string
img5 = read_image('data:image/jpeg;base64,...')

# Convert to base64 (useful for web applications)
base64_str = to_base64(img1, format='PPM')  # or 'BMP'

# Save in different formats
save_image(img1, 'output.bmp')   # Windows Bitmap
save_image(img1, 'output.ppm')   # Portable Pixmap (color)
save_image(img1, 'output.pgm')   # Portable Graymap (grayscale)
```

### Working with NumPy Arrays

```python
import numpy as np
from phovision import read_image, gaussian_blur

# Create a synthetic image
width, height = 200, 200
x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
synthetic = np.sin(10 * x) * np.cos(10 * y) * 127 + 128
synthetic = synthetic.astype(np.uint8)

# Process the synthetic image
processed = gaussian_blur(synthetic, kernel_size=5, sigma=1.0)

# The library works directly with numpy arrays
if isinstance(processed, np.ndarray):
    print("Output is a numpy array!")
```

## Requirements

- Python >= 3.7
- NumPy >= 1.21.0


## License

MIT License 