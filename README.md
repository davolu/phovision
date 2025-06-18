# Phovision

A pure Python computer vision library implementing various image processing algorithms from scratch. This library provides implementations of common image processing operations without depending on other image processing libraries.

## Features

Currently implemented algorithms:
- Gaussian Blur
- Median Filter (Noise Removal)
- Mean Filter (Averaging)
- Bilateral Filter (Edge-preserving smoothing)

## Installation

```bash
pip install phovision
```

## Usage

```python
import numpy as np
from phovision.filters import gaussian_blur, median_filter, mean_filter, bilateral_filter

# Load your image as a numpy array
# image should be in grayscale or RGB format with values in range [0, 255]

# Apply Gaussian Blur
blurred = gaussian_blur(image, kernel_size=5, sigma=1.0)

# Apply Median Filter
denoised = median_filter(image, kernel_size=3)

# Apply Mean Filter
averaged = mean_filter(image, kernel_size=3)

# Apply Bilateral Filter
smoothed = bilateral_filter(image, kernel_size=5, sigma_spatial=1.0, sigma_intensity=50.0)
```

## Requirements
- Python >= 3.7
- NumPy >= 1.21.0

## License
MIT License 