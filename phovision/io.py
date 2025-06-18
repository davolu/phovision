import numpy as np
import base64
import re
from urllib.request import urlopen
from urllib.parse import urlparse
import os

def _is_url(path):
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def _is_base64(string):
    """Check if the given string is base64 encoded."""
    base64_pattern = r'^data:image/[a-zA-Z]+;base64,'
    return bool(re.match(base64_pattern, string))

def _read_file_bytes(path):
    """Read bytes from a file or URL."""
    if _is_url(path):
        with urlopen(path) as response:
            return response.read()
    else:
        with open(path, 'rb') as f:
            return f.read()

def read_image(source):
    """
    Read an image from various sources and convert it to a numpy array.
    
    Args:
        source: Can be one of:
            - Path to image file (str)
            - URL to image (str)
            - Base64 encoded image string (str)
            - Bytes object containing image data (bytes)
            - numpy array (will be returned as-is if valid)
    
    Returns:
        numpy.ndarray: Image as a numpy array with shape (height, width, channels)
                      or (height, width) for grayscale
    
    Raises:
        ValueError: If the image format is not supported or the input is invalid
        FileNotFoundError: If the image file doesn't exist
    """
    # If already a numpy array, validate and return
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise ValueError("Numpy array must be of type uint8")
        if len(source.shape) not in [2, 3]:
            raise ValueError("Image array must be 2D (grayscale) or 3D (RGB/RGBA)")
        return source
    
    # If not a string or bytes, raise error
    if not isinstance(source, (str, bytes)):
        raise ValueError("Source must be a file path, URL, base64 string, bytes, or numpy array")
    
    # Get image bytes
    if isinstance(source, str):
        if _is_base64(source):
            # Extract actual base64 data
            base64_data = source.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
        else:
            image_bytes = _read_file_bytes(source)
    else:
        image_bytes = source
    
    # Convert bytes to numpy array
    try:
        # Try using PIL first (supports most formats)
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))
            return np.array(img)
        except ImportError:
            # Fallback to OpenCV if PIL is not available
            try:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Failed to decode image")
                # Convert BGR to RGB if necessary
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            except ImportError:
                raise ImportError("Either PIL or OpenCV is required for image reading")
    except Exception as e:
        raise ValueError(f"Failed to read image: {str(e)}")

def save_image(image, path, format=None):
    """
    Save a numpy array as an image file.
    
    Args:
        image (numpy.ndarray): Image array to save
        path (str): Path where to save the image
        format (str, optional): Format to save as (e.g., 'PNG', 'JPEG', 'BMP')
                              If None, will be inferred from the file extension
    
    Raises:
        ValueError: If the image format is not supported or the input is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.dtype != np.uint8:
        raise ValueError("Image array must be of type uint8")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image array must be 2D (grayscale) or 3D (RGB/RGBA)")
    
    # Try using PIL first
    try:
        from PIL import Image
        img = Image.fromarray(image)
        img.save(path, format=format)
    except ImportError:
        # Fallback to OpenCV
        try:
            import cv2
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)
        except ImportError:
            raise ImportError("Either PIL or OpenCV is required for image saving")

def to_base64(image, format='PNG'):
    """
    Convert a numpy array image to a base64 string.
    
    Args:
        image (numpy.ndarray): Image array to convert
        format (str): Format to encode as (default: 'PNG')
    
    Returns:
        str: Base64 encoded image string with data URI scheme
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    try:
        from PIL import Image
        import io
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/{format.lower()};base64,{img_str}'
    except ImportError:
        try:
            import cv2
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(f'.{format.lower()}', image)
            img_str = base64.b64encode(buffer).decode()
            return f'data:image/{format.lower()};base64,{img_str}'
        except ImportError:
            raise ImportError("Either PIL or OpenCV is required for base64 conversion") 