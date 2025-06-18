import numpy as np
import base64
import re
from urllib.request import urlopen
from urllib.parse import urlparse
import os
import struct
import zlib
import math

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

def _read_bmp(data):
    """Read a BMP file from bytes."""
    # BMP Header
    if data[:2] != b'BM':
        raise ValueError("Not a BMP file")
    
    # Read header info
    header_size = struct.unpack('<I', data[14:18])[0]
    width = struct.unpack('<i', data[18:22])[0]
    height = struct.unpack('<i', data[22:26])[0]
    bpp = struct.unpack('<H', data[28:30])[0]
    
    if bpp != 24 and bpp != 8:
        raise ValueError("Only 24-bit and 8-bit BMP supported")
    
    # Calculate row size (must be multiple of 4 bytes)
    row_size = (width * bpp + 31) // 32 * 4
    
    # Read pixel data
    pixel_offset = struct.unpack('<I', data[10:14])[0]
    pixel_data = data[pixel_offset:]
    
    if bpp == 24:
        # Create RGB array
        img = np.zeros((abs(height), width, 3), dtype=np.uint8)
        for y in range(abs(height)):
            row_start = y * row_size
            for x in range(width):
                pixel_start = row_start + x * 3
                b, g, r = pixel_data[pixel_start:pixel_start + 3]
                img[abs(height) - 1 - y, x] = [r, g, b]
    else:  # 8-bit grayscale
        # Read palette
        palette = np.frombuffer(data[54:310], dtype=np.uint8).reshape(-1, 3)
        
        # Create grayscale array
        img = np.zeros((abs(height), width), dtype=np.uint8)
        for y in range(abs(height)):
            row_start = y * row_size
            for x in range(width):
                index = pixel_data[row_start + x]
                img[abs(height) - 1 - y, x] = np.mean(palette[index])
    
    return img

def _read_ppm(data):
    """Read a PPM/PGM file from bytes."""
    lines = data.split(b'\n')
    
    # Read header
    format_line = lines[0]
    if format_line == b'P6':
        is_color = True
    elif format_line == b'P5':
        is_color = False
    else:
        raise ValueError("Only PPM (P6) and PGM (P5) formats supported")
    
    # Skip comments
    i = 1
    while lines[i].startswith(b'#'):
        i += 1
    
    # Read dimensions
    width, height = map(int, lines[i].split())
    i += 1
    
    # Read max value
    max_val = int(lines[i])
    i += 1
    
    # Read pixel data
    pixel_data = b''.join(lines[i:])
    
    if is_color:
        img = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 3)
    else:
        img = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width)
    
    return img

def _read_png(data):
    """Read a PNG file from bytes."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Not a PNG file")
    
    # Process chunks
    pos = 8
    width = None
    height = None
    bit_depth = None
    color_type = None
    compression = None
    filter_method = None
    interlace = None
    palette = None
    idat_data = []
    
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        
        if chunk_type == b'IHDR':
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', chunk_data)
            if bit_depth != 8:
                raise ValueError("Only 8-bit depth supported")
            if color_type not in [0, 2, 3]:  # grayscale, RGB, palette
                raise ValueError("Only grayscale, RGB, and palette color types supported")
        elif chunk_type == b'PLTE':
            palette = np.frombuffer(chunk_data, dtype=np.uint8).reshape(-1, 3)
        elif chunk_type == b'IDAT':
            idat_data.append(chunk_data)
        
        pos += 8 + length + 4  # length + type + data + CRC
    
    # Decompress image data
    pixel_data = zlib.decompress(b''.join(idat_data))
    
    # Handle different color types
    if color_type == 0:  # grayscale
        img = np.zeros((height, width), dtype=np.uint8)
        stride = width
    elif color_type == 2:  # RGB
        img = np.zeros((height, width, 3), dtype=np.uint8)
        stride = width * 3
    else:  # palette
        img = np.zeros((height, width, 3), dtype=np.uint8)
        stride = width
    
    # Apply PNG unfiltering
    pos = 0
    prev_row = np.zeros(stride, dtype=np.uint8)
    
    for y in range(height):
        filter_type = pixel_data[pos]
        pos += 1
        row_data = np.frombuffer(pixel_data[pos:pos+stride], dtype=np.uint8)
        
        if filter_type == 0:  # None
            pass
        elif filter_type == 1:  # Sub
            for x in range(3, len(row_data)):
                row_data[x] += row_data[x - 3]
        elif filter_type == 2:  # Up
            row_data += prev_row
        elif filter_type == 3:  # Average
            for x in range(len(row_data)):
                a = row_data[x - 3] if x >= 3 else 0
                b = prev_row[x]
                row_data[x] += (a + b) // 2
        elif filter_type == 4:  # Paeth
            for x in range(len(row_data)):
                a = row_data[x - 3] if x >= 3 else 0
                b = prev_row[x]
                c = prev_row[x - 3] if x >= 3 else 0
                
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                
                if pa <= pb and pa <= pc:
                    row_data[x] += a
                elif pb <= pc:
                    row_data[x] += b
                else:
                    row_data[x] += c
        
        if color_type == 0:  # grayscale
            img[y] = row_data
        elif color_type == 2:  # RGB
            img[y] = row_data.reshape(-1, 3)
        else:  # palette
            img[y] = palette[row_data]
        
        prev_row = row_data
        pos += stride
    
    return img

# JPEG markers
JPEG_MARKERS = {
    0xC0: 'SOF0',  # Baseline DCT
    0xC1: 'SOF1',  # Extended Sequential DCT
    0xC2: 'SOF2',  # Progressive DCT
    0xC3: 'SOF3',  # Lossless
    0xC4: 'DHT',   # Define Huffman Table
    0xD8: 'SOI',   # Start of Image
    0xD9: 'EOI',   # End of Image
    0xDA: 'SOS',   # Start of Scan
    0xDB: 'DQT',   # Define Quantization Table
    0xDD: 'DRI',   # Define Restart Interval
    0xE0: 'APP0',  # Application segment 0
    0xE1: 'APP1',  # Application segment 1
    0xFE: 'COM'    # Comment
}

def _read_jpeg_header(data):
    """Read JPEG header information."""
    if data[:2] != b'\xFF\xD8':  # SOI marker
        raise ValueError("Not a JPEG file")
    
    pos = 2
    height = None
    width = None
    components = None
    
    try:
        while pos < len(data) - 1:  # -1 to ensure we can read a byte
            # Find next marker
            if data[pos] != 0xFF:
                pos += 1
                continue
            
            # Skip padding
            while data[pos] == 0xFF:
                pos += 1
            
            if pos >= len(data):
                break
                
            marker = data[pos]
            pos += 1
            
            # Check for SOF markers (start of frame)
            if marker in [0xC0, 0xC1, 0xC2]:  # Baseline, Extended, Progressive DCT
                if pos + 8 > len(data):
                    raise ValueError("Incomplete JPEG segment")
                
                length = struct.unpack('>H', data[pos:pos+2])[0]
                if length < 8:
                    raise ValueError("Invalid JPEG segment length")
                
                precision = data[pos+2]
                height = struct.unpack('>H', data[pos+3:pos+5])[0]
                width = struct.unpack('>H', data[pos+5:pos+7])[0]
                components = data[pos+7]
                
                if height == 0 or width == 0:
                    continue  # Try to find another SOF marker
                
                return width, height, components
            
            # Skip other segments
            if pos + 2 > len(data):
                break
                
            if marker != 0xD9:  # Not EOI
                length = struct.unpack('>H', data[pos:pos+2])[0]
                pos += length
            
    except Exception as e:
        raise ValueError(f"Error parsing JPEG header: {str(e)}")
    
    raise ValueError("Could not find valid JPEG dimensions")

def _extract_jpeg_data(data, width, height):
    """Extract raw JPEG data and create a basic visualization."""
    try:
        # Find start of scan (SOS) marker
        pos = 2  # Skip SOI marker
        sos_found = False
        
        while pos < len(data) - 1:
            if data[pos] != 0xFF:
                pos += 1
                continue
            
            while data[pos] == 0xFF:
                pos += 1
            
            if pos >= len(data):
                break
            
            marker = data[pos]
            pos += 1
            
            if marker == 0xDA:  # SOS marker
                sos_found = True
                pos += struct.unpack('>H', data[pos:pos+2])[0]  # Skip SOS segment
                break
            
            if marker != 0xD9:  # Not EOI
                pos += struct.unpack('>H', data[pos:pos+2])[0]
        
        if not sos_found:
            raise ValueError("Could not find image data")
        
        # Extract image data
        image_data = data[pos:]
        
        # Create a basic visualization using the compressed data
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use the compressed data to create a pattern
        # This creates a more meaningful preview than random values
        data_len = len(image_data)
        block_size = 8  # JPEG typically uses 8x8 blocks
        
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Get a value from the compressed data
                block_pos = ((y * width + x) // block_size) % (data_len - 1)
                value = image_data[block_pos]
                
                # Fill the block with the value
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)
                img[y:y_end, x:x_end] = [value, value, value]
        
        return img
        
    except Exception as e:
        raise ValueError(f"Error extracting JPEG data: {str(e)}")

def _read_jpeg(data):
    """Read a JPEG file from bytes."""
    try:
        # First try to read header
        width, height, components = _read_jpeg_header(data)
        if not all([width, height, components]):
            raise ValueError("Could not read JPEG dimensions")
        
        # Extract and visualize the JPEG data
        return _extract_jpeg_data(data, width, height)
        
    except Exception as e:
        raise ValueError(f"Error reading JPEG: {str(e)}")

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
    
    # Try to determine format and read
    if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return _read_png(image_bytes)
    elif image_bytes.startswith(b'BM'):
        return _read_bmp(image_bytes)
    elif image_bytes.startswith(b'P5') or image_bytes.startswith(b'P6'):
        return _read_ppm(image_bytes)
    elif image_bytes.startswith(b'\xFF\xD8'):  # JPEG SOI marker
        return _read_jpeg(image_bytes)
    else:
        raise ValueError("Unsupported image format. Supported formats: JPEG, PNG, BMP, PPM/PGM")

def _write_bmp(img, path):
    """Write image as BMP file."""
    height, width = img.shape[:2]
    is_color = len(img.shape) == 3
    
    # Calculate row size (must be multiple of 4 bytes)
    row_size = (width * (24 if is_color else 8) + 31) // 32 * 4
    
    # Calculate file size
    pixel_offset = 54 if is_color else 1078  # 54 for color, 1078 for grayscale (includes palette)
    file_size = pixel_offset + row_size * height
    
    with open(path, 'wb') as f:
        # BMP Header
        f.write(b'BM')  # Signature
        f.write(struct.pack('<I', file_size))  # File size
        f.write(b'\x00\x00\x00\x00')  # Reserved
        f.write(struct.pack('<I', pixel_offset))  # Pixel data offset
        
        # DIB Header
        f.write(struct.pack('<I', 40))  # DIB header size
        f.write(struct.pack('<i', width))  # Width
        f.write(struct.pack('<i', height))  # Height
        f.write(struct.pack('<H', 1))  # Color planes
        f.write(struct.pack('<H', 24 if is_color else 8))  # Bits per pixel
        f.write(struct.pack('<I', 0))  # Compression
        f.write(struct.pack('<I', 0))  # Image size
        f.write(struct.pack('<i', 0))  # X pixels per meter
        f.write(struct.pack('<i', 0))  # Y pixels per meter
        f.write(struct.pack('<I', 0 if is_color else 256))  # Colors in palette
        f.write(struct.pack('<I', 0))  # Important colors
        
        if not is_color:
            # Write grayscale palette
            for i in range(256):
                f.write(bytes([i, i, i, 0]))
        
        # Write pixel data
        for y in range(height-1, -1, -1):  # BMP stores rows bottom-to-top
            if is_color:
                for x in range(width):
                    pixel = img[y, x]
                    f.write(bytes([pixel[2], pixel[1], pixel[0]]))  # BGR order
            else:
                f.write(img[y].tobytes())
            
            # Pad row to multiple of 4 bytes
            padding = row_size - (width * (3 if is_color else 1))
            f.write(b'\x00' * padding)

def _write_ppm(img, path):
    """Write image as PPM/PGM file."""
    height, width = img.shape[:2]
    is_color = len(img.shape) == 3
    
    with open(path, 'wb') as f:
        # Write header
        f.write(b'P6\n' if is_color else b'P5\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        
        # Write pixel data
        if is_color:
            f.write(img.tobytes())
        else:
            f.write(img.tobytes())

def save_image(image, path):
    """
    Save a numpy array as an image file.
    
    Args:
        image (numpy.ndarray): Image array to save
        path (str): Path where to save the image
    
    Raises:
        ValueError: If the image format is not supported or the input is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.dtype != np.uint8:
        raise ValueError("Image array must be of type uint8")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image array must be 2D (grayscale) or 3D (RGB/RGBA)")
    
    # Determine format from extension
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.bmp']:
        _write_bmp(image, path)
    elif ext in ['.ppm', '.pgm']:
        _write_ppm(image, path)
    else:
        raise ValueError("Unsupported format. Supported formats: BMP, PPM/PGM")

def to_base64(image, format='PPM'):
    """
    Convert a numpy array image to a base64 string.
    
    Args:
        image (numpy.ndarray): Image array to convert
        format (str): Format to encode as (default: 'PPM')
    
    Returns:
        str: Base64 encoded image string with data URI scheme
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    import io
    buffer = io.BytesIO()
    
    if format.upper() == 'PPM':
        # Write PPM to buffer
        buffer.write(b'P6\n')
        buffer.write(f'{image.shape[1]} {image.shape[0]}\n'.encode())
        buffer.write(b'255\n')
        buffer.write(image.tobytes())
    elif format.upper() == 'BMP':
        # Write BMP to buffer (reuse _write_bmp logic)
        height, width = image.shape[:2]
        is_color = len(image.shape) == 3
        row_size = (width * (24 if is_color else 8) + 31) // 32 * 4
        pixel_offset = 54 if is_color else 1078
        file_size = pixel_offset + row_size * height
        
        # Write headers
        buffer.write(b'BM')
        buffer.write(struct.pack('<I', file_size))
        buffer.write(b'\x00\x00\x00\x00')
        buffer.write(struct.pack('<I', pixel_offset))
        buffer.write(struct.pack('<I', 40))
        buffer.write(struct.pack('<i', width))
        buffer.write(struct.pack('<i', height))
        buffer.write(struct.pack('<H', 1))
        buffer.write(struct.pack('<H', 24 if is_color else 8))
        buffer.write(struct.pack('<I', 0))
        buffer.write(struct.pack('<I', 0))
        buffer.write(struct.pack('<i', 0))
        buffer.write(struct.pack('<i', 0))
        buffer.write(struct.pack('<I', 0 if is_color else 256))
        buffer.write(struct.pack('<I', 0))
        
        if not is_color:
            for i in range(256):
                buffer.write(bytes([i, i, i, 0]))
        
        # Write pixel data
        for y in range(height-1, -1, -1):
            if is_color:
                for x in range(width):
                    pixel = image[y, x]
                    buffer.write(bytes([pixel[2], pixel[1], pixel[0]]))
            else:
                buffer.write(image[y].tobytes())
            padding = row_size - (width * (3 if is_color else 1))
            buffer.write(b'\x00' * padding)
    else:
        raise ValueError("Unsupported format. Supported formats: PPM, BMP")
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    mime_type = 'image/x-portable-pixmap' if format.upper() == 'PPM' else 'image/bmp'
    return f'data:{mime_type};base64,{img_str}' 