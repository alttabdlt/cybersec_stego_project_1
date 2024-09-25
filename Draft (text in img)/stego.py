import numpy as np
from PIL import Image
import wave

def encode_text_in_image(image_path, text, num_lsb):
    img = Image.open(image_path)
    pixels = np.array(img)
    
    # Convert text to binary
    binary_text = ''.join(format(ord(char), '08b') for char in text)
    
    if len(binary_text) > pixels.size * num_lsb:
        raise ValueError("Text too large for the cover image with given LSB")
    
    binary_text += '00000000'  # Add delimiter
    
    data_index = 0
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            for k in range(pixels.shape[2]):
                if data_index < len(binary_text):
                    pixels[i, j, k] = (pixels[i, j, k] & (256 - 2**num_lsb)) | int(binary_text[data_index:data_index+num_lsb], 2)
                    data_index += num_lsb
    
    stego_img = Image.fromarray(pixels)
    return stego_img

def decode_text_from_image(image_path, num_lsb):
    img = Image.open(image_path)
    pixels = np.array(img)
    
    # Flatten the pixel array and extract LSBs
    flattened = pixels.flatten()
    binary_data = np.unpackbits(flattened[:, np.newaxis], axis=1)[:, -num_lsb:].flatten()
    
    # Pad the binary data to ensure it's divisible by 8
    padding = 8 - (len(binary_data) % 8)
    if padding < 8:
        binary_data = np.pad(binary_data, (0, padding), 'constant')
    
    # Convert bits to bytes
    byte_data = np.packbits(binary_data)
    
    # Find the delimiter (null byte)
    delim_index = np.where(byte_data == 0)[0]
    if len(delim_index) == 0:
        raise ValueError("No hidden text found")
    
    # Extract the message bytes
    message_bytes = byte_data[:delim_index[0]]
    
    # Convert bytes to string
    return message_bytes.tobytes().decode('utf-8', errors='ignore')


