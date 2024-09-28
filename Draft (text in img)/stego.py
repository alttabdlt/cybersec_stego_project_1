import numpy as np
from PIL import Image
import wave
import io
import cv2
import math
import os
import random
import re
from moviepy.editor import VideoFileClip


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

def encode_text_in_audio(audio_file, text, num_lsb):
    # Open the audio file using the wave module
    song = wave.open(audio_file, mode='rb')
    nframes = song.getnframes()
    frames = song.readframes(nframes)
    
    # Convert frames to bytearray for modification
    frame_bytes = bytearray(list(frames))

    # Convert text to binary and add a delimiter (e.g., '00000000')
    binary_text = ''.join(format(ord(i), '08b') for i in text) + '00000000'
    
    # Check if there is enough space in the audio file to hide the text
    if len(binary_text) > len(frame_bytes) * num_lsb:
        raise ValueError("Text is too long to fit in the audio file")

    # Modify the LSBs of each audio frame byte to encode the text
    data_index = 0
    for i in range(len(frame_bytes)):
        if data_index < len(binary_text):
            # Modify the least significant bits
            frame_bytes[i] = (frame_bytes[i] & (256 - 2**num_lsb)) | int(binary_text[data_index:data_index+num_lsb], 2)
            data_index += num_lsb

    # Create a new audio file with the encoded data
    stego_audio = io.BytesIO()
    with wave.open(stego_audio, 'wb') as fd:
        fd.setparams(song.getparams())
        fd.writeframes(bytes(frame_bytes))
    
    song.close()
    stego_audio.seek(0)
    return stego_audio

def decode_text_from_audio(audio_file, num_lsb):
    # Open the audio file
    song = wave.open(audio_file, mode='rb')
    nframes = song.getnframes()
    frames = song.readframes(nframes)

    # Convert frames to bytearray for easier manipulation
    frame_bytes = bytearray(list(frames))

    # Extract the LSBs of each byte in the audio
    extracted_bits = ''
    for i in range(len(frame_bytes)):
        # Extract the LSBs from the current byte without including the '0b' part
        extracted_bits += format(frame_bytes[i], '08b')[-num_lsb:]  # Get the last 'num_lsb' bits

        # Check every 8 bits for a null byte
        if len(extracted_bits) >= 8 and extracted_bits[-8:] == '00000000':
            break

    # Split the binary string into 8-bit chunks and convert to text
    byte_data = [extracted_bits[i:i+8] for i in range(0, len(extracted_bits), 8)]
    decoded_text = ''.join([chr(int(b, 2)) for b in byte_data])

    # Find the delimiter (null byte) and return the extracted text
    null_index = decoded_text.find('\x00')  # Null byte (00000000)
    if null_index != -1:
        decoded_text = decoded_text[:null_index]
    else:
        raise ValueError("No hidden text found")

    song.close()
    return decoded_text

