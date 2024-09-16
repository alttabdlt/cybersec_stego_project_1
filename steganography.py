import numpy as np
from PIL import Image
import io
import wave
import base64
from pydub import AudioSegment
import os
import matplotlib.pyplot as plt
import tempfile
import cv2

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    return ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))

def file_to_binary(file):
    return ''.join(format(byte, '08b') for byte in file.read())

def binary_to_file(binary, file_type):
    try:
        file_bytes = bytes(int(binary[i:i+8], 2) for i in range(0, len(binary), 8))
        return io.BytesIO(file_bytes)
    except Exception as e:
        raise ValueError(f"Error converting binary to file: {str(e)}")

def encode_lsb(data, payload, num_lsb):
    binary_payload = payload + '1111111111111110'
    required_bits = len(binary_payload)
    available_bits = data.size * num_lsb
    
    if required_bits > available_bits:
        raise ValueError(f"Payload too large. Needs {required_bits} bits, but only {available_bits} available.")
    
    binary_payload += '0' * (available_bits - required_bits)
    payload_bits = np.array([int(bit) for bit in binary_payload])
    
    mask = 2**num_lsb - 1
    encoded = ((data & ~mask) | (payload_bits[:data.size].reshape(data.shape) & mask)).astype(data.dtype)
    return encoded

def decode_lsb(data, num_lsb):
    mask = 2**num_lsb - 1
    binary_payload = ''.join(format(byte & mask, f'0{num_lsb}b') for byte in data.flatten())
    payload_end = binary_payload.find('1111111111111110')
    if payload_end == -1:
        raise ValueError("No hidden payload found")
    return binary_payload[:payload_end]

def encode_image(image_file, payload, num_lsb):
    img = Image.open(image_file)
    data = np.array(img)
    shape = data.shape
    encoded_data = encode_lsb(data.flatten(), payload, num_lsb)
    encoded_img = Image.fromarray(encoded_data.reshape(shape))
    return encoded_img

def decode_image(image_file, num_lsb):
    img = Image.open(image_file)
    data = np.array(img)
    return decode_lsb(data, num_lsb)

def encode_audio(audio_file, payload, num_lsb):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
            temp_file.write(audio_file.getvalue())
            temp_file.flush()
        
        audio = AudioSegment.from_file(temp_filename)
        audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)  # Ensure 16-bit integer type
        
        encoded_data = encode_lsb(audio_data, payload, num_lsb)
        
        encoded_audio = AudioSegment(
            encoded_data.tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )
        
        encoded_temp_filename = tempfile.mktemp(suffix='.wav')
        encoded_audio.export(encoded_temp_filename, format="wav")
        
        with open(encoded_temp_filename, 'rb') as encoded_file:
            encoded_content = encoded_file.read()
        
        os.unlink(temp_filename)
        os.unlink(encoded_temp_filename)
        
        return io.BytesIO(encoded_content), audio_data, encoded_data
    except Exception as e:
        raise ValueError(f"Error processing audio file: {str(e)}")

def decode_audio(audio_file, num_lsb):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
            
            temp_file.write(audio_file.getvalue())
            temp_file.flush()
            
            audio = AudioSegment.from_file(temp_filename)
            audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)  # Ensure 16-bit integer type
            
            os.unlink(temp_filename)
            
            return decode_lsb(audio_data, num_lsb), audio_data
    except Exception as e:
        raise ValueError(f"Error processing audio file: {str(e)}")

def visualize_audio(original, encoded):
    plt.figure(figsize=(12, 8))
    
    # Plot original waveform
    plt.subplot(2, 1, 1)
    plt.title("Original Audio")
    plt.plot(original)
    plt.ylim(min(original), max(original))
    
    # Plot encoded waveform
    plt.subplot(2, 1, 2)
    plt.title("Encoded Audio")
    plt.plot(encoded)
    plt.ylim(min(encoded), max(encoded))
    
    # Plot difference
    plt.subplot(2, 1, 3)
    plt.title("Difference (Encoded - Original)")
    difference = encoded - original
    plt.plot(difference)
    plt.ylim(min(difference), max(difference))
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def visualize_image(original, encoded):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(original)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Encoded image
    ax2.imshow(encoded)
    ax2.set_title("Encoded Image")
    ax2.axis('off')
    
    # Difference
    difference = np.abs(np.array(original, dtype=np.int16) - np.array(encoded, dtype=np.int16))
    ax3.imshow(difference, cmap='hot')
    ax3.set_title("Difference")
    ax3.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def visualize_video(original_video, encoded_video):
    temp_original = tempfile.mktemp(suffix='.mp4')
    temp_encoded = tempfile.mktemp(suffix='.mp4')
    
    with open(temp_original, 'wb') as f:
        f.write(original_video.getvalue())
    with open(temp_encoded, 'wb') as f:
        f.write(encoded_video.getvalue())
    
    cap_original = cv2.VideoCapture(temp_original)
    cap_encoded = cv2.VideoCapture(temp_encoded)
    
    ret_original, frame_original = cap_original.read()
    ret_encoded, frame_encoded = cap_encoded.read()
    
    if ret_original and ret_encoded:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original frame
        ax1.imshow(cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Video (First Frame)")
        ax1.axis('off')
        
        # Encoded frame
        ax2.imshow(cv2.cvtColor(frame_encoded, cv2.COLOR_BGR2RGB))
        ax2.set_title("Encoded Video (First Frame)")
        ax2.axis('off')
        
        # Difference
        difference = np.abs(frame_original.astype(np.int16) - frame_encoded.astype(np.int16))
        ax3.imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB), cmap='hot')
        ax3.set_title("Difference")
        ax3.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        cap_original.release()
        cap_encoded.release()
        os.unlink(temp_original)
        os.unlink(temp_encoded)
        
        return buf
    else:
        raise ValueError("Error reading video files")

def advanced_decode(stego_file, num_lsb, payload_type):
    if stego_file.type.startswith('image'):
        binary_payload = decode_image(stego_file, num_lsb)
    elif stego_file.type.startswith('audio'):
        binary_payload, _ = decode_audio(stego_file, num_lsb)
    elif stego_file.type.startswith('video'):
        binary_payload = decode_video(stego_file, num_lsb)
    elif stego_file.type == 'text/plain':
        return decode_text(stego_file.getvalue().decode('utf-8'))
    else:
        raise ValueError("Unsupported file type")
    
    if payload_type == "Text":
        return binary_to_text(binary_payload)
    elif payload_type in ["Image", "Audio", "Document", "Video"]:
        return binary_to_file(binary_payload, payload_type.lower())
    else:
        raise ValueError("Unsupported payload type")

def can_encode(cover_file, payload, num_lsb):
    if cover_file.type.startswith('image'):
        img = Image.open(cover_file)
        data = np.array(img)
    elif cover_file.type.startswith('audio'):
        audio = AudioSegment.from_file(cover_file)
        data = np.array(audio.get_array_of_samples(), dtype=np.int16)  # Ensure 16-bit integer type
    elif cover_file.type.startswith('video'):
        # Add video handling if necessary
        pass
    else:
        raise ValueError("Unsupported file type")
    
    available_bits = data.size * num_lsb
    required_bits = len(payload) + 16  # payload is already in binary, 16 bits for end marker
    return required_bits <= available_bits, available_bits, required_bits

def encode_video(video_file, payload, num_lsb):
    temp_input = tempfile.mktemp(suffix='.mp4')
    temp_output = tempfile.mktemp(suffix='.mp4')
    
    with open(temp_input, 'wb') as f:
        f.write(video_file.getvalue())
    
    cap = cv2.VideoCapture(temp_input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    payload_bits = payload + '1' * 16  # Add end marker
    bits_per_frame = len(payload_bits) // frame_count + 1
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        start = i * bits_per_frame
        end = min((i + 1) * bits_per_frame, len(payload_bits))
        frame_payload = payload_bits[start:end]
        
        if len(frame_payload) > 0:
            encoded_frame = encode_lsb(frame, frame_payload, num_lsb)
            out.write(encoded_frame)
        else:
            out.write(frame)
    
    cap.release()
    out.release()
    
    with open(temp_output, 'rb') as f:
        encoded_content = f.read()
    
    os.unlink(temp_input)
    os.unlink(temp_output)
    
    return io.BytesIO(encoded_content)

def decode_video(video_file, num_lsb):
    temp_input = tempfile.mktemp(suffix='.mp4')
    
    with open(temp_input, 'wb') as f:
        f.write(video_file.getvalue())
    
    cap = cv2.VideoCapture(temp_input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    payload_bits = ""
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_payload = decode_lsb(frame, num_lsb)
        payload_bits += frame_payload
        
        if payload_bits.endswith('1' * 16):
            break
    
    cap.release()
    os.unlink(temp_input)
    
    return payload_bits[:-16]  # Remove end marker

def encode_text(cover_text, payload):
    cover_words = cover_text.split()
    payload_bits = text_to_binary(payload) + '1' * 16  # Add end marker
    
    encoded_words = []
    bit_index = 0
    for word in cover_words:
        if bit_index < len(payload_bits):
            if payload_bits[bit_index] == '1':
                word += '\u200b'  # Zero-width space
            bit_index += 1
        encoded_words.append(word)
    
    return ' '.join(encoded_words)

def decode_text(stego_text):
    words = stego_text.split()
    payload_bits = ''
    
    for word in words:
        if word.endswith('\u200b'):
            payload_bits += '1'
        else:
            payload_bits += '0'
        
        if payload_bits.endswith('1' * 16):
            break
    
    return binary_to_text(payload_bits[:-16])  # Remove end marker