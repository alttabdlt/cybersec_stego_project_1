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
import struct
from tqdm import tqdm
import subprocess

def can_encode(cover_file, payload, num_lsb):
    try:
        cover_file.seek(0)  # Reset file pointer to the beginning
        if cover_file.type.startswith('image'):
            img = Image.open(cover_file)
            data = np.array(img)
            available_bits = data.size * num_lsb
        elif cover_file.type.startswith('audio'):
            audio = AudioSegment.from_file(cover_file)
            data = np.array(audio.get_array_of_samples(), dtype=np.int16)
            available_bits = data.size * num_lsb
        elif cover_file.type.startswith('video'):
            # Get the original file extension
            original_extension = os.path.splitext(cover_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as temp_file:
                cover_file.seek(0)
                temp_file.write(cover_file.read())
                temp_file_path = temp_file.name

            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                cap.release()
                os.unlink(temp_file_path)
                raise ValueError("Unable to open video file. Please ensure the video format is supported and necessary codecs are installed.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            bits_per_frame = width * height * 3 * num_lsb
            available_bits = frame_count * bits_per_frame

            cap.release()
            os.unlink(temp_file_path)
        else:
            raise ValueError("Unsupported cover file type.")

        # Calculate required bits
        if isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        else:
            payload.seek(0)
            payload_bytes = payload.read()

        payload_length = len(payload_bytes)
        payload_length_bits = 32  # 32 bits for the payload length header
        payload_bits = payload_length * 8
        total_bits_needed = payload_length_bits + payload_bits

        return total_bits_needed <= available_bits, available_bits, total_bits_needed
    except Exception as e:
        raise ValueError(f"Error processing cover file: {str(e)}")

def text_to_binary(text): 
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    chars = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def image_to_binary(image_file):
    img = Image.open(image_file)
    img_array = np.array(img)
    binary_list = [format(pixel, '08b') for pixel in img_array.flatten()]
    return ''.join(binary_list)

def binary_to_image(binary_data, dimensions):
    height, width, channels = dimensions
    total_pixels = height * width * channels

    if len(binary_data) < total_pixels * 8:
        raise ValueError("Insufficient binary data to reconstruct the image")

    pixel_values = []
    for i in range(0, total_pixels * 8, 8):
        byte = binary_data[i:i+8]
        pixel_values.append(int(byte, 2))

    img_array = np.array(pixel_values, dtype=np.uint8).reshape((height, width, channels))
    img = Image.fromarray(img_array)
    return img

def encode_video(cover_file, payload_bytes, num_lsb):
    try:
        cover_file.seek(0)
        print("Starting encode_video...")
        print(f"Number of LSBs used: {num_lsb}")

        # Save the input video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_video:
            temp_input_video.write(cover_file.read())
            temp_input_video_path = temp_input_video.name
        print(f"Temporary input video path: {temp_input_video_path}")

        cap = cv2.VideoCapture(temp_input_video_path)
        if not cap.isOpened():
            cap.release()
            os.remove(temp_input_video_path)
            raise ValueError("Unable to open video file for encoding.")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: FPS={fps}, Width={frame_width}, Height={frame_height}, Frame count={frame_count}")

        # Prepare the payload
        payload_length = len(payload_bytes)
        print(f"Payload length: {payload_length} bytes")

        # Convert payload length to 32-bit big-endian binary string
        payload_length_bytes = struct.pack('>I', payload_length)
        payload_length_bits = ''.join(f'{byte:08b}' for byte in payload_length_bytes)
        print(f"Payload length bits ({len(payload_length_bits)} bits): {payload_length_bits}")

        # Convert payload bytes to bit string
        payload_bits = ''.join(f'{byte:08b}' for byte in payload_bytes)

        total_bits_needed = len(payload_bits)
        print(f"Total bits needed for payload: {total_bits_needed} bits")

        # Calculate total available bits in the video excluding the first frame
        bits_per_frame = frame_width * frame_height * 3 * num_lsb
        total_available_bits = (frame_count - 1) * bits_per_frame
        print(f"Total available bits in video (excluding first frame): {total_available_bits} bits")

        if total_bits_needed > total_available_bits:
            cap.release()
            os.remove(temp_input_video_path)
            raise ValueError("Payload is too large to encode in the provided video.")

        # Create a temporary directory to store frames
        temp_frame_dir = tempfile.mkdtemp()
        print(f"Temporary frame directory: {temp_frame_dir}")

        # Read frames and save as PNG images
        frame_idx = 0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_frame_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            frame_idx += 1

        cap.release()
        print(f"Total frames extracted: {len(frames)}")

        # Encode payload length into the first frame
        first_frame = cv2.imread(frames[0])
        print("First frame pixel values before encoding (first 10 pixels):")
        print(first_frame.flatten()[:10])

        frame_capacity = first_frame.size  # Capacity when using 1 LSB per pixel
        if len(payload_length_bits) > frame_capacity:
            os.remove(temp_input_video_path)
            raise ValueError("First frame does not have enough capacity to encode the payload length.")

        first_frame_encoded = encode_frame(first_frame, payload_length_bits, num_lsb=1)
        print("First frame pixel values after encoding (first 10 pixels):")
        print(first_frame_encoded.flatten()[:10])

        # Save the encoded first frame
        cv2.imwrite(frames[0], first_frame_encoded)

        # Encode payload into subsequent frames
        bit_index = 0
        payload_bits_remaining = payload_bits
        for idx in tqdm(range(1, len(frames)), desc="Encoding frames"):
            if bit_index >= total_bits_needed:
                break
            frame = cv2.imread(frames[idx])
            frame_capacity = frame.size * num_lsb

            bits_to_encode = payload_bits_remaining[:frame_capacity]
            bits_encoded = len(bits_to_encode)
            payload_bits_remaining = payload_bits_remaining[bits_encoded:]

            frame_encoded = encode_frame(frame, bits_to_encode, num_lsb)
            cv2.imwrite(frames[idx], frame_encoded)

            bit_index += bits_encoded
            print(f"Frame {idx}: encoded {bits_encoded} bits, total bits encoded: {bit_index}")

        if bit_index < total_bits_needed:
            os.remove(temp_input_video_path)
            raise ValueError("Not all payload bits were encoded into the frames.")

        print("All payload bits have been encoded into the frames.")

        # Reassemble frames into a video using FFmpeg with a lossless codec
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv').name
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-framerate', str(fps),
            '-i', os.path.join(temp_frame_dir, 'frame_%05d.png'),
            '-c:v', 'ffv1',  # Use FFV1 lossless codec
            output_video_path
        ]
        print(f"Reassembling video with FFmpeg using command: {' '.join(ffmpeg_cmd)}")

        process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            print(process.stderr.decode())
            raise ValueError("Error reassembling video with FFmpeg.")

        print("Video reassembled from encoded frames.")

        # Clean up temporary files
        for frame_path in frames:
            os.remove(frame_path)
        os.rmdir(temp_frame_dir)
        os.remove(temp_input_video_path)
        print("Temporary files removed.")

        return output_video_path

    except Exception as e:
        print(f"Error during encode_video: {e}")
        raise ValueError(f"Error encoding video: {str(e)}")

def encode_frame(frame, data_bits, num_lsb):
    frame_flat = frame.flatten()
    max_capacity = frame_flat.size * num_lsb

    if len(data_bits) > max_capacity:
        raise ValueError("Not enough space in the frame to encode data.")

    data_index = 0
    mask = (1 << num_lsb) - 1
    inverse_mask = 0xFF ^ mask  # Inverse mask to clear the LSBs

    for i in range(len(frame_flat)):
        if data_index >= len(data_bits):
            break

        # Get the bits to encode into this pixel
        bits_to_encode = data_bits[data_index:data_index + num_lsb]
        if len(bits_to_encode) < num_lsb:
            bits_to_encode = bits_to_encode.ljust(num_lsb, '0')
        bits = int(bits_to_encode, 2)

        # Clear the LSBs in the pixel and set them to the bits to encode
        frame_flat[i] = (frame_flat[i] & inverse_mask) | bits

        data_index += num_lsb

    encoded_frame = frame_flat.reshape(frame.shape)
    return encoded_frame

def decode_video(stego_video_file, num_lsb):
    try:
        stego_video_file.seek(0)
        print("Starting decode_video...")
        print(f"Number of LSBs used: {num_lsb}")

        # Save the stego video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mkv') as temp_stego_video:
            temp_stego_video.write(stego_video_file.read())
            temp_stego_video_path = temp_stego_video.name
        print(f"Temporary stego video path: {temp_stego_video_path}")

        cap = cv2.VideoCapture(temp_stego_video_path)
        if not cap.isOpened():
            cap.release()
            os.remove(temp_stego_video_path)
            raise ValueError("Unable to open video file for decoding.")

        # Create a temporary directory to store frames
        temp_frame_dir = tempfile.mkdtemp()
        print(f"Temporary frame directory: {temp_frame_dir}")

        # Extract frames from the video
        frame_idx = 0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_frame_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            frame_idx += 1

        cap.release()
        print(f"Total frames extracted: {len(frames)}")

        if len(frames) == 0:
            os.remove(temp_stego_video_path)
            raise ValueError("No frames found in the video.")

        # Read the first frame to extract the payload length
        first_frame = cv2.imread(frames[0])
        print("First frame pixel values during decoding (first 10 pixels):")
        print(first_frame.flatten()[:10])

        # Decode payload length (32 bits) using 1 LSB
        payload_length_bits = decode_frame(first_frame, 32, num_lsb=1)
        print(f"Extracted payload length bits ({len(payload_length_bits)} bits): {payload_length_bits}")
        payload_length = int(payload_length_bits, 2)
        print(f"Decoded payload length from header: {payload_length} bytes")

        total_bits_needed = payload_length * 8  # Total bits for the payload data
        print(f"Total bits needed for payload: {total_bits_needed} bits")

        # Initialize variables for bit collection
        bits_collected = 0
        byte_str = ''
        payload_bytes = bytearray()

        # Read frames and collect bits
        for idx in tqdm(range(1, len(frames)), desc="Decoding frames"):
            if bits_collected >= total_bits_needed:
                break
            frame = cv2.imread(frames[idx])
            frame_capacity = frame.size * num_lsb
            bits_to_extract = min(frame_capacity, total_bits_needed - bits_collected)
            frame_bits = decode_frame(frame, bits_to_extract, num_lsb)
            bits_collected += len(frame_bits)

            # Process bits into bytes incrementally
            for bit in frame_bits:
                byte_str += bit
                if len(byte_str) == 8:
                    payload_bytes.append(int(byte_str, 2))
                    byte_str = ''

            print(f"Frame {idx}: collected {len(frame_bits)} bits, total bits collected: {bits_collected}")

        # Handle any remaining bits
        if len(byte_str) > 0:
            byte_str = byte_str.ljust(8, '0')
            payload_bytes.append(int(byte_str, 2))

        os.remove(temp_stego_video_path)
        # Clean up temporary frames
        for frame_path in frames:
            os.remove(frame_path)
        os.rmdir(temp_frame_dir)
        print("Temporary files removed.")

        if bits_collected < total_bits_needed:
            raise ValueError("Incomplete payload detected. Not enough bits were read.")

        # Return the payload bytes up to the specified payload_length
        decoded_payload = bytes(payload_bytes[:payload_length])

        # Save decoded payload to a file for inspection
        with open('decoded_payload', 'wb') as f:
            f.write(decoded_payload)
        print("Decoded payload saved to 'decoded_payload' file.")

        return decoded_payload

    except Exception as e:
        print(f"Error during decode_video: {e}")
        raise ValueError(f"Error decoding video: {str(e)}")

def decode_frame(frame, num_bits, num_lsb):
    frame_flat = frame.flatten()
    bits_extracted = ''
    data_index = 0
    mask = (1 << num_lsb) - 1

    for i in range(len(frame_flat)):
        if data_index >= num_bits:
            break

        pixel_value = frame_flat[i]
        bits = pixel_value & mask  # Extract the LSBs
        bits_extracted += f'{bits:0{num_lsb}b}'
        data_index += num_lsb

    return bits_extracted[:num_bits]

def advanced_decode(stego_file, num_lsb, payload_type):
    if stego_file.type.startswith('video'):
        decoded_payload = decode_video(stego_file, num_lsb)
        return decoded_payload
    else:
        # Handle other file types (image, audio, text) if necessary
        if stego_file.type.startswith('image'):
            return decode_image(stego_file, num_lsb)
        elif stego_file.type.startswith('audio'):
            return decode_audio(stego_file, num_lsb)
        elif stego_file.type == 'text/plain':
            return decode_text(stego_file.getvalue().decode('utf-8'))
        else:
            raise ValueError("Unsupported file type for advanced decoding.")

def visualize_video(original_video_path, encoded_video_path):
    # Open the original video
    original_cap = cv2.VideoCapture(original_video_path)
    if not original_cap.isOpened():
        raise ValueError(f"Unable to open original video file for visualization: {original_video_path}")
    ret_orig, original_frame = original_cap.read()
    original_cap.release()
    if not ret_orig:
        raise ValueError("Unable to read frames from original video for visualization")
    
    # Open the encoded video
    encoded_cap = cv2.VideoCapture(encoded_video_path)
    if not encoded_cap.isOpened():
        raise ValueError(f"Unable to open encoded video file for visualization: {encoded_video_path}")
    ret_enc, encoded_frame = encoded_cap.read()
    encoded_cap.release()
    if not ret_enc:
        raise ValueError("Unable to read frames from encoded video for visualization")
    
    diff_frame = cv2.absdiff(original_frame, encoded_frame)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Frame')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(encoded_frame, cv2.COLOR_BGR2RGB))
    ax2.set_title('Encoded Frame')
    ax2.axis('off')
    
    ax3.imshow(cv2.cvtColor(diff_frame, cv2.COLOR_BGR2RGB))
    ax3.set_title('Difference')
    ax3.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def file_to_binary(file):
    return ''.join(format(byte, '08b') for byte in file.read())

def binary_to_file(binary, file_type):
    try:
        file_bytes = bytes(int(binary[i:i+8], 2) for i in range(0, len(binary), 8))
        return io.BytesIO(file_bytes)
    except Exception as e:
        raise ValueError(f"Error converting binary to file: {str(e)}")

def encode_lsb(data, binary_payload, num_lsb):
    payload_bits = np.array([int(bit) for bit in binary_payload], dtype=np.uint8)
    
    data_flat = data.flatten()
    bit_index = 0
    for i in range(data_flat.size):
        if bit_index < len(payload_bits):
            bits_to_encode = payload_bits[bit_index:bit_index+num_lsb]
            bits_string = ''.join(map(str, bits_to_encode)).ljust(num_lsb, '0')
            data_flat[i] = (data_flat[i] & ~((1 << num_lsb) - 1)) | int(bits_string, 2)
            bit_index += num_lsb
        else:
            break

    return data_flat.reshape(data.shape)

def decode_lsb(data, num_lsb):
    binary_payload = []
    data_flat = data.flatten()
    
    for i in range(data_flat.size):
        bits = [int(b) for b in f'{data_flat[i] & ((1 << num_lsb) - 1):0{num_lsb}b}']
        binary_payload.extend(bits)
        if len(binary_payload) >= 16 and binary_payload[-16:] == [1]*16 + [0]:
            binary_payload = binary_payload[:-16]  # Remove end marker
            break
    else:
        raise ValueError("No hidden payload found")
    
    return ''.join(map(str, binary_payload))

def encode_image(image_file, payload, num_lsb):
    img = Image.open(image_file).convert('RGB')
    data = np.array(img)

    # Process the payload
    if isinstance(payload, str):
        binary_payload = text_to_binary(payload) + '1111111111111110'  # End marker
    else:
        # Assuming payload is an image file
        img_payload = Image.open(payload).convert('RGB')
        img_array = np.array(img_payload)
        height, width, channels = img_array.shape

        # Encode the image dimensions in 16 bits each
        dimension_bits = (
            format(height, '016b') +
            format(width, '016b') +
            format(channels, '016b')
        )

        # Convert image data to binary
        image_bits = ''.join(format(byte, '08b') for byte in img_array.flatten())

        # Combine dimension bits and image bits and add end marker
        binary_payload = dimension_bits + image_bits + '1111111111111110'  # End marker

    # Encode using LSB
    encoded_data = encode_lsb(data, binary_payload, num_lsb)
    encoded_img = Image.fromarray(encoded_data.astype(np.uint8))
    return encoded_img

def decode_image(image_file, num_lsb):
    img = Image.open(image_file).convert('RGB')
    data = np.array(img)
    binary_payload = decode_lsb(data, num_lsb)

    # Check if image dimensions are included
    if len(binary_payload) >= 48:
        height_bits = binary_payload[:16]
        width_bits = binary_payload[16:32]
        channels_bits = binary_payload[32:48]
        height = int(height_bits, 2)
        width = int(width_bits, 2)
        channels = int(channels_bits, 2)
        image_dimensions = (height, width, channels)
        binary_payload = binary_payload[48:]  # Remove dimension bits from the payload
    else:
        image_dimensions = None

    return binary_payload, image_dimensions

def encode_audio(audio_file, payload, num_lsb):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
            temp_file.write(audio_file.getvalue())
            temp_file.flush()
        
        audio = AudioSegment.from_file(temp_filename)
        audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)  # Ensure 16-bit integer type
        
        # Process the payload
        if isinstance(payload, str):
            binary_payload = text_to_binary(payload) + '1111111111111110'  # End marker
        else:
            # Assuming payload is an image file
            img_payload = Image.open(payload).convert('RGB')
            img_array = np.array(img_payload)
            height, width, channels = img_array.shape

            # Encode the image dimensions in 16 bits each
            dimension_bits = (
                format(height, '016b') +
                format(width, '016b') +
                format(channels, '016b')
            )

            # Convert image data to binary
            image_bits = ''.join(format(byte, '08b') for byte in img_array.flatten())

            # Combine dimension bits and image bits and add end marker
            binary_payload = dimension_bits + image_bits + '1111111111111110'  # End marker

        # Encode using LSB
        encoded_data = encode_lsb(audio_data, binary_payload, num_lsb)
        
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
            
            binary_payload = decode_lsb(audio_data, num_lsb)

            # Check if image dimensions are included
            if len(binary_payload) >= 48:
                height_bits = binary_payload[:16]
                width_bits = binary_payload[16:32]
                channels_bits = binary_payload[32:48]
                height = int(height_bits, 2)
                width = int(width_bits, 2)
                channels = int(channels_bits, 2)
                image_dimensions = (height, width, channels)
                binary_payload = binary_payload[48:]  # Remove dimension bits from the payload
            else:
                image_dimensions = None

            return binary_payload, image_dimensions, audio_data
    except Exception as e:
        raise ValueError(f"Error processing audio file: {str(e)}")

def visualize_audio(original, encoded):
    plt.figure(figsize=(12, 8))
    
    # Plot original waveform
    plt.subplot(3, 1, 1)
    plt.title("Original Audio")
    plt.plot(original)
    plt.ylim(min(original), max(original))
    
    # Plot encoded waveform
    plt.subplot(3, 1, 2)
    plt.title("Encoded Audio")
    plt.plot(encoded)
    plt.ylim(min(encoded), max(encoded))
    
    # Plot difference
    plt.subplot(3, 1, 3)
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

def calculate_optimal_image_size(video_file, num_lsb):
    try:
        video_file.seek(0)  # Reset file pointer to the beginning
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_file_path = temp_file.name

        video = cv2.VideoCapture(temp_file_path)
        if not video.isOpened():
            raise ValueError("Unable to open video file")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        os.unlink(temp_file_path)

        if width == 0 or height == 0 or frame_count == 0:
            raise ValueError("Invalid video dimensions or frame count")

        bits_per_frame = 3 * width * height * num_lsb
        total_bits = max(0, (frame_count - 1) * bits_per_frame - 32)  # Ensure non-negative
        total_pixels = total_bits // (3 * 8)  # 3 channels, 8 bits per channel

        optimal_side = max(32, int(np.sqrt(total_pixels)))  # Ensure at least 32x32 pixels
        return optimal_side, optimal_side  # Return a default size in case of error
    except Exception as e:
        print(f"Error calculating optimal image size: {str(e)}")
        return 32, 32  # Return a default size in case of error