import numpy as np
from PIL import Image
import wave
import io
import cv2
import math
import os
import random
import re
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import matplotlib.pyplot as plt

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

# Convert encoding data into 8-bit binary ASCII
def generateData(data):
    newdata = []
    for i in data:  # List of binary codes of given data
        newdata.append(format(ord(i), '08b'))
    return newdata

# Modify the pixels according to the encoding data in generateData
def modifyPixel(pixel, data):
    datalist = generateData(data)
    lengthofdata = len(datalist)
    imagedata = iter(pixel)
    for i in range(lengthofdata):
        # Extracts 3 pixels at a time
        pixel = [value for value in imagedata.__next__()[:3] + imagedata.__next__()[:3] + imagedata.__next__()[:3]]
        # Pixel value should be made odd for 1 and even for 0
        for j in range(0, 8):
            if (datalist[i][j] == '0' and pixel[j] % 2 != 0):
                pixel[j] -= 1
            elif (datalist[i][j] == '1' and pixel[j] % 2 == 0):
                if pixel[j] != 0:
                    pixel[j] -= 1
                else:
                    pixel[j] += 1
        # Eighth pixel of every set tells whether to stop or read further. 0 means keep reading; 1 means the message is over.
        if i == lengthofdata - 1:
            if pixel[-1] % 2 == 0:
                if pixel[-1] != 0:
                    pixel[-1] -= 1
                else:
                    pixel[-1] += 1
        else:
            if pixel[-1] % 2 != 0:
                pixel[-1] -= 1
        pixel = tuple(pixel)
        yield pixel[0:3]
        yield pixel[3:6]
        yield pixel[6:9]

# Encode data in an image
def encoder(newimage, data):
    w = newimage.size[0]
    (x, y) = (0, 0)

    for pixel in modifyPixel(newimage.getdata(), data):
        # Putting modified pixels in the new image
        newimage.putpixel((x, y), pixel)
        if x == w - 1:
            x = 0
            y += 1
        else:
            x += 1

def encode_random(filename, frame_loc):
    # Load text to hide
    try:
        with open(filename, "r") as fileinput:
            print(filename)
            filedata = fileinput.read()
    except FileNotFoundError:
        print("\nFile to hide not found! Exiting...")
        quit()

    frame_count = len([f for f in os.listdir(frame_loc) if f.endswith(".png")])
    
    # Check if there are any frames available
    if frame_count == 0:
        print("No frames found in the directory! Exiting...")
        return
    
    total_frames = frame_count

    # Pick random start and end frames
    start = random.randint(0, total_frames - 1)
    end = random.randint(start + 1, total_frames)

    print(f"Performing steganography from frame {start} to {end}...")

    # Save the start and end frames to the metadata file
    metadata_path = os.path.join(frame_loc, "frame_metadata.txt")
    message_length = len(filedata)  # Get the length of the message
    with open(metadata_path, "w") as f:
        f.write(f"{start},{end},{message_length}\n")  # Store start, end, and message length in the metadata
        print(f"Metadata content: {f}")

    datapoints = math.ceil(len(filedata) / (end - start))  # Distribute data across selected frames

    counter = start
    for convnum in range(0, len(filedata), datapoints):
        numbering = os.path.join(frame_loc, f"{counter}.png")
        encodetext = filedata[convnum:convnum + datapoints]
        try:
            image = Image.open(numbering, 'r')
        except FileNotFoundError:
            print(f"\n{counter}.png not found! Exiting...")
            quit()

        newimage = image.copy()  # New variable to store hidden data
        encoder(newimage, encodetext)  # Perform steganography on the frame
        newimage.save(numbering)  # Save the modified frame
        counter += 1

    print("Complete!\n")

# # Extract frames from the video
# def get_frames(video_path, output_folder):
#     video_clip = VideoFileClip(video_path)  # Load the video as a VideoFileClip
#     directory = os.path.join(output_folder, 'frames')
    
#     if not os.path.isdir(directory):
#         os.makedirs(directory)
    
#     for index, frame in enumerate(video_clip.iter_frames()):
#         img = Image.fromarray(frame, 'RGB')
#         img.save(f'{directory}/{index}.png')
#         #print(directory)
    
#     print(f"Extracted {index + 1} frames")

# Extract frames from the video
def get_frames(video_path, output_folder):
    video_clip = VideoFileClip(video_path)  # Load the video as a VideoFileClip

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for index, frame in enumerate(video_clip.iter_frames()):
        img = Image.fromarray(frame, 'RGB')
        img.save(f'{output_folder}/{index}.png')  # Save frames directly in output_folder
    
    print(f"Extracted {index + 1} frames")


# Reassemble frames into video
def frames_to_video(frame_folder, output_video_path, fps):
    images = [img for img in sorted(os.listdir(frame_folder)) if img.endswith(".png") and img.split('.')[0].isdigit()]
    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(frame_folder, image))
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")


# Decode the data in the image
def decode_frame(number, frame_location):
    data = ''
    frame_file = os.path.join(frame_location, f"{number}.png")  # Dynamically handle frame path

    try:
        image = Image.open(frame_file, 'r')
    except FileNotFoundError:
        print(f"Frame {number}.png not found! Exiting...")
        return data

    imagedata = iter(image.getdata())
    
    while True:
        pixels = [value for value in imagedata.__next__()[:3] + imagedata.__next__()[:3] + imagedata.__next__()[:3]]
        # Extract binary string from the pixel's LSB
        binstr = ''.join(['0' if (i % 2 == 0) else '1' for i in pixels[:8]])

        # Check if the binary data is printable ASCII
        decoded_char = chr(int(binstr, 2))
        if re.match("[ -~]", decoded_char):  # Only printable ASCII characters
            data += decoded_char
        # Check if the message is over (stop condition)
        if pixels[-1] % 2 != 0:  # End of message if last bit is odd
            return data

# # Decoding text hidden in the video frames using the metadata (start and end frames)
# def decode_hidden_text(frame_location, output_file):
#     # Read the start and end frames from the metadata file
#     metadata_path = os.path.join(frame_location, "frame_metadata.txt")
    
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

#     # Ensure the metadata file contains valid frame numbers (integers)
#     with open(metadata_path, "r") as f:
#         content = f.read().strip()  # Read the content and strip any extra whitespace or newline characters
#         print(f"Metadata content: {content}")
#         try:
#             start, end, message_length = map(int, content.split(','))  # Convert the frame numbers and message length to integers
#         except ValueError:
#             raise ValueError(f"Invalid data in metadata file: {content}")
    
#     print(f"Extracting data from frames {start} to {end}, expecting {message_length} characters...")

#     # Open the output file where decoded data will be stored
#     with open(output_file, 'w') as decoded_text_file:
#         decoded_text_file.write('Decoded Text:\n')
#         total_decoded_text = ""

#         for frame_num in range(start, end + 1):
#             try:
#                 decoded_text = decode_frame(frame_num, frame_location)  # Decode each frame
#                 if decoded_text:
#                     total_decoded_text += decoded_text
#                     print(f"Data found in Frame {frame_num}")
#             except StopIteration:
#                 print(f"No data found in Frame {frame_num}")
#         # Limit the decoded text to the first `message_length` characters
#         decoded_text_file.write(total_decoded_text[:message_length])
#         print(f"Decoded message: {total_decoded_text[:message_length]}")
    
#     print("\nExtraction Complete!")
    
# Decoding text hidden in the video frames using the metadata (start and end frames)
def decode_hidden_text(frame_location, output_file):
    # Read the start and end frames from the metadata file
    metadata_path = os.path.join(frame_location, "frame_metadata.txt")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Ensure the metadata file contains valid frame numbers and message length
    with open(metadata_path, "r") as f:
        content = f.read().strip()
        print(f"Metadata content: {content}")
        try:
            start, end, message_length = map(int, content.split(','))  # Convert frame numbers and message length
        except ValueError:
            raise ValueError(f"Invalid data in metadata file: {content}")
    
    print(f"Extracting data from frames {start} to {end}, expecting {message_length} characters...")

    # Open the output file where decoded data will be stored
    with open(output_file, 'w') as decoded_text_file:
        decoded_text_file.write('Decoded Text:\n')
        total_decoded_text = ""

        for frame_num in range(start, end + 1):
            try:
                decoded_text = decode_frame(frame_num, frame_location)  # Decode each frame
                if decoded_text:
                    total_decoded_text += decoded_text
                    print(f"Data found in Frame {frame_num}")
            except StopIteration:
                print(f"No data found in Frame {frame_num}")

        # Limit the decoded text to the first `message_length` characters
        total_decoded_text = total_decoded_text[:message_length]  # Trim to message length
        decoded_text_file.write(total_decoded_text)
        print(f"Decoded message: {total_decoded_text}")
    
    print("\nExtraction Complete!")


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
    
END_MARKER = ''.join(['1' if i % 2 == 0 else '0' for i in range(64)])  # Alternating 1s and 0s

def encode_lsb(data, payload, num_lsb):
    binary_payload = payload + END_MARKER
    payload_bits = np.array([int(bit) for bit in binary_payload], dtype=np.uint8)
    data_flat = data.flatten()
    
    if len(payload_bits) > data_flat.size * num_lsb:
        raise ValueError("Payload too large for cover image")
    
    bit_index = 0
    for i in range(data_flat.size):
        if bit_index >= len(payload_bits):
            break
        
        bits_to_encode = payload_bits[bit_index:bit_index+num_lsb]
        if len(bits_to_encode) < num_lsb:
            bits_to_encode = np.pad(bits_to_encode, (0, num_lsb - len(bits_to_encode)))
        
        mask = 256 - (1 << num_lsb)
        data_flat[i] = (data_flat[i] & mask) | int(''.join(map(str, bits_to_encode)), 2)
        
        bit_index += num_lsb
    
    return data_flat.reshape(data.shape)

def decode_lsb(data, num_lsb):
    binary_payload = []
    data_flat = data.flatten()
    
    for value in data_flat:
        bits = [int(b) for b in format(value & ((1 << num_lsb) - 1), f'0{num_lsb}b')]
        binary_payload.extend(bits)
        
        if len(binary_payload) >= len(END_MARKER):
            end_sequence = ''.join(map(str, binary_payload[-len(END_MARKER):]))
            if end_sequence == END_MARKER:
                return ''.join(map(str, binary_payload[:-len(END_MARKER)]))
            
            # Check for partial end marker
            if all(end_sequence[i] == END_MARKER[i] for i in range(len(end_sequence))):
                continue
            
            # Check for shifted end marker
            for shift in range(1, num_lsb):
                shifted_end = ''.join(map(str, binary_payload[-len(END_MARKER)-shift:-shift]))
                if shifted_end == END_MARKER:
                    return ''.join(map(str, binary_payload[:-len(END_MARKER)-shift]))
    
    raise ValueError("No hidden payload found")

def encode_image(image_file, payload, num_lsb):
    img = Image.open(image_file).convert('RGB')
    data = np.array(img)
    encoded_data = encode_lsb(data, payload, num_lsb)
    encoded_img = Image.fromarray(encoded_data.astype(np.uint8))
    return encoded_img

def decode_image(image_file, num_lsb):
    img = Image.open(image_file).convert('RGB')
    data = np.array(img)
    return decode_lsb(data, num_lsb)


# Helper function to convert image to binary string
def image_to_binary(image_file):
    img = Image.open(image_file)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return ''.join(format(byte, '08b') for byte in bio.getvalue())

# Helper function to convert binary string back to image
def binary_to_image(binary_string):
    binary_data = int(binary_string, 2).to_bytes((len(binary_string) + 7) // 8, byteorder='big')
    return Image.open(io.BytesIO(binary_data))


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

