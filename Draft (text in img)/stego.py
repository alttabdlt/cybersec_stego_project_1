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
            filedata = fileinput.read()
    except FileNotFoundError:
        print("\nFile to hide not found! Exiting...")
        quit()

    frame_count = len([f for f in os.listdir(frame_loc) if f.endswith(".png")])
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

# Extract frames from the video
def get_frames(video_path, output_folder):
    video_clip = VideoFileClip(video_path)  # Load the video as a VideoFileClip
    directory = os.path.join(output_folder, 'frames')
    
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    for index, frame in enumerate(video_clip.iter_frames()):
        img = Image.fromarray(frame, 'RGB')
        img.save(f'{directory}/{index}.png')
    
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

# Decoding text hidden in the video frames using the metadata (start and end frames)
def decode_hidden_text(frame_location, output_file):
    # Read the start and end frames from the metadata file
    metadata_path = os.path.join(frame_location, "frame_metadata.txt")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Ensure the metadata file contains valid frame numbers (integers)
    with open(metadata_path, "r") as f:
        content = f.read().strip()  # Read the content and strip any extra whitespace or newline characters
        print(f"Metadata content: {content}")
        try:
            start, end, message_length = map(int, content.split(','))  # Convert the frame numbers and message length to integers
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
        decoded_text_file.write(total_decoded_text[:message_length])
        print(f"Decoded message: {total_decoded_text[:message_length]}")
    
    print("\nExtraction Complete!")

