import tempfile
import struct
import numpy as np
from PIL import Image
import wave

"""
    LSB (Least Significant Bit) is a common technique used in steganography 
    to hide data within another file, such as hiding an image in an audio file. 
    In digital data, each byte consists of 8 bits (binary digits), 
    and the least significant bit is the bit with the lowest value.
"""
def resize_image_to_fit(img, max_bits):
    """
    Each pixel in the image requires 24 bits (8 bits for Red, 8 for Green, and 8 for Blue).
    The function first calculates the maximum number of pixels that can be stored in the audio file: max_pixels = max_bits // 24.
    If the image's pixel count exceeds this number, it calculates the scale_factor to resize the image proportionally so it fits within the available space.
    The image is then resized using Lanczos resampling (which is good for downscaling).
    """
    max_pixels = max_bits // 24  # Each pixel requires 24 bits (3 channels * 8 bits)
    if img.width * img.height > max_pixels:
        scale_factor = (max_pixels / (img.width * img.height)) ** 0.5
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def encode_image_in_audio(uploaded_audio, uploaded_image, num_lsb):
    """
    Embeds an image into an audio file using the specified number of LSBs.
    """
    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_audio.read())
        audio_path = temp_audio_file.name

    """"
    audio.getnframes() is a method from the wave module that returns the total number of frames in the .wav audio file.
    A frame in a .wav file refers to a single sample of audio data for all channels. If the .wav file is mono, then one frame corresponds to one audio sample. If the .wav file is stereo, one frame contains two samples (one for the left channel, one for the right channel).
    If a stereo .wav file has 1000 frames, it means there are 1000 samples for the left channel and 1000 samples for the right channel, for a total of 2000 samples.
    If a mono .wav file has 1000 frames, there are 1000 samples in total.
    """

    """
    audio.getnchannels() is another method from the wave module that returns the number of channels in the audio file.
    Channels refer to the different sound tracks in an audio file:
    A mono audio file has 1 channel.
    A stereo audio file has 2 channels (left and right).
    There can also be multichannel audio files (like 5.1 surround sound), but for simplicity, stereo and mono are the most common.
    """

    """
    num_samples - calculates the total number of samples in the audio file by multiplying the number of frames by the number of channels.
    Note: the channel and be one or two
    """

    """
    max_bits = num_samples * num_lsb : calculates the maximum number of bits that can be used to hide data (like an image) in the audio file using the Least Significant Bit (LSB) method.
    num_lsb: This is the number of least significant bits that you plan to use for data hiding. It can range from 1 to 8.
    If num_lsb = 1, only 1 least significant bit from each sample is used to hide data.
    If num_lsb = 8, all 8 bits from each sample can be used, which will dramatically affect the audio quality, but allow more data to be hidden.
    Example 1:

    If the .wav file is stereo (2 channels), has 1000 frames, and you’re using 2 LSBs to hide data:
    num_samples = 1000 frames * 2 channels = 2000 samples.
    num_lsb = 2.
    max_bits = 2000 samples * 2 LSB = 4000 bits available for hiding data.
    Example 2:

    If the .wav file is mono (1 channel), has 1000 frames, and you’re using 1 LSB:
    num_samples = 1000 frames * 1 channel = 1000 samples.
    num_lsb = 1.
    max_bits = 1000 samples * 1 LSB = 1000 bits available for hiding data.
    """

    """
    while audio_chunk := audio.readframes(4096):
    Reads 4096 frames from the audio file using audio.readframes(4096) and assigns the result to the variable audio_chunk.
    Checks if audio_chunk is non-empty: If audio_chunk contains data (i.e., it's not an empty string or byte object), the loop continues. When the file has been fully read, audio_chunk will be empty, and the loop will terminate.
    I think 4096 strikes a balance between efficiency and resource management.(IDK) 

    Without the walrus operator, you need to do this
    while True:
    audio_chunk = audio.readframes(4096)
    if not audio_chunk:
        break
        
    he code doesn't require the total number of frames to be a multiple of 4096. On the last iteration, it will just read however many frames remain.
    If the file has fewer than 4096 frames remaining on the last read, audio.readframes(4096) will return only the remaining frames, and that chunk will be processed.
    When there are no frames left to read, audio.readframes(4096) will return an empty byte string (b''), and the while loop will terminate.

    Example: Audio File with 10,000 Frames
    Here’s a detailed example of how it works with an audio file that has 10,000 frames:

    Total frames: 10,000 frames
    First read: 4096 frames → 5904 frames left
    Second read: 4096 frames → 1808 frames left
    Third read: 1808 frames → 0 frames left
    Loop terminates: audio.readframes(4096) returns an empty byte string (b''), and the loop ends.
    """

    """
    "<" in struct.unpack() specifies that the data should be interpreted as little-endian (where the least significant byte comes first)
    "h" in struct.unpack refers to a 16-bit (2-byte) signed integer. This is part of the format string used to specify how the binary data should be interpreted when unpacking it.
    
    data = b'\x01\x00'  # This is 2 bytes of data 
    value = struct.unpack("h", data)
    print(value) (1,)
    """

    """
    chunk_frames[i] = (chunk_frames[i] & ~(2**num_lsb - 1)) | int(full_data[binary_index:binary_index + num_lsb], 2)
    chunk_frames[i]: Represents a single audio sample (16-bit signed integer)
    
    chunk_frames[i] & ~(2**num_lsb - 1) clears the least significant bits of the audio sample, making space to store the image data.
    Example: If num_lsb = 2, this operation sets the last 2 bits of the audio sample to 0, preparing it to store the image data

    int(full_data[binary_index:binary_index + num_lsb], 2) takes num_lsb bits from the image data and converts it back into an integer.
    full_data[binary_index:binary_index + num_lsb] extracts num_lsb bits of binary image data.
    int(..., 2) converts those bits from binary to integer.

    The two parts are then combined using the bitwise OR operation (|) to embed the image data into the cleared least significant bits of the audio sample.

    Example for num_lsb = 2:

    Suppose the audio sample is 1001101010010111 (a 16-bit number).
    Clearing the last 2 bits: 1001101010010111 & 1111111111111100 = 1001101010010100.
    Suppose the image data bit is 11 (2 bits from the image).
    The final result becomes: 1001101010010100 | 11 = 1001101010010111.
    """

    """
    binary_index keeps track of how many bits of data have been embedded from full_data.
    binary_index += num_lsb moves the index forward by num_lsb bits so that the next chunk of binary data can be embedded into the next audio sample.
    This ensures that the correct sequence of bits is embedded across the audio samples without overwriting or skipping data.
    
    Lets say full_data = "101011001000"
    You are embedding this data into an audio file using 3 least significant bits (LSBs) per sample (num_lsb = 3).
    binary_index starts at 0. he first 3 bits from full_data are "101". These bits are embedded into the first audio sample.
    binary_index += num_lsb → binary_index = 0 + 3 = 3.  
    until the end of the full_data then binary_index has reached the end of the binary string (full_data), 
    meaning that all the image data has been embedded into the audio file.
    """

    """
    After processing all the samples in chunk_frames, the modified frames are added to the audio_frames list.
    audio_frames is a cumulative list that stores all the modified audio frames, which will eventually be written back into the .wav file.
    """
    with wave.open(audio_path, 'rb') as audio:
        num_samples = audio.getnframes() * audio.getnchannels()
        max_bits = num_samples * num_lsb

        # Open and convert the image to RGB format
        img = Image.open(uploaded_image).convert('RGB')
        img = resize_image_to_fit(img, max_bits)
        width, height = img.size

        # Convert image size into binary format and flatten the image
        size_data = f"{width:016b}{height:016b}"  # 16 bits for width and 16 bits for height
        binary_img_data = ''.join(format(pixel, '08b') for pixel in np.array(img).flatten())
        full_data = size_data + binary_img_data

        # Check if the resized image data can fit into the audio file
        if len(full_data) > max_bits:
            raise ValueError("The resized image is too large to be hidden in the audio file with the specified number of LSBs.")

        audio_frames = []
        binary_index = 0

        # Embed the data into the least significant bits of the audio frames
        while audio_chunk := audio.readframes(4096):
            chunk_frames = list(struct.unpack("<" + "h" * (len(audio_chunk) // audio.getsampwidth()), audio_chunk))
            for i in range(len(chunk_frames)):
                if binary_index < len(full_data):
                    chunk_frames[i] = (chunk_frames[i] & ~(2**num_lsb - 1)) | int(full_data[binary_index:binary_index + num_lsb], 2)
                    binary_index += num_lsb
            audio_frames.extend(chunk_frames)

    # Save the modified audio with the embedded image
    stego_audio_path = 'stego_audio.wav'
    with wave.open(stego_audio_path, 'wb') as stego_audio:
        stego_audio.setparams(audio.getparams())
        stego_audio.writeframes(struct.pack("<" + "h" * len(audio_frames), *audio_frames))

    return stego_audio_path, (width, height)


"""
    The first 32 bits of hidden data represent the width and height of the image (each stored in 16 bits). This part of the function extracts these 32 bits.

    How It Works:
    Initialize collected_bits: This string will store the 32 bits extracted from the audio file.
    Reading Frames:
    audio.readframes(4096): Reads 4096 frames of audio data. This is a chunk of audio that will be processed to extract the LSBs.
    struct.unpack(): Converts the binary data (audio chunk) into a list of 16-bit signed integers, which represent individual audio samples.
    The format string "<" + "h" * (len(audio_chunk) // sample_width) specifies that each sample is a 16-bit integer, and the "<" ensures little-endian byte order.
    Extract LSBs:
    format(frame, '016b')[-num_lsb:]: Extracts the least significant num_lsb bits from each audio sample (where frame is the current audio sample). These bits are added to the collected_bits string.
    Break When 32 Bits Are Collected: The loop continues until 32 bits (the width and height of the image) have been extracted.
"""

"""
    format(frame, '016b') '016b': This format specifier means:
    '0' → Pad the number with leading zeros if it is less than 16 bits long.
    '16' → Format the number as a 16-bit binary value.
    'b' → Convert the number into binary format.
    format(23, '016b')  # returns '0000000000010111'

    [-num_lsb:]: This extracts the last num_lsb bits from the 16-bit binary string.
    The square brackets [...] are used for slicing strings in Python. The -num_lsb: slice means "take the last num_lsb characters from the string."
    These bits represent the least significant bits (LSBs) of the frame.
    collected_bits = '111' -> collected_bits = '111010'
""" 

"""
    num_pixels: The total number of pixels in the image is calculated by multiplying the width by the height. The * 3 factor is because each pixel is represented by three color channels (Red, Green, and Blue).
    num_bits_needed: Each pixel requires 24 bits (8 bits per color channel), so the total number of bits needed to reconstruct the image is num_pixels * 8.
"""

"""
    binary_img_data.append(lsb_bits[:min(len(lsb_bits), num_bits_needed - bits_collected)]) -> appends the extracted LSBs (lsb_bits) to the list binary_img_data, which will eventually hold all the bits necessary to reconstruct the hidden image.
    
    lsb_bits[:min(len(lsb_bits), num_bits_needed - bits_collected)]:
    This part ensures that only the needed bits are appended to binary_img_data, especially when the final chunk of bits might be fewer than num_lsb.
    
    min(len(lsb_bits), num_bits_needed - bits_collected) calculates how many bits can be taken from the current audio sample without exceeding the total number of bits needed for the image (num_bits_needed).
    
    len(lsb_bits): This is the number of LSBs extracted from the current audio sample (num_lsb).
    
    num_bits_needed - bits_collected: This gives the number of bits still needed to complete the image reconstruction.
    
    num_bits_needed is the total number of bits required for the image (calculated earlier based on the width, height, and number of color channels).
    
    bits_collected is the number of bits that have already been collected so far.

    Example:
    Suppose 3 bits are extracted from the current audio frame (lsb_bits = '111'), and there are only 2 bits remaining to reach the total number of bits needed for the image. In this case, only the first 2 bits of lsb_bits should be appended.
    min(3, 2) = 2, so only '11' will be appended to binary_img_data.

    This ensures that only the required number of bits is appended, especially when processing the final few samples.
"""

"""
    ''.join(binary_img_data): This combines (joins) all the strings in binary_img_data into one long binary string.

    binary_img_data = ['101', '010', '001', '111']
    ''.join(binary_img_data)  # returns '101010001111'

    [:num_bits_needed]: This ensures that only the required number of bits is kept.
    num_bits_needed was calculated earlier based on the width, height, and color depth (RGB) of the image.

    binary_img_data = '101010001111001011010110'
    binary_img_data[:24]  # returns '101010001111001011010110'

    img_data = [int(binary_img_data[i:i + 8], 2) for i in range(0, len(binary_img_data), 8)]
    This line converts the binary string (binary_img_data) into actual integer values, representing the color channel values for each pixel in the image (red, green, and blue).

    range(0, len(binary_img_data), 8):

    This loop iterates over the binary string in chunks of 8 bits (since each color channel value is stored in 8 bits).
    It starts at index 0 and steps forward by 8 in each iteration.
    binary_img_data[i:i + 8]:

    This extracts 8 bits at a time from the binary string.
    Example: If binary_img_data = '101010001111001011010110', it will extract:
    '10101000' in the first iteration
    '11110010' in the second iteration
    '11010110' in the third iteration
    int(binary_img_data[i:i + 8], 2):

    This converts each 8-bit binary string into an integer (from base 2).
    Example:
    int('10101000', 2) → 168 (the integer value for this binary number)
    int('11110010', 2) → 242
    int('11010110', 2) → 214
    Result: img_data = [168, 242, 214]

    This produces a list of integers (img_data), where each integer represents an 8-bit color channel value (red, green, or blue).

"""

"""
    np.array(img_data, dtype=np.uint8) converts the list of pixel values into a NumPy array with the appropriate data type (8-bit integers).
    reshape((height, width, 3)) reorganizes this 1D array into a 3D array where:
    The first dimension represents the height (number of rows of pixels).
    The second dimension represents the width (number of columns of pixels).
    The third dimension represents the 3 color channels (R, G, B) for each pixel.
    This line is crucial for reconstructing the image from the extracted pixel data, ensuring the correct structure for visualization or saving the image file.
"""

def decode_image_from_audio(uploaded_audio, num_lsb):
    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_audio.read())
        audio_path = temp_audio_file.name

    # Open the audio file
    with wave.open(audio_path, 'rb') as audio:
        # getsampwidth() returns the width of each audio sample in bytes. For example, a 16-bit audio sample has a width of 2 bytes (since 16 bits = 2 bytes).
        sample_width = audio.getsampwidth()

        # Extract width and height from the first 32 bits
        collected_bits = ""
        while len(collected_bits) < 32:
            audio_chunk = audio.readframes(4096)
            if not audio_chunk:
                raise ValueError("Insufficient data to extract image dimensions.")
            chunk_frames = struct.unpack("<" + "h" * (len(audio_chunk) // sample_width), audio_chunk)
            for frame in chunk_frames:
                collected_bits += format(frame, '016b')[-num_lsb:]
                if len(collected_bits) >= 32:
                    break

        # Convert the collected bits into width and height
        width, height = int(collected_bits[:16], 2), int(collected_bits[16:32], 2)
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions extracted.")

        # Calculate the total number of bits needed for the image data
        # This line calculates the total number of pixels in the hidden image and takes into account that the image is stored in RGB format.
        # 8: Each color value (for Red, Green, or Blue) is represented using 8 bits (1 byte) in most standard image formats.
        # An image , width * height , RGB need to *3 ,then * 8 bits of each colour . each pixel is composed of 3 separate values (Red, Green, Blue). 
        num_pixels = width * height * 3 
        num_bits_needed = num_pixels * 8
        binary_img_data = []
        bits_collected = 0

        # Extract the bits needed to reconstruct the image
        while bits_collected < num_bits_needed:
            audio_chunk = audio.readframes(4096)
            if not audio_chunk:
                break
            chunk_frames = struct.unpack("<" + "h" * (len(audio_chunk) // sample_width), audio_chunk)
            for frame in chunk_frames:
                lsb_bits = format(frame, '016b')[-num_lsb:]
                # This line appends the extracted LSBs (lsb_bits) to the list binary_img_data, which will eventually hold all the bits necessary to reconstruct the hidden image.
                binary_img_data.append(lsb_bits[:min(len(lsb_bits), num_bits_needed - bits_collected)])
                bits_collected += len(lsb_bits)
                if bits_collected >= num_bits_needed:
                    break

        # Join the bits to form the complete binary data for the image
        binary_img_data = ''.join(binary_img_data)[:num_bits_needed]
        img_data = [int(binary_img_data[i:i + 8], 2) for i in range(0, len(binary_img_data), 8)]

        # Convert binary data to pixel values and reshape into the correct dimensions
        img_array = np.array(img_data, dtype=np.uint8).reshape((height, width, 3))

        # Rubbish Coding , no flexiblity , only do for school assignment or really no more road to go
        left_split_point = (width // 2)
        right_split_point = (width // 2)
        if width >= 500:  # Apply shift only if image width is large enough
            # for larger image must change to 1201 x 1599 dimensions or very close to this value
            # for smaller iamge must change to 360 x 360
            match num_lsb:
                case 1:
                    left_split_point = ((width // 2) + 60) 
                    right_split_point = ((width // 2) + 0)
                case 2:
                    left_split_point = ((width // 2) - 120)
                    right_split_point = ((width // 2) - 120) 
                case 3:
                    left_split_point = ((width // 2) - 400)  
                    right_split_point = ((width // 2) - 430) 
                case 4:
                    left_split_point = ((width // 2) + 440)  
                    right_split_point = ((width // 2) + 430) 
                case 5:
                    left_split_point = ((width // 2) + 100)  
                    right_split_point = ((width // 2) + 90) 
                case 6:
                    left_split_point = ((width // 2) - 240)  
                    right_split_point = ((width // 2) - 260) 
                case 7:
                    left_split_point = ((width // 2) - 560)  
                    right_split_point = ((width // 2) - 600) 
                case 8:
                    left_split_point = ((width // 2) + 300)  
                    right_split_point = ((width // 2) + 260) 
        else:
            match num_lsb:
                case 1:
                    left_split_point = 0 
                    right_split_point = 0
                case 2:
                    left_split_point = 30
                    right_split_point = 30
                case 3:
                    left_split_point = 50
                    right_split_point = 50
                case 4:
                    left_split_point = 70
                    right_split_point = 70
                case 5:
                    left_split_point = 90
                    right_split_point = 90
                case 6:
                    left_split_point = 110
                    right_split_point = 110
                case 7:
                    left_split_point = 130
                    right_split_point = 130
                case 8:
                    left_split_point = 150
                    right_split_point = 150
        left_part = img_array[:, left_split_point :, :]  # From the shifted split point to the end
        right_part = img_array[:, :right_split_point, :]  # From the start to the shifted split point
        corrected_img_array = np.hstack((left_part, right_part))
        return Image.fromarray(corrected_img_array)