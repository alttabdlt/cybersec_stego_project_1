import streamlit as st
import os
from PIL import Image
import io
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from steganography import encode_image, decode_image, encode_audio, decode_audio, can_encode, text_to_binary, binary_to_text, file_to_binary, binary_to_file, visualize_audio, visualize_image, advanced_decode, encode_video, decode_video, encode_text, decode_text, visualize_video, image_to_binary, calculate_optimal_image_size
import tempfile
import struct

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return True, result.stdout.split('\n')[0]
        else:
            return False, "FFmpeg is installed but there was an error running it."
    except FileNotFoundError:
        return False, "FFmpeg is not found in the system PATH."

def main():
    st.set_page_config(page_title="Steganography Tool", layout="wide")
    st.title("Steganography Tool")

    # Check FFmpeg installation
    ffmpeg_installed, ffmpeg_message = check_ffmpeg()
    if ffmpeg_installed:
        st.success(f"FFmpeg is correctly installed: {ffmpeg_message}")
    else:
        st.error(f"FFmpeg is not correctly installed or not in PATH: {ffmpeg_message}")
        st.error("Please install FFmpeg and make sure it's in your system PATH to use audio steganography features.")

    operation = st.sidebar.selectbox("Choose operation", ["Encode", "Decode"])

    if operation == "Encode":
        encode_ui()
    else:
        decode_ui()

def encode_ui():
    st.header("Encode a payload into a cover")

    st.subheader("Step 1: Select Cover")
    cover_type = st.radio("Choose cover type", ["Image", "Audio", "Video", "Text"])
    cover_file = st.file_uploader(
        "Upload cover file",
        type=["png", "jpg", "bmp", "wav", "mp3", "mp4", "avi"] if cover_type != "Text" else None
    )

    if cover_type == "Text":
        cover_text = st.text_area("Enter cover text")

    st.subheader("Step 2: Encoding Settings")
    num_lsb = st.slider("Number of LSBs", min_value=1, max_value=8, value=2)
    st.info(f"Number of LSBs used for encoding is set to {num_lsb}")

    st.subheader("Step 3: Select Payload")
    payload_options = ["Text", "Image"] if cover_type == "Video" else ["Text", "Image", "Audio", "Document", "Video"]
    payload_type = st.selectbox("Choose payload type", payload_options)

    payload_bytes = None
    payload_file = None  # Initialize payload_file
    if payload_type == "Text":
        payload_text = st.text_area("Enter payload text")
        if payload_text:
            payload_bytes = payload_text.encode('utf-8')
    elif payload_type == "Image":
        payload_file = st.file_uploader("Upload payload image", type=["png", "jpg", "bmp"])
        if payload_file:
            img = Image.open(payload_file)
            img_array = np.array(img)
            height, width, channels = img_array.shape

            # Pack the image dimensions into bytes
            dimension_bytes = struct.pack('>HHH', height, width, channels)

            # Convert image data to bytes
            image_bytes = img_array.tobytes()

            # Combine dimension bytes and image bytes
            payload_bytes = dimension_bytes + image_bytes
    else:
        st.error("Unsupported payload type for the selected cover type.")

    if st.button("Encode"):
        try:
            if cover_file is not None or cover_type == "Text":
                if cover_type == "Video":
                    if payload_bytes is not None:
                        # Check if the payload can be encoded
                        can_encode_result, available_bits, required_bits = can_encode(cover_file, payload_file or payload_text, num_lsb)
                        if not can_encode_result:
                            st.error(f"The payload is too large to encode with the selected number of LSBs.\nAvailable bits: {available_bits}\nRequired bits: {required_bits}")
                            return
                        output_video_path = encode_video(cover_file, payload_bytes, num_lsb)
                        st.success("Payload encoded successfully into the cover video!")

                        # Display the encoded video
                        with open(output_video_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.download_button(
                            label="Download Stego Video",
                            data=video_bytes,
                            file_name="stego_video.mp4",
                            mime="video/mp4"
                        )

                        # Clean up temporary files
                        os.remove(output_video_path)
                    else:
                        st.error("Please provide a payload to encode.")
                else:
                    # Handle other cover types if necessary
                    st.error("Currently, payload encoding into video cover files is only supported for the 'Video' cover type with 'Text' or 'Image' payloads.")
            else:
                st.error("Please upload a cover file.")

        except Exception as e:
            st.error(f"An error occurred during encoding: {str(e)}")
            st.error("Please make sure you have selected the correct number of LSBs and payload type.")

def decode_ui():
    st.header("Decode payload from stego object")

    st.subheader("Step 1: Upload Stego Object")
    stego_file = st.file_uploader(
        "Choose a stego file to decode",
        type=["png", "jpg", "bmp", "wav", "mp3", "mp4", "avi", "txt"]
    )

    st.subheader("Step 2: Decoding Settings")
    num_lsb = st.slider("Number of LSBs used in encoding", min_value=1, max_value=8, value=2)
    st.info(f"Number of LSBs used for decoding is set to {num_lsb}")

    st.subheader("Step 3: Specify Expected Payload Type")
    payload_type = st.selectbox("Choose expected payload type", ["Text", "Image"])

    if stego_file:
        if st.button("Decode Payload"):
            try:
                decoded_payload = advanced_decode(stego_file, num_lsb, payload_type)
                st.success("Payload decoded successfully from the stego object!")

                if payload_type == "Text" and isinstance(decoded_payload, str):
                    st.text_area("Decoded text payload", decoded_payload, height=150)
                elif payload_type == "Image" and isinstance(decoded_payload, Image.Image):
                    st.image(decoded_payload, caption="Decoded Image Payload")
                else:
                    st.error("Decoded payload does not match the expected type.")

            except Exception as e:
                st.error(f"An error occurred during decoding: {str(e)}")
                st.error("Please make sure you have selected the correct number of LSBs and payload type.")

if __name__ == "__main__":
    main()