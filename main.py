import streamlit as st
import os
from PIL import Image
import io
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from steganography import encode_image, decode_image, encode_audio, decode_audio, can_encode, text_to_binary, binary_to_text, file_to_binary, binary_to_file, visualize_audio, visualize_image, advanced_decode, encode_video, decode_video, encode_text, decode_text, visualize_video

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
    cover_file = st.file_uploader("Upload cover file", type=["png", "jpg", "bmp", "wav", "mp3", "mp4", "txt"] if cover_type != "Text" else None)
    
    if cover_type == "Text":
        cover_text = st.text_area("Enter cover text")

    st.subheader("Step 2: Select Payload")
    payload_type = st.selectbox("Choose payload type", ["Text", "Image", "Audio", "Document", "Video"])
    
    binary_payload = None
    if payload_type == "Text":
        payload = st.text_area("Enter text payload")
        if payload:
            binary_payload = text_to_binary(payload)
    elif payload_type in ["Image", "Audio", "Document", "Video"]:
        file_types = {
            "Image": ["png", "jpg", "bmp"],
            "Audio": ["wav"],
            "Document": ["txt", "pdf", "docx"],
            "Video": ["mp4"]
        }
        payload_file = st.file_uploader(f"Choose {payload_type.lower()} payload", type=file_types[payload_type])
        if payload_file:
            binary_payload = file_to_binary(payload_file)
    
    st.subheader("Step 3: Encoding Settings")
    if cover_type == "Audio":
        num_lsb = st.slider("Number of LSBs", min_value=1, max_value=2, value=1)
    else:
        num_lsb = st.slider("Number of LSBs", min_value=1, max_value=8, value=1)

    if (cover_file or (cover_type == "Text" and cover_text)) and binary_payload:
        try:
            if cover_type == "Text":
                encoded_obj = encode_text(cover_text, binary_payload)
                st.success("Payload encoded successfully into the cover text!")
                st.text_area("Original text", cover_text, height=150)
                st.text_area("Encoded text", encoded_obj, height=150)
            else:
                can_encode_result, available_bits, required_bits = can_encode(cover_file, binary_payload, num_lsb)
                st.info(f"Available capacity in cover: {available_bits} bits")
                st.info(f"Required capacity for payload: {required_bits} bits")
                
                if can_encode_result:
                    if st.button("Encode Payload into Cover"):
                        try:
                            progress_bar = st.progress(0)
                            if cover_file.type.startswith('image'):
                                original_img = Image.open(cover_file)
                                encoded_obj = encode_image(cover_file, binary_payload, num_lsb)
                                progress_bar.progress(100)
                                st.success("Payload encoded successfully into the cover image!")
                                
                                visualization = visualize_image(original_img, encoded_obj)
                                st.image(visualization, caption="Original vs Encoded Image vs Difference")
                                
                                buf = io.BytesIO()
                                encoded_obj.save(buf, format="PNG")
                                st.download_button(
                                    label="Download Stego Image",
                                    data=buf.getvalue(),
                                    file_name="stego_image.png",
                                    mime="image/png"
                                )
                            
                            elif cover_file.type.startswith('audio'):
                                encoded_obj, original_audio, encoded_audio = encode_audio(cover_file, binary_payload, num_lsb)
                                progress_bar.progress(100)
                                st.success("Payload encoded successfully into the cover audio!")
                                
                                visualization = visualize_audio(original_audio, encoded_audio)
                                st.image(visualization, caption="Original vs Encoded Audio Waveform vs Difference")
                                
                                st.audio(encoded_obj.getvalue(), format="audio/wav")
                                st.download_button(
                                    label="Download Stego Audio",
                                    data=encoded_obj.getvalue(),
                                    file_name="stego_audio.wav",
                                    mime="audio/wav"
                                )
                            
                            elif cover_file.type.startswith('video'):
                                encoded_obj = encode_video(cover_file, binary_payload, num_lsb)
                                progress_bar.progress(100)
                                st.success("Payload encoded successfully into the cover video!")
                                
                                visualization = visualize_video(cover_file, encoded_obj)
                                st.image(visualization, caption="Original vs Encoded Video (First Frame) vs Difference")
                                
                                st.download_button(
                                    label="Download Stego Video",
                                    data=encoded_obj.getvalue(),
                                    file_name="stego_video.mp4",
                                    mime="video/mp4"
                                )
                        
                        except Exception as e:
                            st.error(f"An error occurred during encoding: {str(e)}")
                            st.error("Please make sure you have ffmpeg installed and accessible in your system PATH.")
                else:
                    st.warning("Payload is too large for the selected cover. Please choose a larger cover or reduce the payload size.")
        except Exception as e:
            st.error(f"An error occurred while checking encoding capacity: {str(e)}")
            st.error("Please make sure the selected cover file is a valid image, audio, or video file.")

def decode_ui():
    st.header("Decode payload from stego object")

    st.subheader("Step 1: Upload Stego Object")
    stego_file = st.file_uploader("Choose a stego file to decode", type=["png", "jpg", "bmp", "wav", "mp3", "mp4", "txt"])
    
    st.subheader("Step 2: Decoding Settings")
    num_lsb = st.slider("Number of LSBs used in encoding", min_value=1, max_value=8, value=1)
    
    st.subheader("Step 3: Specify Expected Payload Type")
    payload_type = st.selectbox("Choose expected payload type", ["Text", "Image", "Audio", "Document", "Video"])

    if stego_file:
        if st.button("Decode Payload"):
            try:
                decoded_payload = advanced_decode(stego_file, num_lsb, payload_type)
                st.success("Payload decoded successfully from the stego object!")

                if payload_type == "Text":
                    st.text_area("Decoded text payload", decoded_payload, height=150)
                elif payload_type == "Image":
                    st.image(decoded_payload, caption="Decoded Image Payload")
                elif payload_type == "Audio":
                    st.audio(decoded_payload, format="audio/wav")
                elif payload_type == "Video":
                    st.download_button(
                        label="Download Decoded Video",
                        data=decoded_payload.getvalue(),
                        file_name="decoded_video.mp4",
                        mime="video/mp4"
                    )
                else:  # Document
                    st.download_button(
                        label="Download Decoded Document",
                        data=decoded_payload.getvalue(),
                        file_name="decoded_document",
                        mime="application/octet-stream"
                    )
                
                # Visualize the decoding process
                if stego_file.type.startswith('image'):
                    original_img = Image.open(stego_file)
                    decoded_data = np.array(original_img)[:,:,0] & ((1 << num_lsb) - 1)
                    visualization = visualize_image(original_img, decoded_data)
                    st.image(visualization, caption="Original Stego Image vs Extracted LSBs")
                elif stego_file.type.startswith('audio'):
                    _, audio_data = decode_audio(stego_file, num_lsb)
                    decoded_data = audio_data & ((1 << num_lsb) - 1)
                    visualization = visualize_audio(audio_data, decoded_data)
                    st.image(visualization, caption="Stego Audio Waveform vs Extracted LSBs")
                elif stego_file.type.startswith('video'):
                    decoded_data = decode_video(stego_file, num_lsb)
                    visualization = visualize_video(stego_file, decoded_data)
                    st.image(visualization, caption="Stego Video (First Frame) vs Extracted LSBs")
            
            except Exception as e:
                st.error(f"An error occurred during decoding: {str(e)}")
                st.error("Please make sure you have selected the correct number of LSBs and payload type.")

if __name__ == "__main__":
    main()