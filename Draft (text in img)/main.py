import streamlit as st
import io
from PIL import Image
import wave
import tempfile
import os
from stego import encode_text_in_image, decode_text_from_image, encode_text_in_audio, decode_text_from_audio, encode_random, decode_frame, get_frames, frames_to_video, decode_hidden_text

def main():
    st.title("LSB Steganography Application")

    # Sidebar for the operation and file type
    with st.sidebar:
        operation = st.radio("Choose operation:", ("Encode", "Decode"))
        file_type = st.radio("Choose Cover file type:", ("Image", "Audio", "Video"))
        num_lsb = st.slider("Number of LSBs to use:", 1, 8, 1)

    if operation == "Encode":
        payload = st.text_area("Enter text to hide:")
        uploaded_file = st.file_uploader("Choose a cover file", type=["jpg", "png", "bmp", "wav","mp4","avi"])
        if uploaded_file is not None and payload:
            if file_type == "Image":
                try:
                    # Open the image and encode text
                    stego_img = encode_text_in_image(uploaded_file, payload, num_lsb)
                    # Display original and stego files side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original File")
                        if file_type == "Image":
                            st.image(uploaded_file)
                    with col2:
                        st.subheader("Stego File")
                        if  file_type == "Image":
                            st.image(stego_img)
                    
                    # Offer download option
                    buf = io.BytesIO()
                    stego_img.save(buf, format="PNG")
                    st.download_button("Download Stego Image", buf.getvalue(), "stego_image.png")
                except ValueError as e:
                    st.error(str(e))

            elif file_type == "Audio":
                try:
                    # Encode text into audio file
                    stego_audio = encode_text_in_audio(uploaded_file, payload, num_lsb)
                    # Display the download button for the stego audio file
                    st.audio(uploaded_file, format="audio/wav")
                    # Allow download of the modified (stego) audio file
                    st.download_button("Download Stego Audio", stego_audio.getvalue(), "stego_audio.wav")
                except ValueError as e:
                    st.error(str(e))
            
            elif file_type == "Video":
                try:
                    # Save uploaded video to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input_video:
                        tmp_input_video.write(uploaded_file.read())
                        input_video_path = tmp_input_video.name

                    # Save the payload text to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_text_file:
                        tmp_text_file.write(payload.encode())
                        text_file_path = tmp_text_file.name

                    output_video_path = os.path.join(tempfile.gettempdir(), "stego_video.avi")
                    

                    # Extract frames, encode text into random frames, and reassemble video
                    frame_folder = os.path.join(tempfile.gettempdir(), "frames")
                    get_frames(input_video_path, frame_folder)
                    encode_random(text_file_path, frame_folder)
                    frames_to_video(frame_folder, output_video_path, fps=30)

                    # Provide the stego video for download
                    with open(output_video_path, "rb") as f:
                        st.download_button("Download Stego Video", f, file_name="stego_video.avi")
                    st.success("Encoding completed successfully!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up temporary files
                    if os.path.exists(input_video_path):
                        os.remove(input_video_path)
                    if os.path.exists(text_file_path):
                        os.remove(text_file_path)
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)

    else:  # Decode
        uploaded_file = st.file_uploader("Choose a stego file", type=["jpg", "png", "bmp", "wav","mp4","avi"])

        if uploaded_file is not None:
            try:
                if file_type == "Image":
                    # Decode text from image
                    hidden_text = decode_text_from_image(uploaded_file, num_lsb)
                    st.text_area("Hidden text:", hidden_text)

                    # Display the uploaded stego image and original image
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original File")
                        st.image(uploaded_file, caption="Uploaded Stego Image", use_column_width=True)
                    with col2:
                        st.subheader("Stego File")
                        st.image(uploaded_file, caption="Decoded Stego Image", use_column_width=True)
                        
                elif file_type == "Audio":
                    # Decode text from audio
                    hidden_text = decode_text_from_audio(uploaded_file, num_lsb)
                    st.text_area("Hidden text:", hidden_text)

                    # Play the uploaded stego audio file
                    st.audio(uploaded_file, format="audio/wav")

                elif file_type == "Video":
                    try:
                        # Save uploaded video to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp_stego_video:
                            tmp_stego_video.write(uploaded_file.read())
                            stego_video_path = tmp_stego_video.name

                        # Prepare a folder for extracting frames
                        frame_folder = os.path.join(tempfile.gettempdir(), "frames")
                        get_frames(stego_video_path, frame_folder)

                        # Temporary output file for decoded text
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_decoded_text_file:
                            decoded_text_file_path = tmp_decoded_text_file.name

                        # Use the length of the original message stored during encoding
                        decode_hidden_text(frame_folder, decoded_text_file_path)  # Pass message_length here

                        # Display the decoded text
                        with open(decoded_text_file_path, "r") as decoded_file:
                            decoded_text = decoded_file.read()

                        st.text_area("Hidden text:", decoded_text)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            
            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
