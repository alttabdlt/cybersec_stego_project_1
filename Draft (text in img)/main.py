import streamlit as st
import io
from PIL import Image
import wave
import tempfile
import os
from stego import encode_text_in_image, decode_text_from_image, encode_text_in_audio, decode_text_from_audio, encode_random, decode_frame, get_frames, frames_to_video, decode_hidden_text, encode_text_in_image, decode_text_from_image, encode_text_in_audio, decode_text_from_audio, encode_random, decode_frame, get_frames, frames_to_video,decode_hidden_text, encode_image, decode_image, encode_audio, decode_audio, can_encode, text_to_binary, binary_to_text, file_to_binary, binary_to_file, visualize_audio, visualize_image, advanced_decode, encode_video, decode_video, encode_text, decode_text, visualize_video
from stego_for_image import encode_image_in_audio , decode_image_from_audio

def stego_payload_text_page():
    st.title("LSB Steganography Application")

    # Sidebar for the operation and file type
    with st.sidebar:
        operation = st.radio("Choose operation:", ("Encode", "Decode"))
        file_type = st.radio("Choose Cover file type:", ("Image", "Audio", "Video"))
        num_lsb = st.slider("Number of LSBs to use:", 1, 8, 1)

    if operation == "Encode":
        payload = st.text_area("Enter text to hide:", height=300)
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
                    st.text_area("Hidden text:", hidden_text, height=300)

                    # Display the uploaded stego image 
                    st.subheader("Stego File")
                    st.image(uploaded_file, caption="Stego Image", use_column_width=True)
                        
                elif file_type == "Audio":
                    # Decode text from audio
                    hidden_text = decode_text_from_audio(uploaded_file, num_lsb)
                    st.text_area("Hidden text:", hidden_text, height=300) 

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

                        st.text_area("Hidden text:", decoded_text, height=300)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            
            except ValueError as e:
                st.error(str(e))

def stego_payload_image_page():
    st.title("LSB Steganography Application")

    # Sidebar for the operation and file type
    with st.sidebar:
        operation = st.radio("Choose operation:", ("Encode", "Decode"))
        file_type = st.radio("Choose Cover file type:", ("Image", "Audio", "Video"))
        num_lsb = st.slider("Number of LSBs to use:", 1, 8, 1)

    if operation == "Encode":
        uploaded_payload = st.file_uploader("Choose an image to hide", type=["jpg", "png", "bmp"], key="image")
        uploaded_cover = st.file_uploader("Choose a cover file", type=["jpg", "png", "bmp", "wav","mp4","avi"])

        if uploaded_cover is not None and uploaded_payload is not None:
            if file_type == "Audio":
                try:
                    # Open the image and encode into the audio
                    stego_audio_path, img_shape = encode_image_in_audio(uploaded_cover, uploaded_payload, num_lsb)
                    
                    # Display original audio and stego files side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original File")
                        st.audio(uploaded_cover)
                    with col2:
                        st.subheader("Stego Audio")
                        st.audio(stego_audio_path)

                    # Offer download option for the stego audio
                    with open(stego_audio_path, "rb") as file:
                        st.download_button("Download Stego Audio", file, "stego_audio.wav")
                except ValueError as e:
                    st.error(str(e))
            elif file_type == "Image":
                try:
                    # Convert uploaded payload to binary
                    payload_binary = file_to_binary(uploaded_payload)
                    
                    # Open the image and encode image
                    stego_img = encode_image(uploaded_cover, payload_binary, num_lsb)
                    
                    # Display original and stego files side by side
                    visualization = visualize_image(Image.open(uploaded_cover), stego_img)
                    st.image(visualization, caption="Original vs Encoded Image vs Difference")
        
                    # Offer download option
                    buf = io.BytesIO()
                    stego_img.save(buf, format="PNG")
                    st.download_button("Download Stego Image", buf.getvalue(), "stego_image.png")
                except ValueError as e:
                    st.error(str(e))

    else:  # Decode
        uploaded_stego = st.file_uploader("Choose a stego file", type=["wav", "jpg", "png", "bmp"])

        if uploaded_stego is not None:
            try:
                if file_type == "Audio":
                    # Decode image from the audio
                    decoded_image = decode_image_from_audio(uploaded_stego, num_lsb)
                    st.image(decoded_image, caption="Hidden Image")
                elif file_type == "Image":
                    # Decode image from the image
                    #decoded_payload = decode_image(uploaded_stego, num_lsb)
                    #print(f"Num LSB: {num_lsb}")
                    decoded_payload = advanced_decode(uploaded_stego, num_lsb, file_type)
                    #decoded_image = binary_to_file(decoded_payload, "image")
                    #st.image(decoded_image, caption="Hidden Image")
                    st.image(decoded_payload, caption="Decoded Image Payload")

            except ValueError as e:
                st.error(str(e))

# Main function to manage the pages
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose A Payload", ["For Text", "For Image"])

    if page == "For Text":
        stego_payload_text_page()
    elif page == "For Image":
        stego_payload_image_page()  # Call the main function from the second file

if __name__ == "__main__":
    main()
