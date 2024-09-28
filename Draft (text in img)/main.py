import streamlit as st
import io
from PIL import Image
import wave
import tempfile
import os
from stego import encode_text_in_image, decode_text_from_image, encode_text_in_audio, decode_text_from_audio

def main():
    st.title("LSB Steganography Application")

    # Sidebar for the operation and file type
    with st.sidebar:
        operation = st.radio("Choose operation:", ("Encode", "Decode"))
        file_type = st.radio("Choose Cover file type:", ("Image", "Audio"))
        num_lsb = st.slider("Number of LSBs to use:", 1, 8, 1)

    if operation == "Encode":
        payload = st.text_area("Enter text to hide:")
        uploaded_file = st.file_uploader("Choose a cover file", type=["jpg", "png", "bmp", "wav"])

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
            

    else:  # Decode
        uploaded_file = st.file_uploader("Choose a stego file", type=["jpg", "png", "bmp", "wav"])

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
            
            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
