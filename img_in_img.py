import streamlit as st
from PIL import Image
import numpy as np

# Function to encode the image
def encode_image(cover_image, secret_image, num_lsb):
    cover_pixels = np.array(cover_image.convert('RGB'))  # Ensure the cover image is RGB
    secret_pixels = np.array(secret_image.convert('RGB'))  # Ensure the secret image is RGB
    
    # Ensure the secret image can be encoded in the cover image
    if secret_pixels.size * num_lsb > cover_pixels.size:
        raise ValueError("Secret image is too large to encode in the cover image.")
    
    # Encode the secret image into the cover image
    for i in range(secret_pixels.shape[0]):
        for j in range(secret_pixels.shape[1]):
            for k in range(3):  # Only process RGB channels
                cover_pixel_bin = format(cover_pixels[i, j, k], '08b')
                secret_pixel_bin = format(secret_pixels[i, j, k], '08b')[:num_lsb]
                new_pixel_bin = cover_pixel_bin[:-num_lsb] + secret_pixel_bin
                cover_pixels[i, j, k] = int(new_pixel_bin, 2)
    
    encoded_image = Image.fromarray(cover_pixels)
    return encoded_image

# Function to decode the image
def decode_image(encoded_image, num_lsb):
    encoded_pixels = np.array(encoded_image)
    
    secret_pixels = np.zeros_like(encoded_pixels)
    for i in range(encoded_pixels.shape[0]):
        for j in range(encoded_pixels.shape[1]):
            for k in range(3):  # Only process RGB channels
                encoded_pixel_bin = format(encoded_pixels[i, j, k], '08b')
                secret_pixel_bin = encoded_pixel_bin[-num_lsb:] + '0' * (8 - num_lsb)
                secret_pixels[i, j, k] = int(secret_pixel_bin, 2)
    
    decoded_image = Image.fromarray(secret_pixels)
    return decoded_image

# Streamlit UI
st.title("Image Steganography with LSB Encoding")
st.header("Encode an Image into Another Image")

# File uploaders
cover_image_file = st.file_uploader("Choose a cover image", type=["png", "jpg", "jpeg"])
secret_image_file = st.file_uploader("Choose a secret image", type=["png", "jpg", "jpeg"])

# LSB selection
num_lsb = st.slider("Number of LSBs to use for encoding", 1, 8, 1)

if cover_image_file and secret_image_file:
    cover_image = Image.open(cover_image_file)
    secret_image = Image.open(secret_image_file)
    
    st.image(cover_image, caption="Cover Image", use_column_width=True)
    st.image(secret_image, caption="Secret Image", use_column_width=True)
    
    # Encode the secret image into the cover image
    if st.button("Encode Image"):
        try:
            encoded_image = encode_image(cover_image, secret_image, num_lsb)
            st.image(encoded_image, caption="Encoded Image", use_column_width=True)
            
            # Download option for the encoded image
            encoded_image.save("encoded_image.png")
            with open("encoded_image.png", "rb") as file:
                st.download_button("Download Encoded Image", file, file_name="encoded_image.png")
        except ValueError as e:
            st.error(e)

# Decoding section
st.header("Decode an Image")
encoded_image_file = st.file_uploader("Choose an encoded image to decode", type=["png", "jpg", "jpeg"])

if encoded_image_file:
    encoded_image = Image.open(encoded_image_file)
    st.image(encoded_image, caption="Encoded Image", use_column_width=True)
    
    if st.button("Decode Image"):
        decoded_image = decode_image(encoded_image, num_lsb)
        st.image(decoded_image, caption="Decoded Image", use_column_width=True)
