# cybersec_stego_project_1

## Encoding/Decoding Image (Payload) into Video (Cover)
### Command to start Streamlit: streamlit run main.py --server.maxUploadSize 500

To ensure compatibility between your current decoding function and encoding done by other software, the following requirements should be met:

### 1. LSB (Least Significant Bit) Method
The encoding software should use the same **LSB steganography technique** as your implementation.

### 2. Number of LSBs
The number of least significant bits used for encoding should match the `num_lsb` parameter in your decoding function.

### 3. Payload Length Encoding
The first frame of the video should contain the **payload length encoded in 32 bits using 1 LSB**, as seen in your `decode_video` function.

### 4. Frame-by-Frame Encoding
The payload data should be encoded **across subsequent frames**, starting from the second frame.

### 5. Bit Order
The bits should be encoded and decoded in the same order (e.g., **most significant bit first**).

### 6. Color Channel Order
The encoding should use the same **color channel order** (typically BGR for OpenCV) as your decoding function.

### 7. Video Codec
The encoded video should use a **lossless codec** (e.g., FFV1) to preserve the encoded data, as your encoding function does.

### 8. Frame Extraction
The encoding process should **not alter the video's frame rate or resolution**, as your decoding function assumes these remain constant.

### 9. Payload Format
For **image payloads**, the encoding should include the **image dimensions in the first 6 bytes** of the payload, as your decoding function expects.

### 10. Error Handling
The encoding software should perform similar **capacity checks** to ensure the payload fits within the cover video, as your `can_encode` function does.
