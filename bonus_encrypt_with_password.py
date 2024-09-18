import sys
import base64
import hashlib
from cryptography.fernet import Fernet, InvalidToken  
from PIL import Image

def password_to_fernet_key(password: str) -> bytes:
    """Convert a password to a Fernet key using SHA-256 hash and Base64 encoding."""
    hash = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(hash)

def image_to_binary_file_encrypted(image_path, output_file_path, password):
    key = password_to_fernet_key(password)
    cipher_suite = Fernet(key)
    with Image.open(image_path) as image:
        image = image.convert('RGB')
    
    width, height = image.size
    total_pixels = width * height
    processed_pixels = 0

    # Write dimensions to the start of the file
    with open(output_file_path, 'wb') as file:
        dimensions_data = f"{width}x{height}\n".encode()
        file.write(dimensions_data)

        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                binary_str = format(r, '08b') + format(g, '08b') + format(b, '08b')
                encrypted_data = cipher_suite.encrypt(binary_str.encode())
                file.write(encrypted_data + b'\n')  # Ensure newline is added to each data item
                processed_pixels += 1
                if processed_pixels  % 7000 == 0:   # Update progress every 7000 pixels
                    percentage_encrypt = (processed_pixels/total_pixels)*100
                    sys.stdout.write(f"\r{percentage_encrypt:.2f}%.")  # Update progress on the same line
                    sys.stdout.flush()
        sys.stdout.flush()
        sys.stdout.write("\nDONE!!!")
        sys.stdout.flush()
                

def reconstruct_image_from_data(input_file_path, output_image_path, password):
    key = password_to_fernet_key(password)
    cipher_suite = Fernet(key)

    with open(input_file_path, 'rb') as file:
        # Read dimensions first
        dimensions_line = file.readline().decode()
        width, height = map(int, dimensions_line.strip().split('x'))
        image_size = (width, height)

        all_data = file.read()

    rgb_binary_strings = all_data.split(b'\n')

    image = Image.new('RGB', image_size)
    pixel_index = 0
    total_pixels = width * height

    try:
        for y in range(image_size[1]):
            for x in range(image_size[0]):
                if pixel_index < len(rgb_binary_strings) and rgb_binary_strings[pixel_index]:
                    try:
                        decrypted_data = cipher_suite.decrypt(rgb_binary_strings[pixel_index])
                    except InvalidToken:  # Correctly handle the InvalidToken exception
                        print("Wrong password!")
                        return
                    binary_str = decrypted_data.decode('utf-8')
                    r = int(binary_str[0:8], 2)
                    g = int(binary_str[8:16], 2)
                    b = int(binary_str[16:24], 2)
                    image.putpixel((x, y), (r, g, b))
                pixel_index += 1
                if pixel_index % 7000 == 0:  
                    percentage_decrypt = (pixel_index/total_pixels)*100
                    sys.stdout.write(f"\r{percentage_decrypt:.2f}%.")  # Update progress on the same line
                    sys.stdout.flush()
        sys.stdout.flush()
        sys.stdout.write("\nDONE!!! YIPEE!!")
        sys.stdout.flush()
    except IndexError:
        print(f"IndexError at pixel index: {pixel_index}, likely due to data mismatch.")

    image.save(output_image_path)
    image.show()


    image.save(output_image_path)
    image.show()




input_file = str(sys.argv[1])
input_file_check = input_file.lower()


if input_file_check[-3:] == "jpg":
    output_file_name = input_file[:-3] + "txt"
    image_to_binary_file_encrypted(input_file, output_file_name, str(sys.argv[2])) 
    
elif input_file_check[-4:] == "jpeg":
    output_file_name = input_file[:-4] + "txt"
    image_to_binary_file_encrypted(input_file, output_file_name, str(sys.argv[2])) 
    
elif input_file_check[-3:] == "txt":
    output_file_name = input_file[:-3] + "jpg"
    reconstruct_image_from_data(input_file, output_file_name, str(sys.argv[2])) #input_file_path, output_image_path, password
    

    
