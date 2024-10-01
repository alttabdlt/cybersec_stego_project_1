import hashlib

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

original_hash = get_file_hash('original_payload')
decoded_hash = get_file_hash('decoded_payload')

print(f"Original payload hash: {original_hash}")
print(f"Decoded payload hash: {decoded_hash}")

if original_hash == decoded_hash:
    print("The decoded payload matches the original payload.")
else:
    print("The decoded payload does NOT match the original payload.")