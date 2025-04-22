from Crypto.Cipher import AES
import base64
import hashlib

SECRET_KEY = 'your-secret-key'  # Same as frontend

def pad(s):
    return s + (16 - len(s) % 16) * chr(16 - len(s) % 16)

def unpad(s):
    return s[:-ord(s[len(s) - 1:])]

def encrypt_message(raw):
    raw = pad(raw)
    key = hashlib.sha256(SECRET_KEY.encode()).digest()
    iv = b'16byteslongiv123'  # 16-byte IV for AES
    cipher = AES.new(key, AES.MODE_CBC, iv)
    enc = cipher.encrypt(raw.encode())
    return base64.b64encode(enc).decode()

def decrypt_message(enc):
    enc = base64.b64decode(enc)
    key = hashlib.sha256(SECRET_KEY.encode()).digest()
    iv = b'16byteslongiv123'
    cipher = AES.new(key, AES.MODE_CBC, iv)
    dec = cipher.decrypt(enc).decode()
    return unpad(dec)
