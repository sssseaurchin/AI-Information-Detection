import base64
import binascii
import os
import uuid
from pathlib import Path


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}

def save_image_from_base64(base64_str: str, ext: str = ".png") -> Path:
    if not isinstance(base64_str, str) or not base64_str.strip():
        raise ValueError("image_base64 must be a non-empty string")

    ext = (ext or ".png").lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported extension: {ext}. Allowed: {sorted(ALLOWED_EXT)}")

    # Handle accidental data URL (even though your frontend strips it)
    s = base64_str.strip()
    if s.startswith("data:"):
        try:
            _, s = s.split(",", 1)
        except ValueError:
            raise ValueError("Invalid data URL format")

    try:
        data = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Invalid base64 payload")

    if not data:
        raise ValueError("Decoded payload is empty")
    if len(data) > MAX_BYTES:
        raise ValueError(f"Image too large (> {MAX_BYTES} bytes)")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / filename

    # Atomic write
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)

    return path