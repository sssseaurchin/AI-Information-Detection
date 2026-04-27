import base64
import binascii
import os
import uuid
from pathlib import Path


def detect_extension(data: bytes) -> str:
    """Detect image type from magic bytes"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
        return ".webp"
    raise ValueError("Unsupported or unknown image format")


def extract_ext_from_data_url(header: str) -> str | None:
    """Extract extension from data URL header"""
    if "image/png" in header:
        return ".png"
    if "image/jpeg" in header or "image/jpg" in header:
        return ".jpg"
    if "image/webp" in header:
        return ".webp"
    return None


def save_image_from_base64(base64_str: str, ext: str = None) -> Path:
    if not isinstance(base64_str, str) or not base64_str.strip():
        raise ValueError("image_base64 must be a non-empty string")

    s = base64_str.strip()
    ext_from_header = None

    # Handle data URL
    if s.startswith("data:"):
        try:
            header, s = s.split(",", 1)
            ext_from_header = extract_ext_from_data_url(header)
        except ValueError:
            raise ValueError("Invalid data URL format")

    # Decode
    try:
        data = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Invalid base64 payload")

    if not data:
        raise ValueError("Decoded payload is empty")
    if len(data) > MAX_BYTES:
        raise ValueError(f"Image too large (> {MAX_BYTES} bytes)")

    # Detect real extension from bytes
    detected_ext = detect_extension(data)

    # Final extension decision priority:
    # 1. Detected from bytes (authoritative)
    # 2. Data URL header (if consistent)
    # 3. User-provided ext (ignored if conflicting)

    final_ext = detected_ext

    if ext_from_header and ext_from_header != detected_ext:
        raise ValueError("Mismatch between data URL type and file content")

    if ext:
        ext = ext.lower().strip()
        if not ext.startswith("."):
            ext = "." + ext
        if ext == ".jpeg":
            ext = ".jpg"
        if ext != detected_ext:
            raise ValueError("Provided extension does not match file content")

    if final_ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported extension: {final_ext}")

    filename = f"{uuid.uuid4().hex}{final_ext}"
    path = UPLOAD_DIR / filename

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)

    return path.absolute()


if __name__ == "__main__":
    base64_sample = "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"
    ext = ".png"
    save_image_from_base64(base64_sample, ext)
    print("Image saved successfully.")
