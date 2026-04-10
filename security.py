from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from base64 import b64decode
from typing import Any

try:
    from Crypto.Cipher import AES  # pycryptodome
    from Crypto.Random import get_random_bytes
except Exception:  # pragma: no cover
    from Cryptodome.Cipher import AES  # pycryptodomex
    from Cryptodome.Random import get_random_bytes

logger = logging.getLogger(__name__)


def _coerce_32_bytes_key(env_value: str) -> bytes:
    """Best-effort decode of a key string into exactly 32 bytes.

    Supported formats:
    - 64-char hex (32 bytes)
    - base64 (raw 32 bytes)
    - otherwise UTF-8 bytes, SHA-256 hashed to 32 bytes
    """
    value = env_value.strip()

    # Hex
    try:
        if len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
            key = bytes.fromhex(value)
            if len(key) == 32:
                return key
    except Exception:
        pass

    # Base64 (accept with or without prefix)
    try:
        if value.lower().startswith("base64:"):
            value = value.split(":", 1)[1].strip()
        key = b64decode(value, validate=True)
        if len(key) == 32:
            return key
    except Exception:
        pass

    # Fallback: derive stable 32 bytes
    logger.warning("Key material is not 32 bytes; deriving via SHA-256")
    return hashlib.sha256(env_value.encode("utf-8")).digest()


_AES_ENV = os.getenv("AES_KEY")
if _AES_ENV:
    AES_KEY: bytes = _coerce_32_bytes_key(_AES_ENV)
else:
    AES_KEY = os.urandom(32)
    logger.warning("AES_KEY not set; generated ephemeral key (decryption will not work after restart)")

_HMAC_ENV = os.getenv("HMAC_KEY")
if _HMAC_ENV:
    HMAC_KEY: bytes = _coerce_32_bytes_key(_HMAC_ENV)
else:
    HMAC_KEY = os.urandom(32)
    logger.warning("HMAC_KEY not set; generated ephemeral key (signatures will not verify after restart)")


def encrypt_aes_gcm(plaintext: str) -> bytes:
    nonce = get_random_bytes(12)
    cipher = AES.new(AES_KEY, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    # format: nonce(12) + tag(16) + ciphertext
    return nonce + tag + ciphertext


def decrypt_aes_gcm(ciphertext: bytes) -> str:
    if len(ciphertext) < 12 + 16:
        raise ValueError("Ciphertext too short")
    nonce = ciphertext[:12]
    tag = ciphertext[12:28]
    ct = ciphertext[28:]
    cipher = AES.new(AES_KEY, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ct, tag)
    return plaintext.decode("utf-8")


def sign_payload(data: dict[str, Any]) -> str:
    message = json.dumps(data, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hmac.new(HMAC_KEY, message, hashlib.sha256).hexdigest()


def verify_signature(data: dict[str, Any], signature: str) -> bool:
    expected = sign_payload(data)
    return hmac.compare_digest(expected, signature)
