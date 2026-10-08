from __future__ import annotations

import gzip
import zlib


def decode_http_body(raw: bytes, content_encoding: str | None = None) -> bytes:
    """Decode HTTP Content-Encoding without changing source/PIT semantics."""
    if not raw:
        return raw

    encoding = (content_encoding or "").lower()
    if "gzip" in encoding or raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)

    if "deflate" in encoding:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)

    return raw
