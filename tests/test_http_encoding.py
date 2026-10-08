from __future__ import annotations

import gzip
import zlib

from src.data.http_encoding import decode_http_body


def test_decode_http_body_identity():
    raw = b'{"status":"ok"}'
    assert decode_http_body(raw) == raw


def test_decode_http_body_gzip_header():
    raw = gzip.compress(b'{"status":"ok"}')
    assert decode_http_body(raw, "gzip") == b'{"status":"ok"}'


def test_decode_http_body_gzip_magic_without_header():
    raw = gzip.compress(b'{"status":"ok"}')
    assert decode_http_body(raw, "") == b'{"status":"ok"}'


def test_decode_http_body_deflate():
    raw = zlib.compress(b'{"status":"ok"}')
    assert decode_http_body(raw, "deflate") == b'{"status":"ok"}'
