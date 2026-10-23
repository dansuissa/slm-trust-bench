from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from trustbench.fetch import fetch_one


class FetchTests(unittest.TestCase):
    def test_hash_mismatch_never_saves_download_or_overwrites_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact'
            expected = sha256(b'expected').hexdigest()
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                fetch_one('https://example.org/source', path, expected, opener=lambda *a, **k: BytesIO(b'wrong'))
            self.assertFalse(path.exists())
            path.write_bytes(b'changed cache')
            with self.assertRaisesRegex(ValueError, 'Existing artifact'):
                fetch_one('https://example.org/source', path, expected, opener=lambda *a, **k: self.fail('Must not download'))
            self.assertEqual(path.read_bytes(), b'changed cache')

    def test_size_bound_and_verified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifact'
            expected = sha256(b'abc').hexdigest()
            with self.assertRaisesRegex(ValueError, 'download limit'):
                fetch_one('https://example.org/source', path, expected, max_bytes=2, opener=lambda *a, **k: BytesIO(b'abc'))
            self.assertFalse(path.exists())
            self.assertEqual(fetch_one('https://example.org/source', path, expected, opener=lambda *a, **k: BytesIO(b'abc')), 'downloaded_and_verified')
            self.assertEqual(fetch_one('https://example.org/source', path, expected, opener=lambda *a, **k: self.fail('Must use cache')), 'verified_cache')
