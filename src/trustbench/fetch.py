"""Fetch pinned artifacts without importing or executing upstream code."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import urllib.request

from .io import digest


def fetch_one(url, destination, expected_sha256, max_bytes=50_000_000, opener=urllib.request.urlopen):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != expected_sha256:
            raise ValueError(f'Existing artifact differs from manifest: {destination.name}')
        return 'verified_cache'
    request = urllib.request.Request(url, headers={'User-Agent': 'slm-trust-bench/0.1'})
    with opener(request, timeout=60) as response:
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError('Artifact exceeds download limit')
    if sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Downloaded artifact hash differs from manifest; nothing saved')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False, suffix='.download') as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return 'downloaded_and_verified'


def fetch(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    names = {'value_reasoning': 'value_reasoning.zip', 'squad_v2_validation': 'squad_v2_validation.parquet'}
    outcomes = {}
    for row in manifest['datasets']:
        name = names.get(row['id'], row['id'] + '.json')
        outcomes[row['id']] = fetch_one(row['url'], root / 'data/raw' / name, row['sha256'])
    for row in manifest['prompt_sources']:
        url = f"https://raw.githubusercontent.com/{row['repository']}/{row['revision']}/{row['file']}"
        destination = root / 'data/sources' / row['repository'].replace('/', '__') / 'files' / row['file']
        outcomes[row['file']] = fetch_one(url, destination, row['sha256'])
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(fetch(args.workspace), indent=2))


if __name__ == '__main__':
    main()
