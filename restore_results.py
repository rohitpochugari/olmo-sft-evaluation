"""Restore saved SFT results without API calls or overwriting different files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def restore(destination):
    root = destination.resolve()
    writes = {}
    with zipfile.ZipFile(Path(__file__).with_name('sft-results.zip')) as archive:
        for name in archive.namelist():
            if not name.startswith('data/'):
                continue
            rel = name.removeprefix('data/')
            if rel.startswith('prepared/'):
                rel = rel.removeprefix('prepared/')
            path = (root / rel).resolve()
            if root not in path.parents:
                raise ValueError('Unsafe archive path')
            writes[path] = archive.read(name)
            if name.endswith('/judgments.jsonl'):
                for line in writes[path].decode().splitlines():
                    record = json.loads(line)
                    suffix = hashlib.sha256(record['record_id'].encode()).hexdigest()[:24]
                    writes[path.parent / ('answer-' + suffix + '.json')] = (
                        json.dumps(record, ensure_ascii=False, indent=2).encode()
                    )
    # Check every collision before writing any file. JSON spacing differences are harmless.
    for path, data in writes.items():
        if path.exists() and path.read_bytes() != data:
            if path.suffix != '.json' or json.loads(path.read_bytes()) != json.loads(data):
                raise ValueError(f'Existing file differs; nothing overwritten: {path}')
    count = 0
    for path, data in writes.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            count += 1
    print(f'Restored {count} missing files to {root}; existing matching files preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    restore(parser.parse_args().destination)
