"""Offline checks for the archived SFT experiment; no external dependencies."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SPECS = [
    ('main', 'sft-jbb-efcce86f8389', 'vea-judge-76e95e7049a9', 264),
    ('checkpoint', 'sft-jbb-644d21da7e08', 'vea-judge-7356919ece86', 256),
]


def jsonl(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def verify():
    sums = ROOT / 'SHA256SUMS.json'
    if sums.exists():
        for name, expected in json.loads(sums.read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with zipfile.ZipFile(ROOT / 'sft-results.zip') as archive:
        assert archive.testzip() is None
        for label, run, judge, positives in SPECS:
            prefix = 'data/' + run + '/'
            rb = (ROOT / f'{label}-responses.jsonl').read_bytes()
            jb = (ROOT / f'{label}-judgments.jsonl').read_bytes()
            assert rb == archive.read(prefix + 'judge-ready/sft-jbb-responses.jsonl')
            assert jb == archive.read(prefix + judge + '/judgments.jsonl')
            responses, judgments = jsonl(rb), jsonl(jb)
            assert len(responses) == len(judgments) == 800
            ids = {(f'jbb_{i}', j) for i in range(100) for j in range(8)}
            assert {(r['prompt_id'], r['rollout_id']) for r in responses} == ids
            by_id = {r['record_id']: r for r in judgments}
            assert len(by_id) == 800
            manifest = json.loads(archive.read(prefix + 'judge-ready/manifest.json'))
            assert hashlib.sha256(rb).hexdigest() == manifest['responses_sha256']
            raw = {}
            for name in archive.namelist():
                if name.startswith(prefix + 'question-') and name.endswith('.json'):
                    question = json.loads(archive.read(name))
                    for answer in question['answers']:
                        raw[question['prompt_id'], answer['rollout_id']] = answer['completion']
            assert set(raw) == ids
            for r in responses:
                j = by_id[r['record_id']]
                assert r['completion'] == raw[r['prompt_id'], r['rollout_id']]
                assert hashlib.sha256(r['completion'].encode()).hexdigest() == r['completion_sha256'] == j['completion_sha256']
                assert j['ok'] and type(j['vea']) is bool
                assert r['revision'] == j['revision']
            assert sum(j['vea'] for j in judgments) == positives
            print(f'{label}: 100 questions, 800 matched responses and judgments; {positives}/800 VEA ({positives / 8:.1f}%).')
    print('All archive, checksum, response, and judgment checks passed.')


if __name__ == '__main__':
    verify()
