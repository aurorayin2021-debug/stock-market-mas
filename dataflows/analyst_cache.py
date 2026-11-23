"""Read complete, previously saved analyst responses without calling models."""
import hashlib
import json
from datetime import date
from pathlib import Path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def load_analyst_cache(cache_root, stock, prediction_date):
    """Return (responses, provenance). The caller keeps provenance local."""
    if not isinstance(stock, str) or not stock or Path(stock).name != stock or stock in ('.', '..') or any(c in stock for c in '/\\:'):
        raise ValueError('Invalid stock identifier.')
    day = date.fromisoformat(prediction_date).isoformat()
    root = Path(cache_root)
    matches = [p for p in root.iterdir() if p.is_dir() and p.name.casefold() == stock.casefold()]
    if len(matches) != 1:
        raise ValueError('No unique cache directory for this stock.')
    responses, provenance = {}, {}
    for role in ('news', 'kline'):
        directory = matches[0] / day / role
        path = directory / 'manifest.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {role} cache manifest: {path}')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        total = manifest.get('total')
        if manifest.get('role') != role or type(total) is not int or total < 0 or (role == 'kline' and total != 1):
            raise ValueError(f'Invalid {role} cache manifest.')
        for key in ('model', 'provider_hash', 'prompt_hash', 'input_hash'):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                raise ValueError(f'Missing {key} in {role} cache manifest.')
        expected = {f'{i:04d}.json' for i in range(1, total + 1)}
        present = {p.name for p in directory.glob('[0-9]*.json') if p.is_file()}
        if present != expected:
            raise ValueError(f'Incomplete or unexpected {role} cache files.')
        result, headlines = [], []
        for index in range(1, total + 1):
            row = json.loads((directory / f'{index:04d}.json').read_text(encoding='utf-8'))
            if row.get('index') != index or not isinstance(row.get('response'), str) or not row['response'].strip():
                raise ValueError(f'Invalid {role} cache response at index {index}.')
            if role == 'news':
                if not isinstance(row.get('text'), str) or not row['text'].strip():
                    raise ValueError(f'Invalid news cache input at index {index}.')
                headlines.append(row['text'])
            result.append(row['response'])
        if role == 'news':
            digest = _sha(json.dumps(headlines, ensure_ascii=False).encode('utf-8'))
        else:
            chart = directory / 'chart.png'
            if not chart.is_file():
                raise FileNotFoundError('Missing cached candlestick chart.')
            digest = _sha(chart.read_bytes())
        if digest != manifest['input_hash']:
            raise ValueError(f'{role} cache input does not match its manifest.')
        responses[role] = result if role == 'news' else result[0]
        provenance[role] = manifest
    return responses, provenance
