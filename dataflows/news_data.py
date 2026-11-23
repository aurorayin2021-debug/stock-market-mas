"""从本地 CMIN 文件读取预测日前一自然日的已去敏新闻。"""
from datetime import date, datetime, timedelta
import json
from pathlib import Path


def load_news(news_root, stock, prediction_date, *, require_complete=True):
    """返回去敏标题列表；预测日不进入模型。

    news_root 是 CMIN-US/news 或 CMIN-CN/news；stock 只用于本地查找。
    如果源文件不存在，无法判断是无新闻还是未下载，明确报错。
    默认要求该日每条源新闻都有对应的已复核记录；可显式允许不完整。
    """
    if not isinstance(stock, str) or not stock or Path(stock).name != stock or stock in ('.', '..'):
        raise ValueError('股票标识必须是单个目录名。')
    day = date.fromisoformat(prediction_date) - timedelta(days=1)
    root = Path(news_root)
    source = root / 'preprocessed' / stock / day.isoformat()
    masked = root / 'masked' / stock / day.isoformat()
    if not source.is_file():
        raise FileNotFoundError(f'找不到源新闻文件，无法区分无新闻和未下载：{source}')

    with source.open(encoding='utf-8-sig') as f:
        originals = [json.loads(line) if line.strip() else None for line in f]
    expected = {n for n, row in enumerate(originals, 1) if row is not None}
    if not masked.is_file():
        if expected:
            raise FileNotFoundError(f'该日尚无已复核的去敏新闻：{masked}')
        return []
    with masked.open(encoding='utf-8-sig') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    seen = set()
    for row in rows:
        n = row.get('source_line')
        if type(n) is not int or n not in expected or n in seen:
            raise ValueError('去敏文件的源行号无效或重复。')
        if row.get('created_at') != originals[n - 1].get('created_at'):
            raise ValueError('去敏文件与源新闻的时间不一致。')
        try:
            recorded_day = datetime.fromisoformat(row['created_at']).date()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('新闻发布时间无法解析。') from error
        if recorded_day != day:
            raise ValueError('新闻发布时间与所在日期文件不一致。')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('去敏标题必须是非空文本。')
        seen.add(n)
    if require_complete and seen != expected:
        raise ValueError(f'{day} 的新闻去敏尚未完成：{len(seen)}/{len(expected)}。')
    return [row['text'] for row in sorted(rows, key=lambda row: row['source_line'])]


def load_exported_news(news_root, stock, prediction_date):
    """读取 data/US/news 下前一自然日的 JSONL，仅返回标题。

    文件必须存在；空文件表示本地数据中该股票当日无新闻。
    """
    if not isinstance(stock, str) or not stock or Path(stock).name != stock or stock in ('.', '..'):
        raise ValueError('股票标识必须是单个目录名。')
    day = date.fromisoformat(prediction_date) - timedelta(days=1)
    path = Path(news_root) / 'masked' / stock / day.isoformat()
    result = []
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {'text', 'created_at'}:
                    raise ValueError('新闻记录必须恰好包含 text 和 created_at。')
                if not isinstance(row['text'], str) or not row['text'].strip():
                    raise ValueError('去敏标题必须是非空文本。')
                timestamp = row['created_at']
                if not isinstance(timestamp, str) or not ('T' in timestamp or ' ' in timestamp):
                    raise ValueError('新闻发布时间必须包含日期和时间。')
                if datetime.fromisoformat(timestamp).date() != day:
                    raise ValueError('新闻发布时间与所在日期文件不一致。')
            except (TypeError, ValueError) as error:
                raise ValueError(f'新闻记录第 {number} 行无效：{error}') from error
            result.append(row['text'])
    return result
