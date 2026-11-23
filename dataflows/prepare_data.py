"""把本地新闻与行情准备为两个 Agent 可用的输入。"""
from pathlib import Path

from dataflows.news_data import load_news, load_exported_news
from dataflows.kline_chart import draw_candlesticks
from dataflows.price_data import read_window


def prepare_analyst_inputs(data_root, market, stock, prediction_date, *, price_symbol=None, price_window=20, require_complete_news=True):
    """prediction_date 是被预测的交易日期；所有输入均早于该日。

    stock 是新闻目录名，price_symbol 是价格文件名（不含 .csv）；
    A 股两者可能不同，须显式提供经过核实的 price_symbol。
    """
    if market not in ('US', 'CN'):
        raise ValueError('市场必须为 US 或 CN。')
    price_symbol = price_symbol if price_symbol is not None else stock
    for name in (stock, price_symbol):
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('股票标识必须是单个文件名。')
    root = Path(data_root) / f'CMIN-{market}'
    news = load_news(root / 'news', stock, prediction_date, require_complete=require_complete_news)
    prices = read_window(root / 'price' / 'raw' / f'{price_symbol}.csv', prediction_date, window=price_window)
    return {'news': news, 'kline': draw_candlesticks(prices)}


def prepare_us_inputs(data_root, stock, prediction_date, *, price_symbol=None, estimator="author_code"):
    """准备 US 新闻、20 个交易日 K 线和供本地关系工具使用的配置。

    `prediction_date` 是预测日；新闻窗口为前一自然日，行情严格早于预测日。
    `estimator` 默认使用作者代码指标，并由关系工具在实际计算时检查。
    """
    if not isinstance(stock, str) or not stock or Path(stock).name != stock or stock in ('.', '..'):
        raise ValueError('股票标识必须是单个文件名。')
    symbol = stock if price_symbol is None else price_symbol
    if not isinstance(symbol, str) or not symbol or Path(symbol).name != symbol or symbol in ('.', '..'):
        raise ValueError('行情标识必须是单个文件名。')
    if symbol.casefold() != stock.casefold():
        raise ValueError('新闻与行情的股票代码不一致，不能隐式映射不同证券。')
    root = Path(data_root)
    news_root = root / 'US' / 'news'
    price_root = root / 'CMIN-Dataset' / 'CMIN-US' / 'price' / 'raw'
    news_matches = [p for p in (news_root / 'masked').iterdir() if p.is_dir() and p.name.casefold() == stock.casefold()]
    price_matches = [p for p in price_root.iterdir() if p.is_file() and p.suffix.lower() == '.csv' and p.stem.casefold() == symbol.casefold()]
    if len(news_matches) != 1 or len(price_matches) != 1:
        raise ValueError('未找到唯一匹配的新闻目录和行情文件。')
    news = load_exported_news(news_root, news_matches[0].name, prediction_date)
    prices = read_window(price_matches[0], prediction_date, window=20)
    return {
        'news': news,
        'kline': draw_candlesticks(prices),
        'relation': {
            'stock': price_matches[0].stem, 'price_root': str(price_root),
            'prediction_date': prediction_date, 'estimator': estimator,
        },
    }
