"""单独运行新闻或 K 线 Agent，逐项保存并支持断点续跑。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from agents import news_agent, kline_agent
from dataflows.news_data import load_exported_news
from dataflows.price_data import read_window
from dataflows.kline_chart import draw_candlesticks
from model_client import ModelClient


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path, data):
    """完整写入临时文件后原子建立新文件，不覆盖已有结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".analyst-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(name, path)  # Windows rename 原子且不会覆盖已有文件。
        else:
            os.link(name, path)  # POSIX rename 可能覆盖，用硬链接保持不覆盖。
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_new(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_bytes(path, data.encode("utf-8"))


def _match(root, stock, *, directories):
    matches = [p for p in root.iterdir() if (p.is_dir() if directories else p.is_file() and p.suffix.lower() == ".csv")
               and (p.name if directories else p.stem).casefold() == stock.casefold()]
    if len(matches) != 1:
        raise ValueError("找不到唯一的股票数据；请核对新闻和行情代码。")
    return matches[0]


def run(stock, prediction_date, role, *, client, model="deepseek-flash", data_root="data", output_root="outputs/analysts_with_reasons", limit=None):
    """只保存去敏文本和模型返回；真实股票与日期仅作本地定位。"""
    from datetime import date
    day = date.fromisoformat(prediction_date).isoformat()
    if not isinstance(stock, str) or not stock or Path(stock).name != stock or stock in (".", "..") or any(c in stock for c in "/\\:"):
        raise ValueError("股票代码必须是单个文件名。")
    if role not in ("news", "kline") or not model:
        raise ValueError("请指定 news 或 kline 及模型。")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit 必须是正整数。")
    root = Path(data_root)
    if role == "news":
        newsroot = root / "US" / "news"
        directory = _match(newsroot / "masked", stock, directories=True)
        inputs = load_exported_news(newsroot, directory.name, day)
        agent = news_agent
        content_digest = _sha(json.dumps(inputs, ensure_ascii=False).encode("utf-8"))
    else:
        prices = _match(root / "CMIN-Dataset" / "CMIN-US" / "price" / "raw", stock, directories=False)
        image = draw_candlesticks(read_window(prices, day, window=20))
        inputs = [image]
        agent = kline_agent
        content_digest = _sha(image)
    provider_hash = _sha(getattr(client, "base_url", "").encode("utf-8"))
    metadata = {"role": role, "model": model, "provider_hash": provider_hash,
                "prompt_hash": _sha((agent.PROMPT + "\n" + agent.INSTRUCTION).encode("utf-8")),
                "input_hash": content_digest, "total": len(inputs)}
    directory = Path(output_root) / stock / day / role
    manifest = directory / "manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("已有结果与当前模型、提示词或输入不一致，请使用新的输出文件夹。")
    else:
        if any(directory.iterdir()):
            raise ValueError("输出文件夹已有文件但无清单，请使用新的输出文件夹。")
        _write_new(manifest, metadata)
    if role == "kline":
        chart_path = directory / "chart.png"
        if chart_path.exists():
            if _sha(chart_path.read_bytes()) != content_digest:
                raise ValueError("保存的 K 线图与当前输入不一致。")
        else:
            _write_bytes(chart_path, image)
    stop = len(inputs) if limit is None else min(limit, len(inputs))
    saved = skipped = 0
    for index in range(stop):
        path = directory / f"{index + 1:04d}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("index") != index + 1 or not isinstance(existing.get("response"), str) or not existing["response"].strip():
                raise ValueError(f"已有结果文件无效：{path}")
            if role == "news" and existing.get("text") != inputs[index]:
                raise ValueError(f"已有结果与新闻标题不一致：{path}")
            skipped += 1
            continue
        response = agent.analyze(inputs[index], client=client, model=model)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("模型未返回有效文本，未保存该结果。")
        result = {"index": index + 1, "response": response}
        if role == "news":
            result["text"] = inputs[index]
        _write_new(path, result)
        saved += 1
    return {"directory": str(directory), "total": len(inputs), "processed_limit": stop, "saved": saved, "skipped": skipped}


def _load_config():
    """当前环境优先；仅读取项目 .env 中服务地址和密钥。"""
    config = Path(__file__).resolve().parent / ".env"
    if config.exists():
        for line in config.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if sep and key in ("MODEL_BASE_URL", "MODEL_API_KEY"):
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
                    value = value[1:-1]
                os.environ.setdefault(key, value)


def main():
    parser = argparse.ArgumentParser(description="单独运行 News 或 K-line Agent。")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--date", required=True, help="预测日 YYYY-MM-DD")
    parser.add_argument("--role", required=True, choices=("news", "kline"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/analysts_with_reasons")
    parser.add_argument("--limit", type=int, help="最多处理前 N 条新闻（可稍后继续）")
    args = parser.parse_args()
    _load_config()
    client = ModelClient.from_environment()
    print(json.dumps(run(args.stock, args.date, args.role, client=client, model=args.model,
                         data_root=args.data_root, output_root=args.output_root, limit=args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
