"""并发运行新闻与 K 线分析：统一限流，逐条保存，支持续跑。"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
import json
import re
from pathlib import Path

from agents import news_agent, kline_agent
from dataflows.news_data import load_exported_news
from dataflows.price_data import read_window
from dataflows.kline_chart import draw_candlesticks
from model_client import ModelClient
from run_analysts import _load_config, _match, _sha, _write_bytes, _write_new


def _prepare(stock, day, role, client, model, data_root, output_root):
    if not isinstance(stock, str) or not stock or Path(stock).name != stock or stock in (".", "..") or any(c in stock for c in "/\\:"):
        raise ValueError("股票代码必须是单个文件名。")
    root = Path(data_root)
    if role == "news":
        news_root = root / "US" / "news"
        stock_dir = _match(news_root / "masked", stock, directories=True)
        inputs = load_exported_news(news_root, stock_dir.name, day)
        agent = news_agent
        digest = _sha(json.dumps(inputs, ensure_ascii=False).encode("utf-8"))
    else:
        price_path = _match(root / "CMIN-Dataset" / "CMIN-US" / "price" / "raw", stock, directories=False)
        inputs = [draw_candlesticks(read_window(price_path, day, window=20))]
        agent = kline_agent
        digest = _sha(inputs[0])
    metadata = {"role": role, "model": model,
                "provider_hash": _sha(getattr(client, "base_url", "").encode("utf-8")),
                "prompt_hash": _sha((agent.PROMPT + "\n" + agent.INSTRUCTION).encode("utf-8")),
                "input_hash": digest, "total": len(inputs)}
    directory = Path(output_root) / stock / day / role
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
            raise ValueError("已有结果与当前模型、提示词或输入不一致，请使用新的输出目录。")
    else:
        if any(directory.iterdir()):
            raise ValueError("输出文件夹已有文件但无清单。")
        _write_new(manifest, metadata)
    if role == "kline":
        chart = directory / "chart.png"
        if chart.exists():
            if _sha(chart.read_bytes()) != digest:
                raise ValueError("保存的 K 线图与当前输入不一致。")
        else:
            _write_bytes(chart, inputs[0])
    jobs, skipped = [], 0
    for index, value in enumerate(inputs, 1):
        path = directory / f"{index:04d}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("index") != index or not isinstance(existing.get("response"), str) or not existing["response"].strip():
                raise ValueError(f"已有结果文件无效：{path}")
            if role == "news" and existing.get("text") != value:
                raise ValueError(f"已有结果与新闻标题不一致：{path}")
            skipped += 1
        else:
            jobs.append((stock, role, index, value, path, agent))
    return jobs, skipped



def _now():
    return datetime.now(timezone.utc)


def _boundary(value):
    if value is None:
        return None
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("时间边界必须包含时区。")
    return value


def _window_status(start_at, stop_at):
    now = _now()
    if start_at is not None and now < start_at:
        return "not_started"
    if stop_at is not None and now >= stop_at:
        return "deadline"
    return None


class WindowClosed(Exception):
    pass


def _fatal_http(exc):
    match = re.search(r"\bHTTP (401|402)(?!\d)", str(exc))
    if match:
        return "http_" + match.group(1)
    cause = getattr(exc, "__cause__", None)
    if getattr(cause, "code", None) in (401, 402):
        return "http_" + str(cause.code)
    return None

def _execute(job, client, model, start_at=None, stop_at=None):
    stock, role, index, value, path, agent = job
    if _window_status(start_at, stop_at):
        raise WindowClosed("不在允许运行的时间范围内。")
    response = agent.analyze(value, client=client, model=model)
    if not isinstance(response, str) or not response.strip():
        raise ValueError("模型未返回有效文本。")
    result = {"index": index, "response": response}
    if role == "news":
        result["text"] = value
    _write_new(path, result)
    return {"stock": stock, "role": role, "index": index}


def run_batch(stocks, prediction_date, *, client, model="deepseek-flash", roles=("news", "kline"), concurrency=5, data_root="data", output_root="outputs/analysts_with_reasons", start_at=None, stop_at=None):
    """独立任务共享同一并发上限；失败不阻碍其他任务且不自动重试。"""
    day = date.fromisoformat(prediction_date).isoformat()
    start_at, stop_at = _boundary(start_at), _boundary(stop_at)
    if start_at is not None and stop_at is not None and start_at >= stop_at:
        raise ValueError("开始时间必须早于停止时间。")
    status = _window_status(start_at, stop_at)
    if status:
        return {"submitted": 0, "saved": 0, "skipped": 0, "failed": 0,
                "failures": [], "stop_reason": status, "remaining": None}
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("并发数必须是正整数。")
    if not model or not stocks or not roles or any(r not in ("news", "kline") for r in roles):
        raise ValueError("必须指定股票、模型和有效的分析类型。")
    if any(not isinstance(s, str) or not s for s in stocks):
        raise ValueError("股票代码必须为非空字符串。")
    unique_stocks = list(dict((s.casefold(), s) for s in stocks).values())
    unique_roles = list(dict.fromkeys(roles))
    jobs, failures, skipped = [], [], 0
    # 准备数据、绘图和初始化文件顺序执行，避免 Matplotlib 线程竞争。
    for stock in unique_stocks:
        for role in unique_roles:
            status = _window_status(start_at, stop_at)
            if status:
                return {"submitted": 0, "saved": 0, "skipped": skipped, "failed": len(failures),
                        "failures": failures, "stop_reason": status, "remaining": None}
            try:
                prepared, already = _prepare(stock, day, role, client, model, data_root, output_root)
                jobs.extend(prepared); skipped += already
            except Exception as exc:
                failures.append({"stock": stock, "role": role, "stage": "prepare", "error": f"{type(exc).__name__}: {exc}"})
    executor = ThreadPoolExecutor(max_workers=concurrency)
    waiting = iter(jobs)
    active = {}
    submitted = completed = expired = 0
    stop_reason = None

    def fill_slots():
        nonlocal submitted, stop_reason
        while len(active) < concurrency and stop_reason is None:
            status = _window_status(start_at, stop_at)
            if status:
                stop_reason = status
                return
            job = next(waiting, None)
            if job is None:
                return
            active[executor.submit(_execute, job, client, model, start_at, stop_at)] = job
            submitted += 1

    try:
        fill_slots()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    future.result(); completed += 1
                except WindowClosed:
                    expired += 1
                    stop_reason = stop_reason or _window_status(start_at, stop_at) or "deadline"
                except Exception as exc:
                    failures.append({"stock": job[0], "role": job[1], "index": job[2], "stage": "request", "error": f"{type(exc).__name__}: {exc}"})
                    stop_reason = stop_reason or _fatal_http(exc)
            fill_slots()
    except KeyboardInterrupt:
        for future in active:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return {"submitted": submitted, "saved": completed, "skipped": skipped, "failed": len(failures),
            "failures": failures, "stop_reason": stop_reason, "remaining": len(jobs) - submitted + expired}


def main():
    parser = argparse.ArgumentParser(description="并发运行多个股票的 News 和 K-line。")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stocks", nargs="+")
    selection.add_argument("--all", action="store_true", help="处理本地去敏新闻目录中的全部股票")
    parser.add_argument("--date", required=True, help="预测日 YYYY-MM-DD")
    parser.add_argument("--roles", nargs="+", choices=("news", "kline"), default=["news", "kline"])
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--start-at", help="允许运行的起始时间，ISO 格式并包含时区")
    parser.add_argument("--stop-at", help="停止提交请求的时间，ISO 格式并包含时区")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/analysts_with_reasons")
    args = parser.parse_args()
    _load_config()
    client = ModelClient.from_environment()
    stocks = args.stocks if not args.all else sorted(p.name for p in (Path(args.data_root) / "US" / "news" / "masked").iterdir() if p.is_dir())
    result = run_batch(stocks, args.date, client=client, model=args.model, roles=args.roles,
                       concurrency=args.concurrency, data_root=args.data_root, output_root=args.output_root,
                       start_at=args.start_at, stop_at=args.stop_at)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failed"] or result["stop_reason"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
