"""Resume per-stock Relation and Manager jobs, then update local reference labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time

from agents import manager_agent, relation_agent
from dataflows.analyst_cache import load_analyst_cache
from model_client import ModelClient
from run_analysts import _load_config, _write_new, _write_bytes

MODEL = "deepseek-flash"
EXTRA = ("prediction", "prediction_reason", "prediction_status", "comparison")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _signature(cached_provenance, model, price_fingerprint, client):
    import tools.relation_tool as relation_tool
    import skills.transfer_entropy.scripts.compute_relations as compute_relations
    modules = (relation_agent, manager_agent, relation_tool, compute_relations)
    code_hashes = {module.__name__: _digest(Path(module.__file__).read_bytes()) for module in modules}
    text = {"model": model, "cache": cached_provenance, "prices": price_fingerprint,
            "provider": _digest(getattr(client, "base_url", "").encode("utf-8")), "code": code_hashes,
            "relation_prompt": relation_agent.PROMPT, "relation_instruction": relation_agent.INSTRUCTION,
            "relation_skill": relation_agent.SKILL_PATH.read_text(encoding="utf-8"),
            "manager_prompt": manager_agent.PROMPT, "manager_instruction": manager_agent.INSTRUCTION}
    return _digest(json.dumps(text, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def price_digest(price_root):
    digest = hashlib.sha256()
    for path in sorted(Path(price_root).glob("*.csv"), key=lambda p: p.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_result(path, stock, day, signature):
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("stock"), value.get("prediction_date"), value.get("signature")) != (stock, day, signature):
        raise ValueError(f"Saved result does not match the current run: {path}")
    return value


def _retry(call, attempts=3):
    for number in range(attempts):
        try:
            return call()
        except RuntimeError as error:
            msg = str(error)
            if not (re.search(r"HTTP (429|5[0-9][0-9])\b", msg) or "连接模型 API 失败" in msg):
                raise
            if number + 1 == attempts:
                raise
            time.sleep(2 ** number)


def _parse(text):
    matches = re.findall(r"^Prediction:\s*(UP|DOWN|INSUFFICIENT)\s*$", text, re.I | re.M)
    reason = re.search(r"^Reason:\s*(.*(?:\n(?!Prediction:).*)*)", text, re.I | re.M)
    return (matches[0].upper() if len(matches) == 1 else "", reason.group(1).strip() if reason else "")


def process(stock, day, *, data_root, cache_root, output_root, model, price_fingerprint, client):
    cached, provenance = load_analyst_cache(cache_root, stock, day)
    price_root = Path(data_root) / "CMIN-Dataset/CMIN-US/price/raw"
    matches = [p for p in price_root.glob("*.csv") if p.stem.casefold() == stock.casefold()]
    if len(matches) != 1:
        raise ValueError("No unique matching price file.")
    signature = _signature(provenance, model, price_fingerprint, client)
    directory = Path(output_root) / stock / day
    directory.mkdir(parents=True, exist_ok=True)
    relation_file, manager_file = directory / "relation" / "0001.json", directory / "manager" / "0001.json"
    if relation_file.exists():
        relation = _read_result(relation_file, stock, day, signature)["result"]
    else:
        task = {"stock": matches[0].stem, "price_root": str(price_root), "prediction_date": day, "estimator": "author_code"}
        relation = _retry(lambda: relation_agent.analyze_with_tool(task, client=client, model=model))
        _write_new(relation_file, {"stock": stock, "prediction_date": day, "signature": signature, "model": model, "result": relation})
    if manager_file.exists():
        return _read_result(manager_file, stock, day, signature)
    reports = {**cached, "relation": relation}
    response = _retry(lambda: manager_agent.analyze(reports, client=client, model=model))
    prediction, reason = _parse(response)
    value = {"stock": stock, "prediction_date": day, "signature": signature, "model": model,
             "response": response, "prediction": prediction, "reason": reason, "cache_provenance": provenance}
    _write_new(manager_file, value)
    return value


def validate_existing(output_root, cache_root, day, model, price_fingerprint, client, eligible):
    """Refuse to mix prior stages from a different run before updating labels."""
    allowed = {symbol.casefold(): symbol for symbol in eligible}
    for directory in Path(output_root).iterdir():
        if not directory.is_dir():
            continue
        relation_file = directory / day / "relation" / "0001.json"
        manager_file = directory / day / "manager" / "0001.json"
        if not relation_file.exists() and not manager_file.exists():
            continue
        stock = allowed.get(directory.name.casefold())
        if stock is None or directory.name != stock:
            raise ValueError(f"Unexpected saved stock in output directory: {directory.name}. Use a new output directory.")
        if manager_file.exists() and not relation_file.exists():
            raise ValueError(f"Saved manager result has no relation result: {stock}. Use a new output directory.")
        _, provenance = load_analyst_cache(cache_root, stock, day)
        signature = _signature(provenance, model, price_fingerprint, client)
        for path in (relation_file, manager_file):
            if path.exists():
                try:
                    _read_result(path, stock, day, signature)
                except (ValueError, KeyError, json.JSONDecodeError) as error:
                    raise ValueError(f"Existing result for {stock} belongs to a different or damaged run. Use a new output directory.") from error


def select_stocks(rows, stock=None):
    available = [r["stock"] for r in rows if r.get("source_coverage") == "matched"]
    if len({s.casefold() for s in available}) != len(available):
        raise ValueError("Duplicate stocks in labels.")
    if stock is None:
        return available, available
    matches = [s for s in available if s.casefold() == stock.casefold()]
    if len(matches) != 1:
        raise ValueError("Stock is not in the matched cache/price sample.")
    return available, matches


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def prepare_backup(labels):
    labels = Path(labels)
    backup = labels.with_name(labels.stem + ".pre_predictions.bak" + labels.suffix)
    if not backup.exists():
        _write_bytes(backup, labels.read_bytes())
    fields, base = _read_csv(backup)
    current_fields, current = _read_csv(labels)
    if any(f in fields for f in EXTRA):
        raise ValueError("Reference backup already contains prediction columns.")
    if len(base) != len(current) or any(f not in current_fields for f in fields):
        raise ValueError("Reference labels differ from backup.")
    for old, now in zip(base, current):
        if any(old[f] != now[f] for f in fields):
            raise ValueError("Reference labels differ from backup.")
    return backup, fields, base


def update_labels(labels, backup, fields, base, output_root, day):
    current_fields, current = _read_csv(labels)
    if len(current) != len(base) or any(f not in current_fields for f in fields):
        raise ValueError("Reference labels changed while running.")
    if any(any(a[f] != b[f] for f in fields) for a, b in zip(base, current)):
        raise ValueError("Reference labels changed while running.")
    stats = {"completed": 0, "evaluated": 0, "correct": 0, "abstained": 0, "invalid": 0}
    for row in current:
        for key in EXTRA:
            row[key] = ""
        symbol = row["stock"]
        path = Path(output_root) / symbol / day / "manager" / "0001.json"
        if not path.is_file():
            if row.get("source_coverage") != "matched":
                row["prediction_status"] = "not_run"
            else:
                row["prediction_status"] = "failed" if (Path(output_root) / symbol / day / "errors.jsonl").is_file() else "pending"
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("stock") != symbol or value.get("prediction_date") != day:
            raise ValueError("Prediction file does not match label row.")
        pred, reason = _parse(value.get("response", ""))
        row["prediction"], row["prediction_reason"] = pred, reason
        stats["completed"] += 1
        if pred == "INSUFFICIENT":
            row["prediction_status"] = "abstained"; stats["abstained"] += 1
        elif pred not in ("UP", "DOWN"):
            row["prediction_status"] = "invalid"; stats["invalid"] += 1
        elif row.get("status") == "ok" and row.get("direction", "").upper() in ("UP", "DOWN"):
            row["prediction_status"] = "evaluated"; stats["evaluated"] += 1
            correct = pred == row["direction"].upper()
            row["comparison"] = "correct" if correct else "incorrect"
            stats["correct"] += correct
        else:
            row["prediction_status"] = "missing_label"
    stats["accuracy"] = stats["correct"] / stats["evaluated"] if stats["evaluated"] else None
    names = fields + [f for f in current_fields if f not in fields and f not in EXTRA] + list(EXTRA)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=names, extrasaction="raise")
    writer.writeheader(); writer.writerows(current)
    fd, tmp = tempfile.mkstemp(prefix=".labels-", suffix=".tmp", dir=Path(labels).parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            stream.write(output.getvalue()); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, labels)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run cached analyst inputs, saving each stage and updating reference labels.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--stock", help="Run one stock using the same output directory for later resumption")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-root", default="outputs/analysts_with_reasons")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    day = date.fromisoformat(args.date).isoformat()
    if args.workers < 1 or args.workers > 32: parser.error("workers must be between 1 and 32")
    labels = Path(args.data_root) / "US/labels" / (day + ".csv")
    output = Path(args.output_root or "outputs/analysts_with_reasons")
    output.mkdir(parents=True, exist_ok=True)
    backup, fields, base = prepare_backup(labels)
    all_eligible, eligible = select_stocks(base, args.stock)
    fingerprint = price_digest(Path(args.data_root) / "CMIN-Dataset/CMIN-US/price/raw")
    _load_config()
    client = ModelClient.from_environment()
    validate_existing(output, args.cache_root, day, args.model, fingerprint, client, all_eligible)
    stats = update_labels(labels, backup, fields, base, output, day)
    print(f"Starting: {len(eligible)} selected; {stats['completed']}/{len(all_eligible)} already completed in output directory.", flush=True)
    remaining = iter(eligible)
    errors = 0
    stop = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        def submit_next():
            symbol = next(remaining, None)
            if symbol is not None:
                f = pool.submit(process, symbol, day, data_root=args.data_root, cache_root=args.cache_root,
                                output_root=output, model=args.model, price_fingerprint=fingerprint, client=client)
                futures[f] = symbol
        for _ in range(args.workers): submit_next()
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in ready:
                symbol = futures.pop(f)
                try:
                    value = f.result()
                    stats = update_labels(labels, backup, fields, base, output, day)
                    print(f"{symbol}: {value['prediction'] or 'INVALID'}; completed {stats['completed']}/{len(all_eligible)}", flush=True)
                except Exception as error:
                    errors += 1
                    msg = str(error)
                    print(f"{symbol}: FAILED {type(error).__name__}: {msg}", flush=True)
                    log = output / symbol / day / "errors.jsonl"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"error_type": type(error).__name__, "error": msg}, ensure_ascii=False) + "\n")
                    if re.search(r"HTTP (401|402)\b", msg): stop = True
                if not stop: submit_next()
    stats = update_labels(labels, backup, fields, base, output, day)
    print(json.dumps({"selected": len(eligible), "total_available": len(all_eligible), "errors": errors, **stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
