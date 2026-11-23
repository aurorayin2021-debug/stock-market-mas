"""只读检查 News/K-line 结果格式；不调用模型、不修改结果。"""
import argparse
import json
import re
from pathlib import Path

FIELDS = {
    "news": ("Relevance", "Sentiment polarity", "Importance", "Impact", "Duration"),
    "kline": ("趋势方向", "趋势时长", "趋势惯性", "趋势强度", "支撑位距离", "阻力位距离"),
}
REASON = {"news": "Reason", "kline": "简短理由"}
KLINE_ENGLISH_LABELS = {
    "trend direction": "趋势方向",
    "trend duration": "趋势时长",
    "trend persistence": "趋势惯性",
    "trend strength": "趋势强度",
    "distance to support": "支撑位距离",
    "distance to resistance": "阻力位距离",
    "reason": "简短理由",
}


def check_response(response, role):
    """返回需复核原因；空列表仅表示格式通过，不代表判断正确。"""
    if role not in FIELDS:
        raise ValueError("未知分析类型。")
    if not isinstance(response, str) or not response.strip():
        return ["模型回答为空"]
    labels = {x.casefold(): [] for x in (*FIELDS[role], REASON[role])}
    lines = response.splitlines()
    reason_line = None
    for index, line in enumerate(lines):
        cleaned = line.strip().replace("**", "").replace("__", "")
        cleaned = re.sub(r"^(?:[-*•]\s+|\d+[.)、]\s*)", "", cleaned)
        match = re.match(r"^([^:：]+)\s*[:：]\s*(.*)$", cleaned)
        if not match:
            continue
        label = match.group(1).strip().casefold()
        if role == "kline":
            label = KLINE_ENGLISH_LABELS.get(label, label)
        if label in labels:
            labels[label].append(match.group(2).strip())
            if label == REASON[role].casefold():
                reason_line = index
    issues = []
    for name in FIELDS[role]:
        values = labels[name.casefold()]
        if len(values) != 1:
            issues.append(f"{name} 缺失或重复")
            continue
        match = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*/\s*10)?(?:\s*(?:$|[（(—-]))", values[0])
        if not match or not 0 <= float(match.group(1)) <= 10:
            issues.append(f"{name} 非法或不在 0–10")
    reasons = labels[REASON[role].casefold()]
    if len(reasons) != 1:
        issues.append("简短理由缺失或重复")
    elif not reasons[0]:
        continuation = []
        for line in lines[reason_line + 1:]:
            cleaned = line.strip().replace("**", "").replace("__", "")
            cleaned = re.sub(r"^(?:[-*•]\s+|\d+[.)、]\s*)", "", cleaned)
            match = re.match(r"^([^:：]+)\s*[:：]", cleaned)
            label = match.group(1).strip().casefold() if match else ""
            if role == "kline":
                label = KLINE_ENGLISH_LABELS.get(label, label)
            if label in labels:
                break
            if cleaned:
                continuation.append(cleaned)
        if not continuation:
            issues.append("简短理由为空")
    return issues


def audit(stock_root, result_root, prediction_date):
    """按股票目录和每个 manifest 的总数检查，未发现结果不推断是否已有新闻。"""
    stock_root, result_root = Path(stock_root), Path(result_root)
    stocks = sorted(p.name for p in stock_root.iterdir() if p.is_dir())
    if not stocks:
        raise ValueError("股票目录为空或不存在。")
    result = {"passed": [], "missing": [], "review": []}
    for stock in stocks:
        for role in FIELDS:
            folder = result_root / stock / prediction_date / role
            manifest = folder / "manifest.json"
            if not manifest.is_file():
                result["missing"].append({"stock": stock, "role": role, "issue": "结果清单不存在"})
                continue
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                total = metadata["total"]
                if type(total) is not int or total < 0 or (role == "kline" and total != 1):
                    raise ValueError("任务总数无效")
            except (OSError, ValueError, TypeError, KeyError) as error:
                result["review"].append({"stock": stock, "role": role, "issue": f"清单无法读取：{error}"})
                continue
            for index in range(1, total + 1):
                path = folder / f"{index:04d}.json"
                item = {"stock": stock, "role": role, "index": index}
                if not path.is_file():
                    result["missing"].append({**item, "issue": "结果文件不存在"})
                    continue
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row.get("index") != index:
                        raise ValueError("序号不匹配")
                    issues = check_response(row.get("response"), role)
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    issues = [f"结果无法读取：{error}"]
                if issues:
                    result["review"].append({**item, "issues": issues})
                else:
                    result["passed"].append(item)
    return {"counts": {name: len(items) for name, items in result.items()}, **result}


def main():
    parser = argparse.ArgumentParser(description="只读检查分析结果格式。")
    parser.add_argument("--stocks-root", default="data/US/news/masked")
    parser.add_argument("--results-root", default="outputs/analysts_with_reasons")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.stocks_root, args.results_root, args.date), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
