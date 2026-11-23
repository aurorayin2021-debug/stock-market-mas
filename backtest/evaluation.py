"""离线比较 Manager 预测与实际结果；参考答案不进入 Agent 输入。"""
import csv
import re


def evaluate_predictions(predictions: dict[str, str], labels_file) -> dict:
    """predictions 映射证券代码到 Manager 原文；统计可评估的方向准确率。"""
    with open(labels_file, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = {}
    for row in rows:
        symbol = row["stock"].casefold()
        if symbol in labels:
            raise ValueError("参考答案存在重复证券代码。")
        labels[symbol] = row
    counts = {"correct": 0, "evaluated": 0, "abstained": 0, "invalid": 0, "missing_label": 0}
    pattern = re.compile(r"^Prediction:\s*(UP|DOWN|INSUFFICIENT)\s*$", re.IGNORECASE | re.MULTILINE)
    for symbol, output in predictions.items():
        if not isinstance(symbol, str) or not isinstance(output, str):
            raise TypeError("预测结果应为股票代码到文本的映射。")
        matches = pattern.findall(output)
        if len(matches) != 1:
            counts["invalid"] += 1
            continue
        prediction = matches[0].upper()
        if prediction == "INSUFFICIENT":
            counts["abstained"] += 1
            continue
        label = labels.get(symbol.casefold())
        if not label or label.get("status") != "ok" or label.get("direction", "").upper() not in ("UP", "DOWN"):
            counts["missing_label"] += 1
            continue
        counts["evaluated"] += 1
        counts["correct"] += prediction == label["direction"].upper()
    counts["accuracy"] = counts["correct"] / counts["evaluated"] if counts["evaluated"] else None
    return counts
