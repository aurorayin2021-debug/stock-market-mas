"""基于本地收盘价准备匿名关联证据；调用者控制目标及预测日期。"""
from __future__ import annotations
import csv
from datetime import date
import math
from pathlib import Path
from skills.transfer_entropy.scripts.compute_relations import author_code_score, lagged_correlation

# 提供给模型的工具说明；真实股票和日期由程序绑定，不作为工具参数暴露。
TOOL = {"type": "function", "function": {
    "name": "get_relation_evidence",
    "description": "Retrieve anonymized historical cross-stock relationships and relevant peer returns.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}}


def _read_prices(path, *, prediction_date):
    prices = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Date", "Close"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} 缺少 Date 或 Close。")
        for row in reader:
            day = date.fromisoformat(row["Date"])
            if day >= prediction_date:
                continue
            if day in prices:
                raise ValueError(f"{path.name} 有重复日期。")
            value = float(row["Close"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path.name} 有无效收盘价。")
            prices[day] = value
    return prices


def run(price_dir, target, prediction_date, *, estimator="author_code",
        history_start="2018-01-01", history_end="2020-12-31", lag=1, top_k=30):
    """返回 (匿名证据, 私有映射)；私有映射不得发送给 LLM。

    股票池为目录内 CSV；联合历史窗口按 110 股票共同日期对齐。
    只处理目标日前价格，任何缺失或不足报错，不填充和越期。
    关联指标复现作者代码中的联合熵差，不等同于标准转移熵。
    """
    if estimator != "author_code":
        raise ValueError("关联指标仅支持作者代码版本 author_code。")
    if lag != 1:
        raise ValueError("当前仅支持用户指定的一交易日滞后。")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k 必须是正整数。")
    prediction = date.fromisoformat(prediction_date)
    start, end = date.fromisoformat(history_start), date.fromisoformat(history_end)
    if start > end or end >= prediction:
        raise ValueError("固定估计窗口必须结束于预测日之前。")
    root = Path(price_dir)
    paths = sorted(root.glob("*.csv"))
    if not paths or not isinstance(target, str) or target not in {p.stem for p in paths}:
        raise ValueError("目标股票不在本地价格文件的候选池中。")
    prices = {p.stem: _read_prices(p, prediction_date=prediction) for p in paths}
    common = set.intersection(*(set(prices[s]) for s in prices))
    training_days = sorted(d for d in common if start <= d <= end)
    if len(training_days) < 4:
        raise ValueError("共同历史价格不足以估计关联。")
    all_days = set.union(*(set(prices[s]) for s in prices))
    if training_days != sorted(d for d in all_days if training_days[0] <= d <= training_days[-1]):
        raise ValueError("共同历史交易日存在内部缺口，需先确定缺失值策略。")
    returns = {s: [prices[s][d] / prices[s][p] - 1 for p, d in zip(training_days, training_days[1:])]
               for s in prices}
    # 以候选池交易日期并集定位最后两天；目标缺失时不能回退到更早日期。
    market_days = sorted(all_days)
    if len(market_days) < 2:
        raise ValueError("市场缺少预测前两日行情。")
    previous, latest = market_days[-2:]
    if previous not in prices[target] or latest not in prices[target]:
        raise ValueError("目标股票缺少最近两个市场交易日的收盘价。")
    candidates = []
    for symbol in sorted(prices):
        if symbol == target:
            continue
        try:
            score = author_code_score(returns[symbol], returns[target], lag=lag)
            corr = lagged_correlation(returns[symbol], returns[target], lag=lag)
        except ValueError:
            continue
        candidates.append((symbol, score, corr))
    if len(candidates) < top_k:
        raise ValueError(f"有效关联候选不足: {len(candidates)}/{top_k}。")
    # 分数从大到小；同分按原代码名稳定打破平局，不暴露给模型。
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = candidates[:top_k]
    peers, private_mapping = [], {}
    for rank, (symbol, score, corr) in enumerate(selected, 1):
        alias = f"Stock {rank}"
        private_mapping[alias] = symbol
        recent = (prices[symbol][latest] / prices[symbol][previous] - 1
                  if previous in prices[symbol] and latest in prices[symbol] else None)
        peers.append({"stock": alias, "rank": rank, "author_code_score": score,
                      "lagged_correlation": corr, "lag_trading_days": lag,
                      "previous_trading_day_return": recent})
    evidence = {"target": "Company A", "method": "author_code_joint_entropy_difference",
                "note": "Author-code score is a difference of joint entropies, not standard transfer entropy; it does not indicate price direction or establish causality.",
                "related_stocks": peers}
    return evidence, private_mapping


def get_relation_evidence(price_dir, target, prediction_date, *, estimator="author_code"):
    """只向 LLM 返回匿名证据；可使用 run 在本地查看私有映射。"""
    evidence, _private_mapping = run(price_dir, target, prediction_date, estimator=estimator)
    return evidence
