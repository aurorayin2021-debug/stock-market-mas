"""关联智能体：读取 Skill，通过模型工具调用获取并解读关系证据。"""
from pathlib import Path
import json

from model_client import ModelClient, text_content
from tools.relation_tool import TOOL, get_relation_evidence

PROMPT = """As a stock relationship analyst, you are a helpful and accurate assistant. Your task is to analyze the historical relationships between the given stock and related stocks, as well as the recent returns of those related stocks."""

INSTRUCTION = """Please provide one or two concise sentences explaining the relationship evidence for Company A. Consider the selected stocks' historical lagged correlations and their recent returns, including missing values when relevant. The relationship score and ranking do not indicate positive or negative direction; use the sign of the lagged correlation when describing historical association.

Use only the provided evidence. Do not invent details or guess the identities of the stocks or dates. Historical relationships do not establish causation or guarantee future effects. Do not predict Company A's price direction or provide step-by-step reasoning.

The relationship table is provided separately by the tool; do not repeat or modify it. Use the following format for your explanation:
Reason: <one or two concise sentences>"""

SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "transfer_entropy" / "SKILL.md"


def analyze(relation_evidence, *, client: ModelClient, model: str) -> dict:
    """兼容已有匿名证据的直接解读入口；调用方负责先匿名化。"""
    if relation_evidence is None:
        raise ValueError("关联关系证据尚未提供。")
    skill = SKILL_PATH.read_text(encoding="utf-8")
    summary = client.complete(model=model, prompt=PROMPT + "\n\n" + skill, content=INSTRUCTION + "\n\n[Stock]: Company A\n[Relationship evidence]: " + text_content(relation_evidence))
    return {"relations": relation_evidence, "analysis": summary}


def analyze_with_tool(task, *, client: ModelClient, model: str) -> dict:
    """由模型发起工具调用；真实代码与日期只交给本地工具，不发给模型。

    task 需含 price_root、stock、prediction_date；estimator 使用已确认的作者代码方法。
    """
    if not isinstance(task, dict):
        raise TypeError("关联分析需要本地任务配置。")
    required = ("price_root", "stock", "prediction_date", "estimator")
    if any(not task.get(key) for key in required):
        raise ValueError("关联分析缺少行情目录、股票、预测日或已确认的计算方式。")
    if task["estimator"] != "author_code":
        raise ValueError("关联计算仅支持作者代码版本 author_code。")
    skill = SKILL_PATH.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": PROMPT + "\n\n" + skill},
        {"role": "user", "content": INSTRUCTION + "\n\n[Stock]: Company A\nUse the tool to retrieve relationship evidence."},
    ]
    turn = client.chat(model=model, messages=messages, tools=[TOOL], tool_choice="auto")
    calls = turn.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "get_relation_evidence":
        raise ValueError("模型未正确调用关联关系工具，请检查模型的工具调用能力。")
    call = calls[0]
    try:
        arguments = json.loads(call["function"].get("arguments", "{}"))
    except (ValueError, TypeError) as error:
        raise ValueError("工具调用参数格式错误。") from error
    if arguments != {} or not isinstance(call.get("id"), str) or not call["id"]:
        raise ValueError("关联工具不接收模型提供的参数，且需要有效调用 ID。")
    evidence = get_relation_evidence(
        task["price_root"], task["stock"], task["prediction_date"],
        estimator=task["estimator"],
    )
    messages.append(turn)
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": text_content(evidence)})
    final = client.chat(model=model, messages=messages, tools=[TOOL], tool_choice="none")
    summary = final.get("content")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("模型没有返回关联关系分析。")
    return {"relations": evidence, "analysis": summary}
