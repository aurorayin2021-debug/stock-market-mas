"""三路分析汇合的最小流程；占位内容不是投资预测。"""
from __future__ import annotations
import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict
from langgraph.graph import START, END, StateGraph

class State(TypedDict, total=False):
    inputs: dict[str, Any]
    news: dict[str, Any]
    kline: dict[str, Any]
    relation: dict[str, Any]
    report: dict[str, Any]

SOURCES = ("news", "kline", "relation")
Analyzer = Callable[[Any], Any]

def placeholder(source: str) -> Analyzer:
    def analyze(data: Any) -> str:
        return f"【占位】{source} 尚未接入真实数据、分析方法或模型。"
    return analyze

def collect(state: State) -> dict[str, Any]:
    """目前仅保存三路结果，绝不生成涨跌判断。"""
    items = {name: state[name] for name in SOURCES if name in state}
    missing = [name for name in SOURCES if name not in items]
    return {"report": {
        "analyses": items, "missing": missing,
        "status": "incomplete" if missing else "collected",
        "prediction": None, "note": "仅演示汇合，未实现预测。"
    }}

def build_graph(analyzers: Mapping[str, Analyzer] | None = None, *, placeholder_results: bool = True, manager_fn: Callable[[dict[str, Any]], Any] | None = None, require_inputs: bool = False):
    """注入替代函数；节点只接收本节点的输入，内部格式可调整。"""
    supplied = dict(analyzers or {})
    unknown = supplied.keys() - set(SOURCES)
    if unknown:
        raise ValueError(f"未知节点: {sorted(unknown)}")
    graph = StateGraph(State)
    if require_inputs:
        def validate_inputs(state: State):
            inputs = state.get("inputs", {})
            missing = [name for name in SOURCES if inputs.get(name) is None]
            if missing:
                raise ValueError(f"缺少输入: {', '.join(missing)}")
            return {}
        graph.add_node("validate_inputs", validate_inputs)
        graph.add_edge(START, "validate_inputs")
    for name in SOURCES:
        fn = supplied.get(name, placeholder(name))
        marked = name not in supplied or placeholder_results
        def node(state: State, *, analyze=fn, source=name, is_placeholder=marked):
            content = analyze(state.get("inputs", {}).get(source))
            json.dumps(content, ensure_ascii=False, allow_nan=False)
            return {source: {"placeholder": is_placeholder, "content": content}}
        node_name = f"analyze_{name}"
        graph.add_node(node_name, node)
        graph.add_edge("validate_inputs" if require_inputs else START, node_name)
    def manager_node(state: State):
        result = collect(state)
        if manager_fn is None:
            return result
        reports = result["report"]["analyses"]
        if result["report"]["missing"] or any(item["placeholder"] for item in reports.values()):
            raise ValueError("分析结果缺失或仍为占位，不能调用 Manager。")
        contents = {name: reports[name]["content"] for name in SOURCES}
        result["report"]["manager_result"] = manager_fn(contents)
        result["report"]["status"] = "manager_completed"
        result["report"]["note"] = "Manager 已返回原始内容；输出格式和预测定义仍待确定。"
        return result
    graph.add_node("manager", manager_node)
    graph.add_edge([f"analyze_{name}" for name in SOURCES], "manager")
    graph.add_edge("manager", END)
    return graph.compile()

def main() -> None:
    parser = argparse.ArgumentParser(description="演示占位流程，不生成预测。")
    parser.add_argument("--output", type=Path, help="可选：保存结果到新文件，父目录须已存在")
    args = parser.parse_args()
    result = build_graph().invoke({"inputs": {}})["report"]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
