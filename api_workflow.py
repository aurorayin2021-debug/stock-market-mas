"""模型调用工作流；运行前检查配置和关联计算方式。"""
import os
from functools import partial

from agents import news_agent, kline_agent, relation_agent, manager_agent
from model_client import ModelClient
from stock_mas import build_graph


def build_api_graph():
    """确认模型配置并使用已确定的作者代码指标。"""
    agents = {
        "news": (news_agent, "NEWS_MODEL"),
        "kline": (kline_agent, "KLINE_MODEL"),
        "relation": (relation_agent, "RELATION_MODEL"),
        "manager": (manager_agent, "MANAGER_MODEL"),
    }
    missing = []
    for name, (module, model_var) in agents.items():
        if not module.PROMPT.strip():
            missing.append(f"{name} 的 PROMPT")
        if not os.getenv(model_var, "").strip():
            missing.append(model_var)
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY"):
        if not os.getenv(key, "").strip():
            missing.append(key)
    estimator = os.getenv("RELATION_ESTIMATOR", "author_code").strip() or "author_code"
    if estimator != "author_code":
        missing.append("RELATION_ESTIMATOR（须为 author_code）")
    if missing:
        raise ValueError("请先填写配置：" + "、".join(missing))
    client = ModelClient.from_environment()
    functions = {
        name: partial(module.analyze, client=client, model=os.environ[model_var])
        for name, (module, model_var) in agents.items() if name != "relation"
    }
    def relation(task):
        if not isinstance(task, dict):
            raise TypeError("关联分析需要本地任务配置。")
        return relation_agent.analyze_with_tool(
            {**task, "estimator": estimator}, client=client, model=os.environ["RELATION_MODEL"]
        )
    return build_graph(
        analyzers={"news": functions["news"], "kline": functions["kline"], "relation": relation},
        placeholder_results=False, manager_fn=functions["manager"], require_inputs=True,
    )
