"""K 线智能体：按已给定的六个维度分析去标识图像。"""
import base64

from model_client import ModelClient

PROMPT = """As a financial data analysis assistant, you are helpful and accurate. Your task is to analyze the provided candlestick chart and assess indicators associated with any long-term trend shown in the chart and their potential impact."""

INSTRUCTION = """Please analyze the trend information that may be shown in the candlestick chart using the provided input. Assess the following six dimensions:
1. Trend direction: Rate the long-term trend from 0 to 10. A score closer to 0 indicates a stronger long-term downward trend, near 5 indicates a relatively unclear trend, and closer to 10 indicates a stronger long-term upward trend.
2. Trend duration: Rate how long the trend has lasted within the chart from 0 to 10. A higher score indicates the trend has lasted longer.
3. Trend persistence: Rate how long the trend may continue beyond the chart from 0 to 10. A higher score indicates it may persist for longer.
4. Trend strength: Rate the strength of the trend signal from 0 to 10. A higher score indicates a stronger signal that may affect the stock price.
5. Distance to support: Rate the distance to the stock's support level from 0 to 10. A score closer to 0 indicates a nearer support level, while a score closer to 10 indicates a more distant support level.
6. Distance to resistance: Rate the distance to the stock's resistance level from 0 to 10. A score closer to 0 indicates a nearer resistance level, while a score closer to 10 indicates a more distant resistance level.

If the chart is difficult to analyze, make a reasonable assessment based on the chart rather than simply assigning zero to all dimensions.
After the scores, provide one or two concise sentences explaining the observed trend or pattern on which your assessment is based. Do not invent information absent from the chart or provide step-by-step reasoning. Use the following format.

Output format:
Trend direction: <score for the long-term trend direction>
Trend duration: <score for the trend's duration within the chart>
Trend persistence: <score for the trend's persistence beyond the chart>
Trend strength: <score for the trend signal's strength>
Distance to support: <score for the distance to the support level>
Distance to resistance: <score for the distance to the resistance level>
Reason: <one or two concise sentences>"""


def analyze(chart_content, *, client: ModelClient, model: str) -> str:
    """接收 PNG 字节或调用方已构造的多模态消息，返回模型原文。"""
    if chart_content is None:
        raise ValueError("K 线输入尚未提供。")
    task = f"{INSTRUCTION}\n\n[Stock]: Company A\n[Time]: Trading days before the prediction date"
    if isinstance(chart_content, bytes):
        if not chart_content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("K 线图片必须是 PNG 格式。")
        encoded = base64.b64encode(chart_content).decode("ascii")
        chart_content = [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    elif isinstance(chart_content, list):
        chart_content = [{"type": "text", "text": task}, *chart_content]
    elif isinstance(chart_content, str):
        chart_content = task + "\n\n" + chart_content
    return client.complete(model=model, prompt=PROMPT, content=chart_content)
