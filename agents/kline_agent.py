"""K 线智能体：按已给定的六个维度分析去标识图像。"""
import base64

from model_client import ModelClient

PROMPT = """作为一名金融数据分析助手，您是一位乐于助人且精确的助手。您的任务是通过分析提供的 K 线图，评估图形中可能体现的长期趋势的相关指标及潜在影响。"""

INSTRUCTION = """我需要您结合以上输入，分析 K 线图中可能体现的趋势信息。您需要从六个维度进行评估：
1. 趋势方向：请在 0 到 10 的范围内评估长期趋势，靠近 0 的分值表示更强的长期下跌趋势，靠近 5 的分值表示趋势相对模糊，而靠近 10 的分值表示更强的长期上涨趋势。
2. 趋势时长：请在 0 到 10 的范围内评估该趋势在图像内的持续时间，分数越高表示该趋势已经持续的时间越长。
3. 趋势惯性：请在 0 到 10 的范围内评估该趋势在图像外继续保持的时间，分数越高表示该趋势可能持续的时间越长。
4. 趋势强度：请在 0 到 10 的范围内评估信号强度，分数越高表示该趋势信号越强烈，可能影响股票价格。
5. 支撑位距离：请在 0 到 10 的范围内评估支撑位距离，靠近 0 的分值表示距离股票支撑位越近，靠近 10 的分值表示距离股票支撑位置越远。
6. 阻力位距离：请在 0 到 10 的范围内评估阻力位距离，靠近 0 的分值表示距离股票阻力位越近，靠近 10 的分值表示距离股票阻力位越远。

如无法进行分析，请在提供分数时结合图形进行合理推测，而非简单赋予所有维度零分。
请在评分后用一到两句话简要说明判断所依据的图中走势或形态；不要编造图中没有的信息，也不要展开详细推理过程。请按照以下格式提供答案。

输出格式：
趋势方向：<长期趋势方向的得分>
趋势时长：<趋势在图像内的持续时间的得分>
趋势惯性：<趋势在图像外的持续时间的得分>
趋势强度：<趋势信号强度的得分>
支撑位距离：<支撑位距离的得分>
阻力位距离：<阻力位距离的得分>
简短理由：<一到两句话>"""


def analyze(chart_content, *, client: ModelClient, model: str) -> str:
    """接收 PNG 字节或调用方已构造的多模态消息，返回模型原文。"""
    if chart_content is None:
        raise ValueError("K 线输入尚未提供。")
    task = f"{INSTRUCTION}\n\n[股票名称]：Company A\n[时间]：预测日前的交易日"
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
