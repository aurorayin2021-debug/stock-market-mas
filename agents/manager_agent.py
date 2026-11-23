"""Manager: synthesize the three analyst reports for next-day prediction."""
from model_client import ModelClient, text_content

PROMPT = """You are the manager of a stock analysis team. Your task is to synthesize the news, candlestick, and cross-stock relationship analysts' reports and predict whether Company A's closing price on the next trading day will rise or fall relative to the previous trading day's close."""

INSTRUCTION = """Consider the available results and reasons from all three analysts according to the meaning of each measure:

1. News: Consider each news report's relevance, sentiment polarity, importance, impact magnitude, duration, and reason. A sentiment score near 0 indicates a negative impact and near 10 a positive impact; the other scores are not directional. An empty list means no news report was available, not a neutral signal.
2. Candlestick: Consider the trend direction, duration, persistence, strength, distance to support, distance to resistance, and reason. A trend-direction score near 0 indicates a downward trend, near 5 an unclear trend, and near 10 an upward trend. The other scores describe different aspects and are not themselves directional.
3. Relationships: Consider the complete relationship table and the analyst's reason, including the historical ranking and score, lagged correlations, and recent peer returns. The score is based on the author's entropy-difference calculation; it is not standard transfer entropy and does not itself indicate positive or negative price movement or causation. The sign of a lagged correlation describes historical linear association, which does not guarantee a future effect. Do not assume anonymized labels in different reports refer to the same company.

Use only the supplied reports. Do not guess identities, actual dates, missing information, future prices, or realized outcomes. Do not invent scores or numerical relationships. Consider agreement or disagreement across the evidence. If a report is unavailable, use the remaining information without treating missing information as neutral. If the available evidence is genuinely insufficient to support a judgment, output INSUFFICIENT; otherwise output UP or DOWN.

Use the following format:
Prediction: <UP, DOWN, or INSUFFICIENT>
Reason: <one or two concise sentences explaining the decision and any relevant disagreement>"""


def analyze(analyses, *, client: ModelClient, model: str) -> str:
    """Send the anonymized analyst reports and return the Manager's response."""
    if analyses is None:
        raise ValueError("分析报告尚未提供。")
    content = INSTRUCTION + "\n\n[Analyst reports]: " + text_content(analyses)
    return client.complete(model=model, prompt=PROMPT, content=content)
