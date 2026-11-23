"""新闻智能体：逐条分析已去敏的新闻标题。"""
from model_client import ModelClient

PROMPT = """As a stock trading news analyst, you are a helpful and accurate assistant. Your task is to analyze the relevance of news to the given stock, its sentiment polarity and importance, its impact on the stock price, and the duration of that impact."""

INSTRUCTION = """Please analyze the provided news about the given stock along the following five dimensions:
1. Relevance to the given stock: Rate relevance from 0 to 10. A higher score indicates that the news is more relevant to the given stock.
2. Sentiment polarity: Rate sentiment polarity from 0 to 10. A score closer to 0 indicates a stronger negative impact, while a score closer to 10 indicates a stronger positive impact.
3. Importance of the news event: Rate its importance from 0 to 10. A higher score indicates a more important event.
4. Impact on the stock price: Rate the degree of impact from 0 to 10. A higher score indicates a greater impact on the stock price.
5. Duration of the news impact: Rate the duration from 0 to 10. A higher score indicates that the news may have an impact for a longer period.

If you find the news difficult to analyze, avoid assigning zero to every dimension where possible. Instead, try to analyze the text and provide scores accordingly. Assign zero to every dimension only when analysis is truly impossible.
After the scores, provide one or two concise sentences explaining the assessment using only information in the headline. Do not invent missing details or provide step-by-step reasoning. Use the following format.

Output format:
Relevance: <score for relevance to the stock>
Sentiment polarity: <score for sentiment polarity>
Importance: <score for importance of the news event>
Impact: <score for impact on the stock price>
Duration: <score for duration of the news impact>
Reason: <one or two concise sentences>"""


def analyze(news, *, client: ModelClient, model: str):
    """逐条调用模型；只发送去敏标题，不发送内部标的和真实日期。

    兼容单条文本输入并返回文本；输入列表时，返回与输入顺序一致的原始响应列表。
    列表元素可为文本或含 text 的 masked 记录。调用方负责确保传入的是已去敏标题。
    """
    if news is None:
        raise ValueError("新闻输入尚未提供。")
    single = isinstance(news, str)
    items = [news] if single else news
    if not isinstance(items, list):
        raise TypeError("新闻应为单条文本或标题列表。")

    responses = []
    for item in items:
        headline = item.get("text") if isinstance(item, dict) else item
        if not isinstance(headline, str) or not headline.strip():
            raise ValueError("新闻标题必须是非空文本。")
        content = f"{INSTRUCTION}\n\n[Stock]: Company A\n[News]: {headline}\n[Publication time]: Previous day"
        responses.append(client.complete(model=model, prompt=PROMPT, content=content))
    return responses[0] if single else responses
