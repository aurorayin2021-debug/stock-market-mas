"""对 CMIN 单个新闻文件去标识；从项目根目录用 python -m dataflows.news_masking 运行。"""
import argparse
import json
import os
from pathlib import Path

from model_client import ModelClient


PROMPT = """你负责识别新闻标题中的公司、股票代码和产品名称，不进行预测。
目标公司名称和股票代码由用户提供。只输出 JSON 对象，格式为：
{"entities":[{"text":"标题中原文片段","kind":"target|company|product","id":"同一实体的统一编号"}]}
text 必须与标题中的片段完全一致。目标公司及其股票代码标为 target，其他公司标为 company，
产品标为 product；产品指具体品牌或商业产品名称，普通金融术语、行业、概念和一般股票投资组合不算产品。
具体命名的基金等商业产品名称可以标记。id 必须是字符串，同一实体的不同写法应使用同一 id。没有明确提到的实体不要添加。
不要根据文件所属股票假定标题一定提到目标公司。除公司、股票代码、产品外不要标记。
不需要解释或改写标题；没有实体时返回 {"entities":[]}。"""


def mask_title(client, model, title, target_name, ticker, market="CN"):
    """识别实体并在原文中替换；每条标题的其他公司和产品编号独立。"""
    content = json.dumps({"目标公司": target_name, "股票代码": ticker, "标题": title}, ensure_ascii=False)
    result = json.loads(client.complete(model=model, prompt=PROMPT, content=content))
    if not isinstance(result, dict) or not isinstance(result.get("entities"), list):
        raise ValueError("模型响应缺少 entities 列表。")
    replacements = []
    company_ids, product_ids = {}, {}
    company, product = ("Company", "Product") if market == "US" else ("公司", "产品")
    for entity in result["entities"]:
        if not isinstance(entity, dict):
            raise ValueError("模型返回了无效的实体。")
        span, kind, identity = entity.get("text"), entity.get("kind"), entity.get("id")
        if not isinstance(span, str) or not span or kind not in ("target", "company", "product"):
            raise ValueError("模型返回了无效的实体文字或类型。")
        if not isinstance(identity, str) or not identity.strip() or span not in title:
            raise ValueError("模型返回的实体编号无效，或实体不在标题中。")
        if kind == "target":
            label = f"{company} A"
        elif kind == "company":
            if identity not in company_ids:
                number = len(company_ids)
                company_ids[identity] = f"{company} {chr(66 + number) if number < 25 else number + 2}"
            label = company_ids[identity]
        else:
            if identity not in product_ids:
                product_ids[identity] = f"{product} {len(product_ids) + 1}"
            label = product_ids[identity]
        start = 0
        while (start := title.find(span, start)) != -1:
            replacements.append((start, start + len(span), label))
            start += len(span)
    replacements.sort()
    if any(replacements[i][0] < replacements[i - 1][1] for i in range(1, len(replacements))):
        raise ValueError("模型返回的实体片段重复或重叠，请检查后重试。")
    for start, end, label in reversed(replacements):
        title = title[:start] + label + title[end:]
    return title


def mask_file(source, destination, market, target_name, ticker, client, model, limit=0):
    """读取预处理 JSONL，生成新文件；不覆盖已存在的结果。"""
    if destination.exists():
        raise FileExistsError(f"输出文件已存在，不会覆盖：{destination}")
    records = []
    with source.open(encoding="utf-8") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["text"]
            if not isinstance(text, list) or not all(isinstance(part, str) for part in text):
                raise ValueError("新闻 text 应为字符串列表。")
            title = (" " if market == "US" else "").join(text)
            records.append({"text": mask_title(client, model, title, target_name, ticker, market=market),
                            "created_at": record["created_at"]})
            if limit and len(records) >= limit:
                break
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="对 CMIN 单只股票、单日新闻执行名称 mask")
    parser.add_argument("--market", choices=("US", "CN"), required=True)
    parser.add_argument("--stock", required=True, help="preprocessed 下的股票文件夹名")
    parser.add_argument("--date", required=True, help="新闻文件名，例如 2021-01-04")
    parser.add_argument("--target-name", required=True, help="目标公司名称，例如 Apple")
    parser.add_argument("--ticker", required=True, help="目标股票代码，例如 AAPL")
    parser.add_argument("--limit", type=int, default=5, help="试跑条数；0 表示整份文件")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 不能为负数")
    for value in (args.stock, args.date):
        if value in (".", "..") or "/" in value or "\\" in value or ":" in value:
            parser.error("股票文件夹名和日期必须是单个文件名")
    # 读取项目根目录的本地配置；已有环境变量优先，密钥不会写入代码或输出。
    config = Path(".env")
    keys = ("MODEL_BASE_URL", "MODEL_API_KEY", "MASK_MODEL")
    if config.exists():
        for line in config.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if separator and key in keys:
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    missing = [key for key in keys if not os.getenv(key, "").strip()]
    if missing:
        raise ValueError("请在项目根目录 .env 或环境变量中配置：" + ", ".join(missing))
    model = os.environ["MASK_MODEL"]
    client = ModelClient.from_environment()
    news = Path("data/CMIN-Dataset") / f"CMIN-{args.market}" / "news"
    source = news / "preprocessed" / args.stock / args.date
    # 试跑结果单独命名，避免和整日数据混淆。
    name = args.date if args.limit == 0 else f"{args.date}.sample-{args.limit}"
    destination = news / "masked" / args.stock / name
    count = mask_file(source, destination, args.market, args.target_name, args.ticker, client, model, args.limit)
    print(f"已处理 {count} 条新闻，保存至 {destination}")


if __name__ == "__main__":
    main()
