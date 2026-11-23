"""单只股票的 API 工作流；只在显式请求时读取参考答案进行离线评价。"""
import argparse
from datetime import date
import json
from pathlib import Path

from api_workflow import build_api_graph
from backtest.evaluation import evaluate_predictions
from dataflows.prepare_data import prepare_us_inputs


def run(stock, prediction_date, *, data_root="data", evaluate=False):
    day = date.fromisoformat(prediction_date).isoformat()
    labels = Path(data_root) / "US" / "labels" / f"{day}.csv"
    if evaluate and not labels.is_file():
        raise FileNotFoundError("找不到该预测日的参考答案。")
    graph = build_api_graph()
    inputs = prepare_us_inputs(data_root, stock, day)
    report = graph.invoke({"inputs": inputs})["report"]
    result = {"stock": stock, "prediction_date": day, "report": report}
    if evaluate:
        result["evaluation"] = evaluate_predictions({stock: report["manager_result"]}, labels)
    return result


def main():
    parser = argparse.ArgumentParser(description="运行单只股票的多智能体预测。")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--date", required=True, help="预测日，格式 YYYY-MM-DD")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--evaluate", action="store_true", help="预测完成后离线对照参考答案")
    parser.add_argument("--output", type=Path, help="可选：保存到新文件，不覆盖已有文件")
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("输出文件已存在或父文件夹不存在。")
    result = run(args.stock, args.date, data_root=args.data_root, evaluate=args.evaluate)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
