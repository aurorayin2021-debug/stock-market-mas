"""Predict one stock using saved News/K-line outputs and a fresh Relation analysis."""
import argparse
from datetime import date
import json
from functools import partial
from pathlib import Path

from agents import relation_agent, manager_agent
from backtest.evaluation import evaluate_predictions
from dataflows.analyst_cache import load_analyst_cache
from model_client import ModelClient
from run_analysts import _load_config
from stock_mas import build_graph


def run(stock, prediction_date, *, data_root='data', cache_root='outputs/analysts_with_reasons', evaluate=False,
        relation_model='deepseek-flash', manager_model='deepseek-flash'):
    day = date.fromisoformat(prediction_date).isoformat()
    cached, provenance = load_analyst_cache(cache_root, stock, day)
    price_root = Path(data_root) / 'CMIN-Dataset' / 'CMIN-US' / 'price' / 'raw'
    matches = [p for p in price_root.glob('*.csv') if p.stem.casefold() == stock.casefold()]
    if len(matches) != 1:
        raise ValueError('No unique matching price file; do not map different securities implicitly.')
    label_path = Path(data_root) / 'US' / 'labels' / f'{day}.csv'
    if evaluate and not label_path.is_file():
        raise FileNotFoundError('Missing reference labels.')
    if not relation_model or not manager_model:
        raise ValueError('RELATION_MODEL and MANAGER_MODEL must be configured.')
    client = ModelClient.from_environment()
    relation_task = {'stock': matches[0].stem, 'price_root': str(price_root),
                     'prediction_date': day, 'estimator': 'author_code'}
    graph = build_graph(
        {'news': lambda value: value, 'kline': lambda value: value,
         'relation': partial(relation_agent.analyze_with_tool, client=client, model=relation_model)},
        placeholder_results=False, manager_fn=partial(manager_agent.analyze, client=client, model=manager_model),
        require_inputs=True)
    report = graph.invoke({'inputs': {**cached, 'relation': relation_task}})['report']
    result = {'stock': stock, 'prediction_date': day, 'cache_provenance': provenance, 'report': report}
    if evaluate:
        result['evaluation'] = evaluate_predictions({stock: report['manager_result']}, label_path)
    return result


def main():
    parser = argparse.ArgumentParser(description='Predict using existing News/K-line cached responses.')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--cache-root', default='outputs/analysts_with_reasons')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--relation-model', default='deepseek-flash')
    parser.add_argument('--manager-model', default='deepseek-flash')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error('Output already exists or parent directory does not exist.')
    _load_config()
    result = run(args.stock, args.date, data_root=args.data_root, cache_root=args.cache_root, evaluate=args.evaluate,
                 relation_model=args.relation_model, manager_model=args.manager_model)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')


if __name__ == '__main__':
    main()
