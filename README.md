# 股票预测多智能体系统：第一版

本项目使用新闻、K 线和股票历史关联三个分析模块，为目标股票预测下一交易日收盘价相对于前一交易日收盘价的涨跌。三个分析结果交由 Manager 汇总，输出预测方向和简短理由。项目使用 LangGraph 编排，模型通过兼容 Chat Completions 的 API 调用。

## 第一版方法

- **News**：读取预测日前一自然日的去敏新闻标题，逐条输出五维评分和简短理由。
- **K 线**：从预测日前最近 20 个交易日的 OHLCV 行情生成图像，输出六维评分和简短理由。
- **Relation**：使用 2018—2020 年共同交易日收盘价收益率，按照作者代码的联合熵差计算关联分数，选出 Top 30，并提供一日滞后相关系数和最近交易日收益率。该分数不是标准转移熵，不应解释为价格涨跌方向或因果关系。
- **Manager**：综合三路匿名分析，输出 `Prediction: UP`、`DOWN` 或 `INSUFFICIENT`，以及 `Reason`。

历史窗口、股票池和数据处理采用第一版设置，仍需在正式研究中检验适用性。对新闻进行名称替换也不能保证完全消除历史知识污染。

## 目录

- `agents/`：四个智能体的任务和提示词。
- `tools/`、`skills/transfer_entropy/`：关联工具、Skill 和作者代码指标的计算。
- `dataflows/`：本地新闻、行情、K 线图和分析结果缓存的读取与处理。
- `model_client.py`：公共模型 API 调用。
- `stock_mas.py`、`api_workflow.py`：工作流与模型调用接线。
- `backtest/evaluation.py`：预测方向与参考答案的离线对照；不包含交易模拟。
- `baselines/`：对比方法的预留目录。

仓库不上传本地数据、模型输出、测试文件、API 密钥或虚拟环境。数据和此前生成的缓存需要自行准备。

## 安装环境

在项目根目录执行：

```powershell
conda env create --prefix ./.conda -f environment.yml
conda activate ./.conda
python stock_mas.py
```

最后一条运行的是不调用 API 的占位演示，不会生成真实预测。

## 数据和模型配置

第一版运行入口使用本地数据，主要路径如下：

```text
data/
├── US/news/masked/{股票}/{新闻日期}     # 每行一个去敏新闻 JSON 对象
├── US/labels/{预测日期}.csv             # 仅在评价时使用
└── CMIN-Dataset/CMIN-US/price/raw/{股票}.csv  # 日行情
```

新闻记录须包含 `text` 和 `created_at`；行情须包含 `Date`、`Open`、`High`、`Low`、`Close`、`Volume`。请自行准备与代码约定一致的数据，并确认证券代码与时间窗口。股票真实身份、日期和参考答案由本地程序管理，参考答案不输入模型。

复制 `.env.example` 为本地 `.env`，填写 `MODEL_BASE_URL` 和 `MODEL_API_KEY`；不要将真实密钥提交到仓库。`run_analysts.py`、`run_analysts_batch.py`、`run_cached_prediction.py` 和 `run_relation_manager_batch.py` 的命令行入口会读取这两个设置，已有环境变量优先；默认模型名为 `deepseek-flash`。运行 News、K 线、Relation、Manager 时，请确认服务和模型分别支持所需的文本、图像与工具调用。

`run_prediction.py` 使用的 API 工作流只读取进程环境变量，不自动读取 `.env`；使用它时还需设置四个 Agent 对应的模型名。`.env.example` 列出了相关名称。

## 运行与缓存

可以先分别生成新闻和 K 线分析：

```powershell
python run_analysts.py --stock BAC --date 2021-02-02 --role news
python run_analysts.py --stock BAC --date 2021-02-02 --role kline
```

结果逐项保存在 `outputs/analysts_with_reasons/{股票}/{预测日期}/` 下。缓存记录模型、提示词和输入的标识；重新运行时检查一致性并跳过已完成内容。

已有完整 News/K 线缓存时，可运行 Relation 和 Manager：

```powershell
python run_relation_manager_batch.py --date 2021-02-02 --stock BAC --workers 1
```

去掉 `--stock BAC` 会处理参考答案文件中标记为可匹配的所有股票。每只股票的 Relation 和 Manager 结果分别保存，可在中断后续跑。该入口会先备份参考答案 CSV，再将预测、理由和对错写入该 CSV；它只在预测后使用答案评价。请先核对运行范围，模型调用可能产生费用。

`run_cached_prediction.py` 也可对单只股票复用已有 News/K 线结果；`run_prediction.py` 则直接运行完整工作流，不自动复用这些缓存。
