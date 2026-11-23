# Stock Relationship Calculation Skill

## Tool
When a tool is available, call `get_relation_evidence` to obtain the relationship table. If the evidence has already been provided, use it directly. The program supplies the calculation settings and executes the tool.

## Calculation
1. Use daily returns calculated from closing prices and align the available stocks by trading day within the configured historical window.
2. For each candidate, calculate the author-code relationship score: H(target current return, target past return) minus H(target current return, candidate past return, target past return). These are joint entropies; the score is not standard transfer entropy. Use a lag of one trading day. The calculation script uses the author's 10-bin, base-2 and smoothing settings.
3. Calculate the historical correlation between the candidate's lagged returns and the target's current returns. Rank candidates by the author-code relationship score and select the top 30.
4. For each selected stock, include its return on the most recent trading day before the prediction date. Return the resulting relationship table.
