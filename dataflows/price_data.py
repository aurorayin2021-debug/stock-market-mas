"""Load historical OHLCV observations for an explicit prediction date."""
from __future__ import annotations
import csv
from datetime import date
import math
from pathlib import Path


def read_window(price_file: str | Path, prediction_date: str | date, window: int = 20) -> list[dict]:
    """Read the latest `window` observed trading rows strictly before prediction_date."""
    if isinstance(prediction_date, str):
        prediction_date = date.fromisoformat(prediction_date)
    if not isinstance(prediction_date, date):
        raise TypeError("prediction_date must be an ISO date or datetime.date")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    rows = []
    with Path(price_file).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("price CSV is missing required Date or OHLCV columns")
        for item in reader:
            day = date.fromisoformat(item["Date"])
            if day < prediction_date:
                rows.append((day, item))
    rows.sort(key=lambda row: row[0])
    if len({day for day, _ in rows}) != len(rows):
        raise ValueError("price CSV contains duplicate trading dates")
    if len(rows) < window:
        raise ValueError(f"insufficient trading history: need {window}, found {len(rows)}")
    result = []
    for day, row in rows[-window:]:
        try:
            values = {name.lower(): float(row[name]) for name in ("Open", "High", "Low", "Close", "Volume")}
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid OHLCV data for {day.isoformat()}") from error
        o, h, l, c, v = (values[name] for name in ("open", "high", "low", "close", "volume"))
        if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0 or h < max(o, c, l) or l > min(o, c, h):
            raise ValueError(f"invalid OHLCV data for {day.isoformat()}")
        result.append({"date": day, **values})
    return result
