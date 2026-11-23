"""Draw an anonymous K-line chart; runnable as python -m dataflows.kline_chart."""
from __future__ import annotations
import argparse
from io import BytesIO
from pathlib import Path
from .price_data import read_window


def draw_candlesticks(rows: list[dict]) -> bytes:
    """Render PNG with relative day ticks; no company or actual dates in the image."""
    if not rows:
        raise ValueError("cannot draw an empty price window")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except ImportError as error:
        raise RuntimeError("drawing a chart requires matplotlib") from error
    fig = Figure(figsize=(9, 5.5), dpi=120, layout="constrained")
    FigureCanvasAgg(fig)
    ax, volume_ax = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i, row in enumerate(rows):
        color = "#ba4a4a" if row["close"] >= row["open"] else "#38826a"
        ax.vlines(i, row["low"], row["high"], color=color, linewidth=1)
        body = abs(row["close"] - row["open"])
        if body == 0:
            ax.hlines(row["open"], i - .3, i + .3, color=color, linewidth=1.5)
        else:
            ax.add_patch(Rectangle((i - .3, min(row["open"], row["close"])), .6, body,
                                   facecolor=color, edgecolor=color))
        volume_ax.bar(i, row["volume"], width=.6, color=color)
    ax.set_ylabel("Price")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color="#ba4a4a", label="Close ≥ Open"),
                       Rectangle((0, 0), 1, 1, color="#38826a", label="Close < Open")],
              loc="upper left", frameon=False)
    volume_ax.set_ylabel("Volume")
    volume_ax.set_xlabel("Trading days (oldest to newest)")
    ticks = sorted({0, (len(rows) - 1) // 2, len(rows) - 1})
    volume_ax.set_xticks(ticks, [f"Day {i + 1}" for i in ticks])
    ax.set_xlim(-.8, len(rows) - .2)
    ax.margins(y=.08)
    image = BytesIO()
    fig.savefig(image, format="png")
    return image.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw an anonymous OHLCV chart using trading days before prediction date.")
    parser.add_argument("--price-file", required=True, type=Path, help="Path to raw OHLCV CSV")
    parser.add_argument("--prediction-date", required=True, help="Prediction date (YYYY-MM-DD), excluded from the window")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path; must not already exist")
    parser.add_argument("--window", type=int, default=20, help="Number of prior trading observations (default: 20)")
    args = parser.parse_args(argv)
    try:
        rows = read_window(args.price_file, args.prediction_date, args.window)
        png = draw_candlesticks(rows)
        with args.output.open("xb") as stream:
            stream.write(png)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
