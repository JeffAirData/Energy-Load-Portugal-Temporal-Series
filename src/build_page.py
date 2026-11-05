"""Builds docs/index.html and the PNG charts/cards from the hourly CSV (not included in the repo).

Usage: python src/build_page.py path/to/energy_load_pt.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"; IMG.mkdir(parents=True, exist_ok=True)
BLUE, RED, AMBER, GREY = "#2a78d6", "#e34948", "#eda100", "#c3c2b7"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GREY, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
FONTE = "Data: hourly electricity load of Portugal, Nov 2020 to May 2022."
fmt = lambda v: f"{v:,.0f}"


def main(csv: str) -> None:
    hourly = analysis.load_hourly(csv)
    daily, n0, results, game = analysis.compare(hourly)
    ds = daily.diff().dropna()
    aic = {"arima": SARIMAX(ds, order=(2, 0, 2)).fit(disp=False).aic,
           "sarima": SARIMAX(ds, order=(2, 0, 2), seasonal_order=(1, 0, 1, 7)).fit(disp=False).aic}
    weekday = daily.groupby(daily.index.dayofweek).mean()
    quiet = sorted(int(i) for i in weekday.sort_values().index[:2])   # the two lowest days of the weekly cycle
    mask = hourly.index.dayofweek.isin(quiet)
    we = hourly[mask].groupby(hourly[mask].index.hour).mean(); wd = hourly[~mask].groupby(hourly[~mask].index.hour).mean()
    mean_load = float(daily.iloc[n0:].mean())
    data = {"dates": [str(d.date()) for d in daily.index], "daily": [round(float(v)) for v in daily], "n0": n0,
            "hour_other": [round(float(v)) for v in wd], "hour_quiet": [round(float(v)) for v in we], "quiet": quiet,
            "weekday": [round(float(v)) for v in weekday], "results": {str(h): {k: round(v, 1) for k, v in r.items()} for h, r in results.items()},
            "mean_load": round(mean_load), "aic": {k: round(float(v), 1) for k, v in aic.items()}, "game": game}
    html = (ROOT / "src" / "page_template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    charts(daily, n0, wd, we, weekday, results, game, mean_load, aic, quiet)
    print("ok", {h: {k: round(v, 1) for k, v in r.items()} for h, r in results.items()}, aic, round(mean_load))


def charts(daily, n0, wd, we, weekday, results, game, mean_load, aic, quiet):
    # 01 series
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(daily.index, daily.values, color=INK, lw=1.2)
    ax.axvspan(daily.index[n0], daily.index[-1], color=BLUE, alpha=.08, lw=0); ax.text(daily.index[n0], 7300, " test period", color=BLUE, va="top")
    ax.set_title("Portugal's daily electricity load (average, MW)", loc="left", fontweight="bold"); ax.grid(axis="y", color=GRID)
    fig.text(0.01, 0.01, FONTE, color=MUTED, fontsize=8); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(IMG / "01_serie_diaria.png", dpi=150); plt.close(fig)
    # 02 hour and week
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    a.plot(wd.index, wd.values, color=BLUE, lw=2.4, label="Five busier days"); a.plot(we.index, we.values, color=AMBER, lw=2.4, label="The two quietest days")
    a.set_title("Average load by hour of the day", loc="left", fontweight="bold"); a.legend(frameon=False); a.grid(axis="y", color=GRID); a.set_xlabel("Hour as recorded in the file")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; b.bar(names, weekday.values, color=[AMBER if i in quiet else BLUE for i in range(7)])
    b.set_ylim(weekday.min() * .85, weekday.max() * 1.03); b.set_title("Average load by day of the week (as recorded)", loc="left", fontweight="bold"); b.grid(axis="y", color=GRID)
    fig.text(0.01, 0.01, FONTE, color=MUTED, fontsize=8); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(IMG / "02_ritmo.png", dpi=150); plt.close(fig)
    # 03 ranking
    colors = {"Random Forest (7 lags + weekday)": RED, "Random Forest (7 lags)": RED, "SARIMA(2,0,2)(1,0,1)[7]": BLUE, "KNN (7 lags)": AMBER}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    order = [k for k, _ in sorted(results[1].items(), key=lambda kv: kv[1], reverse=True)]  # same row order in both panels
    for ax, h, t in zip(axs, (1, 7), ("1 day ahead", "7 days ahead")):
        items = [(k, results[h][k]) for k in order]
        ax.barh([k for k, _ in items], [v for _, v in items], color=[colors.get(k, GREY) for k, _ in items])
        for i, (_, v) in enumerate(items): ax.text(v + 4, i, fmt(v), va="center", fontsize=9)
        ax.set_title(f"Error, {t} (MW, lower is better)", loc="left", fontweight="bold", fontsize=10); ax.set_xlim(0, max(results[h].values()) * 1.15)
        if h == 7: ax.set_yticklabels([])
    fig.text(0.01, 0.01, FONTE + " Test period: last 30% of the days.", color=MUTED, fontsize=8); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(IMG / "03_modelos.png", dpi=150); plt.close(fig)
    # 04 forecast zoom
    g = pd.DataFrame(game); g["date"] = pd.to_datetime(g["date"]); g = g.iloc[:60]
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(g.date, g.actual, color=INK, lw=2, label="Actual")
    ax.plot(g.date, g.sarima, color=BLUE, lw=1.6, label="SARIMA"); ax.plot(g.date, g.naive_today, color=GREY, lw=1.4, ls="--", label="Naive: tomorrow = today")
    ax.set_title("Predicting tomorrow, first 60 days of the test period (MW)", loc="left", fontweight="bold"); ax.legend(frameon=False, ncol=3); ax.grid(axis="y", color=GRID)
    fig.text(0.01, 0.01, FONTE, color=MUTED, fontsize=8); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(IMG / "04_previsao.png", dpi=150); plt.close(fig)
    # cards 1080x1080
    r1, r7 = results[1], results[7]; best = r1["Random Forest (7 lags + weekday)"]; naive = r1["Naive: same as today"]
    cards = [
        ("Can we guess how much electricity Portugal will use tomorrow?", "18 months of hourly data. Time series, forecasting models and a bit of fun.", None),
        (f"{(1 - weekday.iloc[quiet].mean() / weekday.drop(weekday.index[quiet]).mean()) * 100:.0f}%", "less electricity on the two quietest days of the week. Demand follows a weekly rhythm.", None),
        (f"{fmt(naive)} → {fmt(best)}", "MW of error when guessing tomorrow: copying today vs. a model that knows the weekly rhythm.", None),
        (f"{aic['arima']:,.0f} → {aic['sarima']:,.0f}", "AIC (lower is better) when a weekly cycle was added to the ARIMA. The missing piece was the week.", None),
        ("7 days ahead is a different story", f"The simple models ({fmt(r7['Random Forest (7 lags)'])} MW) barely beat \"same weekday last week\" ({fmt(r7['Naive: same weekday last week'])} MW). Honest limits matter.", None),
    ]
    for i, (big, small, _) in enumerate(cards, 1):
        fig = plt.figure(figsize=(10.8, 10.8), dpi=100); fig.patch.set_facecolor(SURFACE)
        fs = 66 if len(big) < 14 else (48 if len(big) < 30 else 40)
        fig.text(.08, .62, big, fontsize=fs, fontweight="bold", color=BLUE if i > 1 else INK, wrap=True, va="center")
        fig.text(.08, .40, "\n".join(__import__("textwrap").wrap(small, 38)), fontsize=26, color=INK_2, va="top")
        fig.text(.08, .06, f"{i}/5 · Portugal electricity load · github.com/JeffAirData", fontsize=15, color=MUTED)
        fig.savefig(IMG / f"cartao_{i}.png"); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
