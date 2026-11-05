"""Daily comparison used by the page: same logic as the last section of the notebook."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")


def load_hourly(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    return df.set_index("datetime")["load"]


def compare(hourly: pd.Series):
    daily = hourly.resample("D").mean().interpolate()
    ds = daily.diff()
    N = len(daily); n0 = int(N * 0.7); K = 7
    dow = pd.get_dummies(pd.Series(daily.index.dayofweek, index=daily.index), dtype=float)
    dow.columns = [f"dow{c}" for c in dow.columns]
    lags = pd.concat([ds.shift(i) for i in range(K)], axis=1)
    lags.columns = [f"lag{i}" for i in range(K)]
    pos = np.arange(N)
    base = {n: SARIMAX(ds.iloc[1:n0], order=(2, 0, 2), seasonal_order=so).fit(disp=False)
            for n, so in [("ARIMA(2,0,2)", (0, 0, 0, 0)), ("SARIMA(2,0,2)(1,0,1)[7]", (1, 0, 1, 7))]}
    results, game = {}, None
    for h in (1, 7):
        te = [p for p in range(n0, N) if p + h < N]
        truth = np.array([daily.iloc[p + h] - daily.iloc[p] for p in te])
        P = {"Naive: same as today": np.zeros(len(te)),
             "Naive: same weekday last week": np.array([daily.iloc[p + h - 7] - daily.iloc[p] for p in te])}
        tgt = daily.shift(-h) - daily
        ok = lags.notna().all(axis=1) & tgt.notna() & (pos + h <= n0 - 1)
        Xc = pd.concat([lags, dow.shift(-h)], axis=1)
        for name, m, X in [("KNN (7 lags)", KNeighborsRegressor(n_neighbors=10), lags),
                           ("Random Forest (7 lags)", RandomForestRegressor(n_estimators=200, random_state=42), lags),
                           ("Random Forest (7 lags + weekday)", RandomForestRegressor(n_estimators=200, random_state=42), Xc)]:
            m.fit(X[ok].fillna(0), tgt[ok]); P[name] = m.predict(X.iloc[te].fillna(0))
        for name, f in base.items():
            r = f.append(ds.iloc[n0:te[0] + 1], refit=False); out = []
            for p in te:
                out.append(r.forecast(h).sum())
                if p + 1 < N:
                    r = r.append(ds.iloc[p + 1:p + 2], refit=False)
            P[name] = np.array(out)
        results[h] = {m: float(np.abs(v - truth).mean()) for m, v in P.items()}
        if h == 1:
            game = []
            for k, p in enumerate(te):
                game.append({"date": str(daily.index[p + 1].date()),
                             "last7": [round(float(x)) for x in daily.iloc[p - 6:p + 1]],
                             "actual": round(float(daily.iloc[p + 1])),
                             "sarima": round(float(daily.iloc[p] + P["SARIMA(2,0,2)(1,0,1)[7]"][k])),
                             "rf": round(float(daily.iloc[p] + P["Random Forest (7 lags + weekday)"][k])),
                             "naive_week": round(float(daily.iloc[p + 1 - 7])),
                             "naive_today": round(float(daily.iloc[p]))})
    return daily, n0, results, game
