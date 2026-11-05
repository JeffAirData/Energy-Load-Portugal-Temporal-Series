# Energy Load Portugal: Time Series Analysis and Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeffAirData/Energy-Load-Portugal-Temporal-Series/blob/main/Energy_Load_'Portugal'.ipynb)

A time series study of Portugal's electricity load, combining classical statistical analysis (decomposition, stationarity tests, ARIMA) with supervised machine learning models for forecasting.

## Data

- 13,104 hourly observations of electricity load in Portugal, starting on 2020-11-17.
- The series is resampled to daily frequency for the classical analysis; lagged values are used as features for the supervised models.
- **Reproducing:** the notebook originally loaded `energy_load_pt.csv` from the public repository `vcerqueira/energy_load_data`, which is no longer available. The file is not redistributed here because its redistribution terms are unknown (the load figures come from Portugal's transmission operator, REN). To run the notebook, obtain a copy of `energy_load_pt.csv` (hourly, columns `datetime` and `load`, 13,104 rows from 2020-11-17 08:00 to 2022-05-17 08:00) and point the data-loading cell (`pd.read_csv(...)`) to its local path. Equivalent public data can be exported from the ENTSO-E Transparency Platform (Total Load, Portugal) or the REN Data Hub, but figures may differ slightly from those below.

## What the notebook does

1. **Exploration** of the hourly load series.
2. **Additive decomposition** into trend, seasonality and residual.
3. **Stationarity**: Augmented Dickey-Fuller test, `ndiffs` and `nsdiffs`.
4. **PACF / ACF** analysis to suggest AR and MA orders.
5. **Shapiro-Wilk** (normality) and **White** (heteroscedasticity) tests.
6. **ARIMA** models on the differenced daily series, with candidate orders (2,0,2), (1,0,2) and (1,0,1) compared by coefficient significance and AIC (ARIMA(2,0,2) has the lowest AIC), plus residual diagnostics (Ljung-Box and Shapiro-Wilk still reject white-noise, normal residuals, which points to unmodelled weekly seasonality).
7. **Time-delay embedding** to turn the series into a supervised problem, with a chronological 70/30 train/test split and `TimeSeriesSplit` (5 folds).
8. **Model selection** among Ridge, Lasso, Decision Tree, Random Forest and KNN, followed by a `GridSearchCV` on KNN.

## Results

Error metric: MAE. The supervised models are trained on the **differenced daily series** (day-over-day change in the daily mean load, in the same units as the load), so these errors refer to the daily change, not to the load level. Tree-based models use `random_state=42`; Ridge, Lasso, KNN and the tuned KNN test MAE do not depend on a seed.

| Model | CV MAE | Test MAE |
|---|---|---|
| KNN | 126.4 | – |
| Random Forest | 128.3 | – |
| Decision Tree | 162.9 | – |
| Ridge | 182.9 | 195.1 |
| Lasso | 182.9 | – |
| KNN (tuned with GridSearchCV) | – | **87.5** |

## Tech stack

Python, pandas, NumPy, statsmodels, pmdarima, scikit-learn, SciPy, Matplotlib.

## How to run

- **Colab:** click the badge above (remember to provide the data file, see *Data*).
- **Locally:**

```bash
pip install pandas numpy scipy matplotlib statsmodels pmdarima scikit-learn patsy requests jupyter
jupyter notebook "Energy_Load_'Portugal'.ipynb"
```

## Next steps

- The Shapiro-Wilk test rejects normality of the series; check whether a Box-Cox transformation improves the residual diagnostics.
- Add seasonal terms (SARIMA, weekly seasonality) to address the remaining residual autocorrelation and compare against the machine learning models on the same folds.
- Evaluate more forecast horizons and add exogenous variables (e.g., calendar and weather).

## Author

Jefferson Melo
[LinkedIn](https://www.linkedin.com/in/jefferson-omelo) · [GitHub](https://github.com/jeffairdata)
