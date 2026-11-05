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
8. **Model selection** among Ridge, Lasso, Decision Tree, Random Forest and KNN on the hourly series, followed by a `GridSearchCV` on KNN.
9. **Seasonal model and fair comparison**: SARIMA with a weekly cycle, plus all models compared on the same daily problem at 1 and 7 days ahead, against two naive baselines.

## Results

Two different problems appear in the notebook, so there are two result tables.

**1. Hourly: predict the next hour (machine learning models).** The target is the change in load from one hour to the next, using the previous 6 hourly changes. MAE is in the same units as the load (MW).

| Model | CV MAE | Test MAE |
|---|---|---|
| Naive (no change) | – | 273.4 |
| KNN | 126.4 | – |
| Random Forest | 128.3 | – |
| Decision Tree | 162.9 | – |
| Ridge | 182.9 | 195.1 |
| Lasso | 182.9 | – |
| KNN (tuned with GridSearchCV) | – | **87.5** (68% below the naive forecast) |

**2. Daily: predict the load 1 and 7 days ahead (all models on the same split).** MAE on the load level, last 30% of the days as the test period (about five months, roughly 5,650 MW on average).

| Model | MAE, 1 day ahead | MAE, 7 days ahead |
|---|---|---|
| Random Forest (7 lags + weekday and holidays) | **103.5** | 205.3 |
| SARIMA(2,0,2)(1,0,1)[7] | 105.3 | 226.9 |
| Random Forest (7 lags) | 107.0 | **204.0** |
| KNN (7 lags) | 108.7 | 210.3 |
| Naive: same weekday last week | 219.1 | 223.1 |
| ARIMA(2,0,2) | 236.0 | 389.4 |
| Naive: same as today | 355.7 | 223.1 |

Adding the weekly cycle to the ARIMA (SARIMA) lowers the AIC from 7773 to 7191 and removes the leftover residual autocorrelation. One day ahead, the best models cut the naive error to about a third; seven days ahead the gain is small. The differences among SARIMA, KNN and Random Forest are within a few MW and come from a single test period. Tree-based models use `random_state=42`.

## Tech stack

Python, pandas, NumPy, statsmodels, pmdarima, scikit-learn, SciPy, Matplotlib, holidays.

## How to run

- **Colab:** click the badge above (remember to provide the data file, see *Data*).
- **Locally:**

```bash
pip install pandas numpy scipy matplotlib statsmodels pmdarima scikit-learn patsy requests holidays jupyter
jupyter notebook "Energy_Load_'Portugal'.ipynb"
```

## Next steps

- The Shapiro-Wilk test rejects normality of the series; check whether a Box-Cox transformation improves the residual diagnostics.
- Add weather (temperature) as an exogenous variable and test hourly seasonality (24 h and 168 h cycles).
- Evaluate on several test periods (rolling origin) before declaring a winner among the close models.

## Author

Jefferson Melo
[LinkedIn](https://www.linkedin.com/in/jefferson-omelo) · [GitHub](https://github.com/jeffairdata)
