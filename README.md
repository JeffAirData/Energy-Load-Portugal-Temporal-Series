# Energy Load Portugal: Time Series Analysis and Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeffAirData/Energy-Load-Portugal-Temporal-Series/blob/main/Energy_Load_'Portugal'.ipynb)

A time series study of Portugal's electricity load, combining classical statistical analysis (decomposition, stationarity tests, ARIMA) with supervised machine learning models for forecasting.

## Data

- 13,104 hourly observations of electricity load in Portugal, starting on 2020-11-17.
- The series is resampled to daily frequency for the classical analysis; lagged values are used as features for the supervised models.
- **Data source note:** the notebook originally loaded `energy_load_pt.csv` from the public repository `vcerqueira/energy_load_data`. That repository is no longer available (404), so the data-loading cell does not run as is. To reproduce the analysis, place a copy of `energy_load_pt.csv` (hourly, columns `datetime` and `load`) in the working directory and adjust the path in the data-loading cell. The dataset is not redistributed here.

## What the notebook does

1. **Exploration** of the hourly load series.
2. **Additive decomposition** into trend, seasonality and residual.
3. **Stationarity**: Augmented Dickey-Fuller test, `ndiffs` and `nsdiffs`.
4. **PACF / ACF** analysis to suggest AR and MA orders.
5. **Shapiro-Wilk** (normality) and **White** (heteroscedasticity) tests.
6. **ARIMA** models with candidate orders (2,1,2), (1,1,2) and (1,1,1), selected by coefficient significance, plus residual diagnostics.
7. **Time-delay embedding** to turn the series into a supervised problem, with a chronological 70/30 train/test split and `TimeSeriesSplit` (5 folds).
8. **Model selection** among Ridge, Lasso, Decision Tree, Random Forest and KNN, followed by a `GridSearchCV` on KNN.

## Results

Error metric: MAE, in the same units as the load series.

| Model | CV MAE | Test MAE |
|---|---|---|
| KNN | 126.4 | – |
| Random Forest | 130.0 | – |
| Decision Tree | 159.3 | – |
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

- Revisit the ARIMA differencing order: the series is differenced once before modelling, and `d=1` then applies a second difference. Refit with `d=0` on the differenced series (or `d=1` on the original) and compare.
- The Shapiro-Wilk test rejects normality of the series; check whether a Box-Cox transformation improves the residual diagnostics.
- Add seasonal terms (SARIMA) and compare against the machine learning models on the same folds.
- Evaluate more forecast horizons and add exogenous variables (e.g., calendar and weather).

## Author

Jefferson Melo
[LinkedIn](https://www.linkedin.com/in/jefferson-omelo) · [GitHub](https://github.com/jeffairdata)
