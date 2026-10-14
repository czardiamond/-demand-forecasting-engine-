# Demand Forecasting & Inventory Optimization Engine

Predicts daily product sales and turns the forecast into inventory decisions (safety stock, reorder point, days of stock left). Built around a Lagos water-production business (sachet water, 75cl bottles, 19L dispensers).

## How it works
1. **Ingest** historical sales (`date, product, units_sold` CSV), or use the built-in sample data.
2. **Feature engineering:** day of week, month, day of year, ISO week, weekend flag, Nigerian public holidays, yearly seasonality (sin/cos), lags (1, 7, 14, 28 days) and rolling means (7, 28 days).
3. **Model:** scikit-learn `HistGradientBoostingRegressor` (swap in CatBoost easily).
4. **Validation:** 5-fold `TimeSeriesSplit`, so the model is always tested on future data. Reports **MAE** and **RMSE**.
5. **Forecast:** recursive multi-step forecast up to 90 days.
6. **Inventory:** safety stock = z x residual std x sqrt(lead time); reorder point = lead-time demand + safety stock.

## Sample results (synthetic data)
| Product | CV MAE | CV RMSE |
|---|---|---|
| Sachet water (bag) | 56.9 | 67.5 |
| 75cl bottle (pack) | 18.1 | 21.4 |
| Dispenser 19L | 6.1 | 7.1 |

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack
Python, Pandas, NumPy, scikit-learn, Streamlit.
