"""Demand forecasting & inventory optimization engine."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

HOLIDAYS = {(1,1),(5,1),(5,27),(6,12),(10,1),(12,25),(12,26)}  # fixed-date Nigerian holidays
FEATS = ["dow","month","doy","week","weekend","holiday","sin_doy","cos_doy",
         "lag_1","lag_7","lag_14","lag_28","roll_7","roll_28"]
Z = {0.90: 1.28, 0.95: 1.645, 0.975: 1.96, 0.99: 2.33}

def sample_data(days=730, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    rows = []
    for prod, base in {"Sachet water (bag)": 400, "75cl bottle (pack)": 120, "Dispenser 19L": 35}.items():
        t = np.arange(days)
        dry = 1 + 0.25*np.isin(dates.month, [11,12,1,2,3])
        wk = np.where(dates.dayofweek >= 5, 1.15, 1.0)
        hol = np.array([1.3 if (d.month, d.day) in HOLIDAYS else 1 for d in dates])
        y = base*(1+0.0004*t)*dry*wk*hol + rng.normal(0, base*0.08, days)
        rows.append(pd.DataFrame({"date": dates, "product": prod, "units_sold": np.clip(y,0,None).round()}))
    return pd.concat(rows, ignore_index=True)

def add_features(df):
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["dow"] = d.date.dt.dayofweek; d["month"] = d.date.dt.month
    d["doy"] = d.date.dt.dayofyear; d["week"] = d.date.dt.isocalendar().week.astype(int)
    d["weekend"] = (d.dow >= 5).astype(int)
    d["holiday"] = [int((m, x) in HOLIDAYS) for m, x in zip(d.date.dt.month, d.date.dt.day)]
    d["sin_doy"] = np.sin(2*np.pi*d.doy/365.25); d["cos_doy"] = np.cos(2*np.pi*d.doy/365.25)
    for lag in (1, 7, 14, 28):
        d[f"lag_{lag}"] = d.units_sold.shift(lag)
    d["roll_7"] = d.units_sold.shift(1).rolling(7).mean()
    d["roll_28"] = d.units_sold.shift(1).rolling(28).mean()
    return d

def new_model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)

def cross_validate(d, splits=5):
    X, y = d[FEATS], d.units_sold
    maes, rmses, resid = [], [], []
    for tr, te in TimeSeriesSplit(n_splits=splits).split(X):
        p = new_model().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        maes.append(mean_absolute_error(y.iloc[te], p))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], p)))
        resid.extend(y.iloc[te] - p)
    return float(np.mean(maes)), float(np.mean(rmses)), float(np.std(resid))

def forecast(hist, model, horizon):
    h = hist[["date","units_sold"]].copy(); out = []
    for _ in range(horizon):
        nxt = pd.DataFrame({"date": [h.date.max() + pd.Timedelta(days=1)], "units_sold": [np.nan]})
        f = add_features(pd.concat([h, nxt], ignore_index=True)).iloc[[-1]]
        yhat = max(0.0, float(model.predict(f[FEATS])[0]))
        nxt["units_sold"] = yhat
        h = pd.concat([h, nxt], ignore_index=True); out.append((nxt.date.iloc[0], yhat))
    return pd.DataFrame(out, columns=["date","forecast"])

def inventory_plan(fc, sigma, lead, service, on_hand):
    lead_demand = fc.forecast.head(lead).sum()
    safety = Z[service]*sigma*np.sqrt(lead)
    rop = lead_demand + safety
    days_left = int((fc.forecast.cumsum() < on_hand).sum())
    return dict(safety=safety, rop=rop, days_left=days_left)

def run(df, product, horizon=30, lead=5, service=0.95, on_hand=3000):
    d = add_features(df[df["product"] == product].groupby("date", as_index=False).units_sold.sum()).dropna()
    mae, rmse, sigma = cross_validate(d)
    model = new_model().fit(d[FEATS], d.units_sold)
    fc = forecast(d, model, horizon)
    return d, fc, dict(mae=mae, rmse=rmse, sigma=sigma, **inventory_plan(fc, sigma, lead, service, on_hand))
