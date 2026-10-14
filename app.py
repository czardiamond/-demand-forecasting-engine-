import pandas as pd, streamlit as st
from engine import sample_data, run

st.set_page_config(page_title="Demand Forecasting Engine", layout="wide")
st.title("Demand Forecasting & Inventory Optimization")

up = st.sidebar.file_uploader("CSV: date, product, units_sold", type="csv")
df = pd.read_csv(up, parse_dates=["date"]) if up else st.cache_data(sample_data)()
product = st.sidebar.selectbox("Product", sorted(df["product"].unique()))
horizon = st.sidebar.slider("Forecast days", 7, 90, 30)
lead = st.sidebar.number_input("Supplier lead time (days)", 1, 60, 5)
service = st.sidebar.select_slider("Service level", [0.90, 0.95, 0.975, 0.99], 0.95)
on_hand = st.sidebar.number_input("Stock on hand (units)", 0, 10_000_000, 3000)

d, fc, r = run(df, product, horizon, lead, service, on_hand)
c = st.columns(5)
c[0].metric("CV MAE", f"{r['mae']:,.1f}"); c[1].metric("CV RMSE", f"{r['rmse']:,.1f}")
c[2].metric("Safety stock", f"{r['safety']:,.0f}"); c[3].metric("Reorder point", f"{r['rop']:,.0f}")
c[4].metric("Days of stock left", f"{r['days_left']}+" if r['days_left'] == horizon else r['days_left'])
if on_hand <= r["rop"]:
    st.error(f"Reorder now: stock ({on_hand:,}) is at or below the reorder point ({r['rop']:,.0f}).")
else:
    st.success(f"Stock is above the reorder point. Reorder in about {max(0, r['days_left'] - lead)} days.")
st.line_chart(pd.concat([d.set_index("date").units_sold.tail(120).rename("actual"),
                         fc.set_index("date").forecast], axis=1))
st.subheader("Forecast table")
st.dataframe(fc.assign(forecast=fc.forecast.round(0)), use_container_width=True)
st.download_button("Download forecast CSV", fc.to_csv(index=False), f"{product}_forecast.csv")
