import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Startup Funding Intelligence",
    page_icon="🚀",
    layout="wide"
)

API = "https://india-startup-intelligence-production.up.railway.app"

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🚀 India Startup Funding Intelligence")
st.markdown("Explore startup funding trends across India from 2015–2024")
st.divider()

# ── SECTION 1 — KPI CARDS ─────────────────────────────────────────────────────
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

try:
    r = requests.get(f"{API}/startups?limit=5000").json()
    total_startups = r["count"]

    r2 = requests.get(f"{API}/trends").json()
    trends_df = pd.DataFrame(r2["data"])
    total_funding = trends_df["total_funding_million_usd"].sum()

    r3 = requests.get(f"{API}/cities").json()
    top_city = pd.DataFrame(r3["data"])["city"].iloc[0]

    r4 = requests.get(f"{API}/sectors").json()
    top_sector = pd.DataFrame(r4["data"])["sector"].iloc[0]

    col1.metric("Total Startups", f"{total_startups:,}")
    col2.metric("Total Funding", f"${total_funding:,.0f}M")
    col3.metric("Top City", top_city)
    col4.metric("Top Sector", top_sector)

except Exception as e:
    st.error(f"Could not connect to API: {e}")
    st.stop()

st.divider()

# ── SECTION 2 — TRENDS CHART ──────────────────────────────────────────────────
st.subheader("Funding Trends by Year")

if not trends_df.empty:
    fig = px.bar(
        trends_df,
        x="year",
        y="total_funding_million_usd",
        labels={"total_funding_million_usd": "Total Funding (Million USD)", "year": "Year"},
        color="total_funding_million_usd",
        color_continuous_scale="Blues"
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── SECTION 3 — TOP SECTORS & CITIES ─────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top Sectors by Deal Count")
    sectors_df = pd.DataFrame(r4["data"])
    fig2 = px.bar(
        sectors_df,
        x="deals",
        y="sector",
        orientation="h",
        color="deals",
        color_continuous_scale="Greens"
    )
    fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("Top Cities by Deal Count")
    cities_df = pd.DataFrame(r3["data"])
    fig3 = px.pie(
        cities_df,
        values="deals",
        names="city",
        hole=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── SECTION 4 — STARTUP EXPLORER ─────────────────────────────────────────────
st.subheader("Startup Explorer")

col_s, col_c = st.columns(2)
with col_s:
    sector_filter = st.text_input("Filter by Sector", placeholder="e.g. FinTech")
with col_c:
    city_filter = st.text_input("Filter by City", placeholder="e.g. Bangalore")

params = {"limit": 100}
if sector_filter:
    params["sector"] = sector_filter
if city_filter:
    params["city"] = city_filter

explore_r = requests.get(f"{API}/startups", params=params).json()
explore_df = pd.DataFrame(explore_r["data"])

st.markdown(f"Showing **{explore_r['count']}** results")
st.dataframe(explore_df, use_container_width=True)

st.divider()

# ── SECTION 5 — PREDICT ───────────────────────────────────────────────────────
st.subheader("🔮 Predict Fundraising Likelihood")
st.markdown("Enter startup details to predict whether it's likely to raise again.")

col_p1, col_p2 = st.columns(2)
with col_p1:
    p_sector = st.text_input("Sector", value="FinTech")
    p_city   = st.text_input("City",   value="Bangalore")
with col_p2:
    p_type   = st.selectbox("Investment Type", [
        "Seed Round", "Series A", "Series B", "Series C",
        "Series D", "Pre-series A", "Venture Round",
        "Private Equity Round", "Debt Funding"
    ])
    p_amount = st.number_input("Amount Raised (USD)", min_value=0, value=5000000, step=500000)

if st.button("Predict", type="primary"):
    pred_r = requests.get(f"{API}/predict", params={
        "sector": p_sector,
        "city": p_city,
        "investment_type": p_type,
        "amount_usd": p_amount
    }).json()

    prob = pred_r["probability"]
    label = pred_r["prediction"]

    if prob >= 0.5:
        st.success(f"✅ {label} — Probability: {prob:.1%}")
    else:
        st.warning(f"⚠️ {label} — Probability: {prob:.1%}")

st.divider()
st.caption("Built with FastAPI + XGBoost + Streamlit | Data: Kaggle (2015–2024)")