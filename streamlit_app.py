import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nse_analysis_api import analyze_stock

st.set_page_config(page_title="NSE Stock Analysis", page_icon="📈", layout="wide")
st.title("📈 NSE Stock Analysis Dashboard")
st.caption("Technical indicators and basic fundamentals. Educational use only—not investment advice.")

with st.sidebar:
    st.header("Stock selection")
    symbol = st.text_input("NSE symbol", value="RELIANCE", help="Enter TCS, INFY, RELIANCE, HDFCBANK, etc.")
    analyze_button = st.button("Analyze stock", type="primary", use_container_width=True)
    st.caption("Data source: Yahoo Finance. Cloud data access can sometimes be delayed or blocked.")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if analyze_button or st.session_state.analysis is None:
    if symbol.strip():
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            try:
                st.session_state.analysis = analyze_stock(symbol)
                st.session_state.error = None
            except Exception as exc:
                st.session_state.error = str(exc)
                st.session_state.analysis = None

if st.session_state.get("error"):
    st.error(st.session_state.error)
    st.info("Check the symbol, internet/data availability, and the requirements.txt file.")
    st.stop()

data = st.session_state.analysis
if not data:
    st.stop()

st.subheader(f"{data['company']} ({data['symbol']})")
st.caption(f"Latest available data date: {data['as_of']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest price (₹)", f"{data['price']:,.2f}" if data["price"] is not None else "N/A",
          f"{data['daily_change_pct']:.2f}%" if data["daily_change_pct"] is not None else None)
c2.metric("Signal", data["signal"])
c3.metric("Signal score", f"{data['score']} / {data['score_out_of']}")
c4.metric("RSI (14)", f"{data['technicals']['RSI14']:.2f}" if data["technicals"]["RSI14"] is not None else "N/A")

overview, chart_tab, risk_tab, performance_tab, fundamentals_tab = st.tabs(
    ["Overview", "Chart", "Risk", "Performance", "Fundamentals"]
)

with overview:
    st.markdown("### Why this signal?")
    for reason in data["reasons"]:
        st.write("•", reason)
    st.info("The signal is a simple rule-based summary of four indicators, not a recommendation to buy or sell.")
    st.markdown("### Technical indicators")
    tech_df = pd.DataFrame([data["technicals"]]).T.reset_index()
    tech_df.columns = ["Indicator", "Latest value"]
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

with chart_tab:
    hist = pd.DataFrame(data["history"])
    if not hist.empty:
        hist["Date"] = pd.to_datetime(hist["Date"])
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, row_heights=[0.60, 0.20, 0.20],
                            subplot_titles=("Price and moving averages", "RSI (14)", "MACD"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["SMA50"], name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["SMA200"], name="SMA 200"), row=1, col=1)
        # RSI/MACD are fetched from the analysis history.
        # The API chart history includes RSI and MACD values.
        # Add them using the same shared date axis.
        # RSI reference levels:
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["RSI14"], name="RSI 14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MACD_Signal"], name="MACD signal"), row=3, col=1)
        fig.update_layout(height=800, hovermode="x unified", legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No chart history available.")

with risk_tab:
    risk = data["risk"]
    a, b, c = st.columns(3)
    a.metric("ATR-based stop estimate (₹)", f"{risk['estimated_atr_stop_2x']:,.2f}" if risk["estimated_atr_stop_2x"] is not None else "N/A")
    b.metric("60-day support estimate (₹)", f"{risk['support_60d']:,.2f}" if risk["support_60d"] is not None else "N/A")
    c.metric("60-day resistance estimate (₹)", f"{risk['resistance_60d']:,.2f}" if risk["resistance_60d"] is not None else "N/A")
    d, e = st.columns(2)
    d.metric("Maximum drawdown in loaded period", f"{risk['maximum_drawdown_pct']:.2f}%" if risk["maximum_drawdown_pct"] is not None else "N/A")
    e.metric("Annualized 20-day volatility", f"{data['technicals']['Annualized Volatility 20D']:.2f}%" if data["technicals"]["Annualized Volatility 20D"] is not None else "N/A")
    st.caption("Support, resistance, and ATR stop estimates are simple historical calculations, not guaranteed levels.")

with performance_tab:
    perf = data["performance_pct"]
    perf_df = pd.DataFrame([{"Period": k.replace("_", " ").title(),
                             "Return (%)": v} for k, v in perf.items()])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption("Approximate returns use trading-day counts (about 21 trading days per month). Nifty 50 comparison is not included in this starter version.")

with fundamentals_tab:
    fund_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in data["fundamentals"].items()])
    st.dataframe(fund_df, use_container_width=True, hide_index=True)
    st.caption("Fundamental data may be incomplete for some symbols or unavailable from the data provider.")

st.divider()
st.caption(data["disclaimer"])
