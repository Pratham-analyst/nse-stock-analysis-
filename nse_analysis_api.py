"""
NSE Stock Analysis: reusable analysis functions + optional FastAPI endpoint.

Streamlit imports analyze_stock() directly.
Render can run this file as a FastAPI service:
    uvicorn nse_analysis_api:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

from typing import Any
import math
import pandas as pd
import numpy as np
import yfinance as yf

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # Allows the analysis functions to be used without FastAPI installed
    FastAPI = None
    HTTPException = Exception

app = FastAPI(title="NSE Stock Analysis API", version="1.0.0") if FastAPI else None


def _clean_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Please enter a stock symbol.")
    # Accept either TCS or TCS.NS
    return symbol if "." in symbol else f"{symbol}.NS"


def _safe_float(value: Any):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def analyze_stock(symbol: str, period: str = "2y") -> dict:
    """
    Fetch Yahoo Finance data and calculate common technical indicators.
    This is educational information, not investment advice.
    """
    ticker_symbol = _clean_symbol(symbol)
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period, auto_adjust=True)

    if hist is None or hist.empty:
        raise ValueError(
            f"No price history found for {ticker_symbol}. Check the symbol or try again later."
        )

    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index)
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"] if "Volume" in hist else pd.Series(index=hist.index, dtype=float)

    hist["SMA20"] = close.rolling(20).mean()
    hist["SMA50"] = close.rolling(50).mean()
    hist["SMA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    hist["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    hist["MACD"] = ema12 - ema26
    hist["MACD_Signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
    hist["MACD_Hist"] = hist["MACD"] - hist["MACD_Signal"]

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    hist["ATR14"] = true_range.rolling(14).mean()
    hist["DailyReturn"] = close.pct_change()
    hist["Volatility20"] = hist["DailyReturn"].rolling(20).std() * np.sqrt(252)

    latest = hist.iloc[-1]
    previous_close = _safe_float(close.iloc[-2]) if len(close) > 1 else None
    current_price = _safe_float(latest["Close"])
    change_pct = (
        ((current_price / previous_close) - 1) * 100
        if current_price is not None and previous_close not in (None, 0)
        else None
    )

    score = 0
    reasons = []
    if pd.notna(latest["SMA50"]) and current_price is not None:
        if current_price > latest["SMA50"]:
            score += 1
            reasons.append("Price is above the 50-day moving average.")
        else:
            reasons.append("Price is below the 50-day moving average.")
    if pd.notna(latest["SMA200"]) and current_price is not None:
        if current_price > latest["SMA200"]:
            score += 1
            reasons.append("Price is above the 200-day moving average.")
        else:
            reasons.append("Price is below the 200-day moving average.")
    if pd.notna(latest["RSI14"]):
        if 50 <= latest["RSI14"] < 70:
            score += 1
            reasons.append("RSI indicates positive momentum without being above 70.")
        elif latest["RSI14"] >= 70:
            reasons.append("RSI is 70 or higher; the stock may be overbought.")
        elif latest["RSI14"] < 30:
            reasons.append("RSI is below 30; the stock may be oversold.")
        else:
            reasons.append("RSI is below 50, indicating weaker momentum.")
    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_Signal"]):
        if latest["MACD"] > latest["MACD_Signal"]:
            score += 1
            reasons.append("MACD is above its signal line.")
        else:
            reasons.append("MACD is below its signal line.")

    signal = "Positive" if score >= 3 else "Neutral" if score == 2 else "Cautious"

    # Support/resistance are simple recent-period estimates, not guaranteed levels.
    recent = hist.tail(60)
    support = _safe_float(recent["Low"].min())
    resistance = _safe_float(recent["High"].max())
    atr = _safe_float(latest["ATR14"])
    stop_loss = current_price - 2 * atr if current_price is not None and atr is not None else None

    # Drawdown from the highest adjusted close in the requested history.
    rolling_peak = close.cummax()
    drawdown = (close / rolling_peak) - 1
    max_drawdown_pct = _safe_float(drawdown.min() * 100)

    # 1, 3, 6 and 12 month approximate performance (21 trading days/month).
    performance = {}
    for months, label in [(1, "1_month"), (3, "3_months"), (6, "6_months"), (12, "12_months")]:
        n = months * 21
        if len(close) > n and _safe_float(close.iloc[-n-1]) not in (None, 0):
            performance[label] = _safe_float((close.iloc[-1] / close.iloc[-n-1] - 1) * 100)
        else:
            performance[label] = None

    # Keep a compact chart-ready history in JSON.
    chart_columns = ["Close", "SMA50", "SMA200", "RSI14", "MACD", "MACD_Signal", "Volume"]
    chart = hist[chart_columns].tail(260).replace({np.nan: None})
    chart_records = []
    for date, row in chart.iterrows():
        record = {"Date": date.strftime("%Y-%m-%d")}
        for col in chart_columns:
            record[col] = _safe_float(row[col])
        chart_records.append(record)

    try:
        info = ticker.get_info()
    except Exception:
        info = {}
    fundamentals = {
        "Company": info.get("longName") or info.get("shortName") or ticker_symbol,
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market Cap": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Price to Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "52-week High": info.get("fiftyTwoWeekHigh"),
        "52-week Low": info.get("fiftyTwoWeekLow"),
    }

    return {
        "symbol": ticker_symbol,
        "company": fundamentals["Company"],
        "as_of": hist.index[-1].strftime("%Y-%m-%d"),
        "price": current_price,
        "previous_close": previous_close,
        "daily_change_pct": _safe_float(change_pct),
        "signal": signal,
        "score": score,
        "score_out_of": 4,
        "reasons": reasons,
        "technicals": {
            "SMA20": _safe_float(latest["SMA20"]),
            "SMA50": _safe_float(latest["SMA50"]),
            "SMA200": _safe_float(latest["SMA200"]),
            "RSI14": _safe_float(latest["RSI14"]),
            "MACD": _safe_float(latest["MACD"]),
            "MACD_Signal": _safe_float(latest["MACD_Signal"]),
            "ATR14": atr,
            "Annualized Volatility 20D": _safe_float(latest["Volatility20"] * 100),
        },
        "risk": {
            "estimated_atr_stop_2x": _safe_float(stop_loss),
            "support_60d": support,
            "resistance_60d": resistance,
            "maximum_drawdown_pct": max_drawdown_pct,
            "latest_volume": _safe_float(volume.iloc[-1]) if len(volume) else None,
        },
        "performance_pct": performance,
        "fundamentals": fundamentals,
        "history": chart_records,
        "disclaimer": "Educational analysis only; not investment advice. Data is provided by Yahoo Finance and may be delayed or unavailable.",
    }


if app is not None:
    @app.get("/")
    def root():
        return {
            "message": "NSE Stock Analysis API is running",
            "example": "/analyze/RELIANCE",
            "docs": "/docs",
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/analyze/{symbol}")
    def analyze_endpoint(symbol: str):
        try:
            return analyze_stock(symbol)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}")
