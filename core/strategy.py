# core/strategy.py
from __future__ import annotations
import pandas as pd, numpy as np
from typing import Literal, TypedDict

Horizon = Literal["short","swing","position"]
Action  = Literal["BUY","SHORT","WAIT"]

class Signal(TypedDict):
    symbol: str
    horizon: Horizon
    action: Action
    confidence: float
    entry: float; tp1: float; tp2: float; sl: float
    key_mark: float; upper_zone: float; lower_zone: float

# ---------- helpers ----------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    r_up  = up.ewm(alpha=1/period, adjust=False).mean()
    r_dn  = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    return rsi.fillna(50)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(h-l, np.maximum((h-pc).abs(), (l-pc).abs()))
    w = min(period, max(5, len(df)//2))
    return tr.rolling(w).mean().fillna(tr.ewm(span=w, adjust=False).mean())

def _landmarks(df: pd.DataFrame) -> pd.DataFrame:
    h1, l1, c1 = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    key_mark   = (h1 + l1 + c1)/3.0
    upper_zone = 2*key_mark - l1
    lower_zone = 2*key_mark - h1
    return pd.DataFrame({"key_mark":key_mark, "upper_zone":upper_zone, "lower_zone":lower_zone})

# ---------- main ----------
def compute_signal(df: pd.DataFrame, symbol: str, horizon: Horizon) -> Signal:
    if df is None or df.empty:
        raise ValueError("empty dataframe")

    df = df.copy()
    df["EMA_fast"] = _ema(df["Close"], 20)
    df["EMA_mid"]  = _ema(df["Close"], 50)
    df["EMA_slow"] = _ema(df["Close"], 200)
    df["RSI14"]    = _rsi(df["Close"], 14)
    df["ATR14"]    = _atr(df, 14)
    df = pd.concat([df, _landmarks(df)], axis=1)

    # --- фондовые пресеты ---
    if horizon == "short":
        k_tp1, k_tp2, k_sl = 0.6, 1.2, 0.9
        trend_w = 0.6
    elif horizon == "position":
        k_tp1, k_tp2, k_sl = 1.0, 2.2, 1.4
        trend_w = 0.7
    else:  # swing
        k_tp1, k_tp2, k_sl = 0.8, 1.6, 1.0
        trend_w = 0.65

    x  = df.iloc[-1]
    px = float(x["Close"])
    atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) else 0.0
    km = float(x["key_mark"]) if pd.notna(x["key_mark"]) else px
    uz = float(x["upper_zone"]) if pd.notna(x["upper_zone"]) else px
    lz = float(x["lower_zone"]) if pd.notna(x["lower_zone"]) else px

    # тренд
    trend_pos = (x["EMA_fast"] > x["EMA_mid"] > x["EMA_slow"]) and (df["EMA_mid"].iloc[-1] > df["EMA_mid"].iloc[-5])
    trend_neg = (x["EMA_fast"] < x["EMA_mid"] < x["EMA_slow"]) and (df["EMA_mid"].iloc[-1] < df["EMA_mid"].iloc[-5])

    # положение цены относительно ключевой отметки
    pos_vs_key = px - km
    rsi = float(x["RSI14"]) if pd.notna(x["RSI14"]) else 50.0
    rsi_bias_up, rsi_bias_down = (rsi <= 68), (rsi >= 32)

    buy_bias   = trend_pos and (pos_vs_key >= 0) and rsi_bias_up
    short_bias = trend_neg and (pos_vs_key <= 0) and rsi_bias_down

    def _conf(sign: int) -> float:
        if atr <= 0:
            base = 0.5
        else:
            base = 0.5 + sign * (pos_vs_key / (2.8*atr))
        base = float(np.clip(base, 0.0, 1.0))
        trend_part = trend_w if (trend_pos and sign>0) or (trend_neg and sign<0) else 0.0
        return float(np.clip(0.5*base + 0.5*trend_part, 0.0, 1.0))

    if buy_bias and atr > 0:
        action = "BUY"; confidence = _conf(+1)
        entry = px
        tp1   = px + max(k_tp1*atr, 0.01)
        tp2   = px + max(k_tp2*atr, 0.02)
        sl    = px - max(k_sl*atr, 0.01)
        tp1, tp2 = max(tp1, px), max(tp2, tp1)
        sl       = min(sl, px)
    elif short_bias and atr > 0:
        action = "SHORT"; confidence = _conf(-1)
        entry = px
        tp1   = px - max(k_tp1*atr, 0.01)
        tp2   = px - max(k_tp2*atr, 0.02)
        sl    = px + max(k_sl*atr, 0.01)
        tp1, tp2 = min(tp1, px), min(tp2, tp1)
        sl       = max(sl, px)
    else:
        action = "WAIT"
        confidence = float(np.clip(0.5 + (pos_vs_key/(4.0*atr) if atr>0 else 0.0), 0.0, 1.0))
        entry = px
        tp1   = px + 0.6*atr
        tp2   = px + 1.2*atr
        sl    = px - 0.9*atr

    return {
        "symbol": symbol, "horizon": horizon, "action": action,
        "confidence": round(confidence, 2),
        "entry": round(float(entry), 2),
        "tp1":   round(float(tp1), 2),
        "tp2":   round(float(tp2), 2),
        "sl":    round(float(sl), 2),
        "key_mark":   round(float(km), 2),
        "upper_zone": round(float(uz), 2),
        "lower_zone": round(float(lz), 2),
    }
