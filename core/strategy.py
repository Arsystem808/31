from __future__ import annotations
import pandas as pd, numpy as np
from typing import Literal, TypedDict

Horizon = Literal["short","swing","position"]
Action = Literal["BUY","SHORT","WAIT"]

class Signal(TypedDict):
    symbol: str
    horizon: Horizon
    action: Action
    confidence: float
    entry: float; tp1: float; tp2: float; sl: float
    key_mark: float; upper_zone: float; lower_zone: float

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(h-l, np.maximum((h-pc).abs(), (l-pc).abs()))
    w = min(window, max(5, len(df)//2))
    return tr.rolling(w).mean().fillna(tr.ewm(span=w, adjust=False).mean())

def _landmarks(df: pd.DataFrame) -> pd.DataFrame:
    h1, l1, c1 = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    key_mark = (h1 + l1 + c1) / 3.0
    upper = 2*key_mark - l1
    lower = 2*key_mark - h1
    return pd.DataFrame({"key_mark":key_mark, "upper_zone":upper, "lower_zone":lower})

def compute_signal(df: pd.DataFrame, symbol: str, horizon: Horizon) -> Signal:
    if df is None or df.empty:
        raise ValueError("empty df")
    df = df.copy()
    df["ATR14"] = _atr(df, 14)
    df = pd.concat([df, _landmarks(df)], axis=1)
    x = df.iloc[-1]

    px = float(x["Close"])
    a  = float(x["ATR14"] or 0.0)
    KM = float(x["key_mark"] or px)
    UZ = float(x["upper_zone"] or px)
    LZ = float(x["lower_zone"] or px)

    if horizon == "short":
        scale = 2.0; tp1m, tp2m, slm = 0.5, 1.0, 0.8
    elif horizon == "position":
        scale = 3.5; tp1m, tp2m, slm = 0.8, 1.8, 1.2
    else:
        scale = 2.8; tp1m, tp2m, slm = 0.6, 1.4, 1.0

    score = 0.5 if a <= 0 else max(0.0, min(1.0, 0.5 + (px - KM)/(scale*a)))
    action: Action = "BUY" if score >= 0.6 else ("SHORT" if score <= 0.4 else "WAIT")

    if action == "BUY":
        entry = px; tp1 = max(px+tp1m*a, UZ); tp2 = max(px+tp2m*a, KM+(UZ-KM)*1.5); sl = px - slm*a
        assert tp2 >= tp1 >= entry >= sl
    elif action == "SHORT":
        entry = px; tp1 = min(px-tp1m*a, LZ); tp2 = min(px-tp2m*a, KM-(KM-LZ)*1.5); sl = px + slm*a
        assert tp2 <= tp1 <= entry <= sl
    else:
        entry = px; tp1 = px+(tp1m+0.2)*a; tp2 = px+(tp2m+0.4)*a; sl = px-(slm-0.2)*a

    return {
        "symbol": symbol, "horizon": horizon, "action": action, "confidence": round(float(score),2),
        "entry": round(entry,2), "tp1": round(tp1,2), "tp2": round(tp2,2), "sl": round(sl,2),
        "key_mark": round(KM,2), "upper_zone": round(UZ,2), "lower_zone": round(LZ,2)
    }
