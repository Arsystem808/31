# app.py
import os, pathlib, sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# --- локальные модули ---
BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE))  # чтобы видеть пакет core/

from core.data_loader import DataLoader
from core.strategy import compute_signal
from core.llm import build_rationale  # локальный офлайн NLG (без GPT)

# --- UI ---
st.set_page_config(page_title="AI Trading — Final App", layout="wide")
st.title("AI Trading — Final App")
st.caption("Данные: Polygon → Yahoo → CSV. Стратегия: кастом. Описание — офлайн (без GPT) с выбором детализации.")

# Настройки тикеров и горизонта
default_tickers = os.getenv("DEFAULT_TICKERS", "QQQ,AAPL,MSFT,NVDA")
tickers = st.text_input("Tickers (через запятую)", value=default_tickers).upper()
symbols = [t.strip() for t in tickers.split(",") if t.strip()]

h_map = {"Краткосрок":"short","Среднесрок":"swing","Долгосрок":"position"}
horizon_ui = st.selectbox("Горизонт", list(h_map.keys()), index=1)
horizon = h_map[horizon_ui]

# Степень детализации описания (новое)
detail = st.selectbox(
    "Степень детализации описания",
    ["Коротко", "Стандарт", "Подробно"],
    index=1
)

# Выбор тикера
try:
    idx = symbols.index("QQQ")
except ValueError:
    idx = 0 if symbols else 0
symbol = st.selectbox("Тикер", symbols if symbols else ["QQQ"], index=idx)

# --- Данные и расчёт ---
loader = DataLoader()

colA, colB = st.columns([1,2])
with colA:
    if st.button("Сгенерировать сигнал"):
        try:
            fetched = loader.history(symbol, period="6mo", interval="1d")
            st.session_state["source"] = fetched.source
            st.session_state["df"] = fetched.df

            # расчёт сигнала
            sig = compute_signal(fetched.df, symbol, horizon)

            # добавим источник в сигнал — офлайн-описание использует его в тексте
            sig["source"] = fetched.source

            st.session_state["signal"] = sig
        except Exception as e:
            st.error(str(e))

with colB:
    sig = st.session_state.get("signal")
    df: pd.DataFrame | None = st.session_state.get("df")
    source = st.session_state.get("source", "—")

    if sig and df is not None:
        st.subheader(f"{sig['symbol']} — {horizon_ui} | источник: {source}")

        # Бейдж действия
        color = "#16a34a" if sig["action"]=="BUY" else ("#dc2626" if sig["action"]=="SHORT" else "#6b7280")
        st.markdown(
            f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};"
            f"color:white;font-weight:600'>{sig['action']}</div>",
            unsafe_allow_html=True
        )

        st.write("")
        # Метрики уровней
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry", f"{sig['entry']:.2f}")
        m2.metric("TP1", f"{sig['tp1']:.2f}")
        m3.metric("TP2", f"{sig['tp2']:.2f}")
        m4.metric("SL", f"{sig['sl']:.2f}")

        d1, d2, d3 = st.columns(3)
        d1.metric("Ключевая отметка", f"{sig['key_mark']:.2f}")
        d2.metric("Верхняя зона", f"{sig['upper_zone']:.2f}")
        d3.metric("Нижняя зона", f"{sig['lower_zone']:.2f}")

        st.metric("Confidence", f"{sig['confidence']:.2f}")

        # --- График ---
        fig = go.Figure([
            go.Candlestick(
                x=df["Date"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"]
            )
        ])

        # Линии уровней (включая пивоты, если есть)
        lines = {
            "Entry": sig["entry"], "TP1": sig["tp1"], "TP2": sig["tp2"], "SL": sig["sl"],
            "Ключевая отметка": sig["key_mark"],
            "Верхняя зона": sig["upper_zone"], "Нижняя зона": sig["lower_zone"],
            "P": sig.get("pivot_P", None), "R1": sig.get("R1", None), "R2": sig.get("R2", None), "R3": sig.get("R3", None),
            "S1": sig.get("S1", None), "S2": sig.get("S2", None), "S3": sig.get("S3", None)
        }
        colors = {
            "Entry":"#2563eb","TP1":"#16a34a","TP2":"#16a34a","SL":"#dc2626",
            "Ключевая отметка":"#6b7280","Верхняя зона":"#f59e0b","Нижняя зона":"#10b981",
            "P":"#9ca3af","R1":"#a78bfa","R2":"#a78bfa","R3":"#a78bfa",
            "S1":"#f472b6","S2":"#f472b6","S3":"#f472b6"
        }
        for label, y in lines.items():
            if y is None: 
                continue
            fig.add_hline(
                y=y, line_width=1, line_dash="dot",
                line_color=colors.get(label, "#999"),
                annotation_text=label, annotation_position="top left"
            )

        # Подсветка TP / SL зон
        try:
            x0 = df["Date"].iloc[-min(len(df), 40)]
            x1 = df["Date"].iloc[-1]
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=min(sig["entry"], sig["tp2"]), y1=max(sig["entry"], sig["tp2"]),
                fillcolor="rgba(34,197,94,0.08)", line=dict(width=0), layer="below"
            )
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=min(sig["sl"], sig["entry"]), y1=max(sig["sl"], sig["entry"]),
                fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), layer="below"
            )
        except Exception:
            pass

        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=460, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Текстовая аналитика (офлайн) ---
        text = build_rationale(sig["symbol"], horizon_ui, sig, detail=detail)
        st.write(text)

        with st.expander("Последние строки данных"):
            st.dataframe(df.tail(12))
    else:
        st.info("Нажмите «Сгенерировать сигнал». Если Polygon/Yahoo недоступны — возьмём demo CSV.")

    
