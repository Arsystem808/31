import os, pathlib, sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE))

from core.data_loader import DataLoader
from core.strategy import compute_signal
from core.llm import build_rationale

st.set_page_config(page_title="AI Trading — Final App", layout="wide")
st.title("AI Trading — Final App")
st.caption("Данные: Polygon→Yahoo→CSV. Стратегия: кастом. LLM-анализ — опционально.")

default_tickers = os.getenv("DEFAULT_TICKERS", "QQQ,AAPL,MSFT,NVDA")
tickers = st.text_input("Tickers", value=default_tickers).upper()
symbols = [t.strip() for t in tickers.split(",") if t.strip()]
h_map = {"Краткосрок":"short","Среднесрок":"swing","Долгосрок":"position"}
horizon_ui = st.selectbox("Горизонт", list(h_map.keys()), index=1)
horizon = h_map[horizon_ui]

try:
    idx = symbols.index("QQQ")
except ValueError:
    idx = 0
symbol = st.selectbox("Тикер", symbols, index=idx if symbols else 0)

loader = DataLoader()

colA, colB = st.columns([1,2])
with colA:
    if st.button("Сгенерировать сигнал"):
        try:
            fetched = loader.history(symbol, period="6mo", interval="1d")
            st.session_state["source"] = fetched.source
            st.session_state["df"] = fetched.df
            st.session_state["signal"] = compute_signal(fetched.df, symbol, horizon)
        except Exception as e:
            st.error(str(e))

with colB:
    sig = st.session_state.get("signal")
    df: pd.DataFrame | None = st.session_state.get("df")
    source = st.session_state.get("source", "—")
    if sig and df is not None:
        st.subheader(f"{sig['symbol']} — {horizon_ui} | source: {source}")
        color = "#16a34a" if sig["action"]=="BUY" else ("#dc2626" if sig["action"]=="SHORT" else "#6b7280")
        st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:600'>{sig['action']}</div>", unsafe_allow_html=True)

        st.write(" ")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry", f"{sig['entry']:.2f}")
        m2.metric("TP1", f"{sig['tp1']:.2f}")
        m3.metric("TP2", f"{sig['tp2']:.2f}")
        m4.metric("SL", f"{sig['sl']:.2f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Ключевая отметка", f"{sig['key_mark']:.2f}")
        d2.metric("Верхняя зона сопротивления", f"{sig['upper_zone']:.2f}")
        d3.metric("Нижняя зона спроса", f"{sig['lower_zone']:.2f}")
        st.metric("Confidence", f"{sig['confidence']:.2f}")

        fig = go.Figure([go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
        lines = {"Entry":sig["entry"], "TP1":sig["tp1"], "TP2":sig["tp2"], "SL":sig["sl"],
                 "Ключевая отметка":sig["key_mark"], "Верхняя зона сопротивления":sig["upper_zone"], "Нижняя зона спроса":sig["lower_zone"]}
        colors = {"Entry":"#2563eb","TP1":"#16a34a","TP2":"#16a34a","SL":"#dc2626",
                  "Ключевая отметка":"#6b7280","Верхняя зона сопротивления":"#f59e0b","Нижняя зона спроса":"#10b981"}
        for label, y in lines.items():
            fig.add_hline(y=y, line_width=1, line_dash="dot", line_color=colors.get(label, "#999"),
                          annotation_text=label, annotation_position="top left")
        x0 = df["Date"].iloc[-min(len(df), 40)]
        x1 = df["Date"].iloc[-1]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(sig["entry"], sig["tp2"]), y1=max(sig["entry"], sig["tp2"]),
                      fillcolor="rgba(34,197,94,0.08)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(sig["sl"], sig["entry"]), y1=max(sig["sl"], sig["entry"]),
                      fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), layer="below")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=440, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        text = build_rationale(sig["symbol"], horizon_ui, sig)
        st.write(text)

        with st.expander("Последние строки данных"):
            st.dataframe(df.tail(12))
    else:
        st.info("Нажмите «Сгенерировать сигнал». Если Polygon/Yahoo недоступны — возьмём demo CSV.")
