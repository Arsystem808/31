from __future__ import annotations
import os, json

def build_rationale(symbol: str, horizon_ui: str, payload: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prompt = f"""Сформулируй краткий разбор для инвестора без тех. терминов.
Тикер: {symbol}
Горизонт: {horizon_ui}
Данные: {json.dumps(payload, ensure_ascii=False)}
Требования: 3–5 предложений, нейтральный стиль, без слов pivot/MACD/RSI.
Упомяни план (Entry/SL/TP1/TP2), волатильность (ATR), ключевую отметку,
верхнюю/нижнюю зоны и диапазон наблюдения. Не обещай прибыль.
"""
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3,
            )
            return rsp.choices[0].message.content.strip()
        except Exception:
            pass
    return (f"Для {symbol} сигнал сформирован с учётом текущего движения и волатильности. "
            f"План: вход {payload['entry']} • стоп {payload['sl']} • цели {payload['tp1']}/{payload['tp2']}. "
            f"Ориентируемся на ключевую отметку {payload['key_mark']} и диапазон {payload['lower_zone']}–{payload['upper_zone']} "
            f"на горизонте {horizon_ui}. Confidence {payload['confidence']}.") 
