from __future__ import annotations
import os, json, random

def _human_fallback(symbol: str, horizon_ui: str, p: dict, detail: str) -> str:
    chunks = [
        f"{symbol}: динамика спокойная, без резких выбросов.",
        f"Опорная отметка {p['key_mark']}, рабочий коридор {p['lower_zone']}–{p['upper_zone']}.",
        f"План: вход {p['entry']}, стоп {p['sl']}, цели {p['tp1']} / {p['tp2']}.",
        f"Уверенность оценки — {('низкая' if p['confidence']<0.45 else 'средняя' if p['confidence']<0.7 else 'высокая')}.",
        "Следим за реакцией у ближайших уровней: откат — защищаем позицию, пробой — держим цели."
    ]
    random.shuffle(chunks)
    n = 3 if detail == "Коротко" else (4 if detail == "Стандарт" else 5)
    return " ".join(chunks[:n])

def build_rationale(symbol: str, horizon_ui: str, payload: dict, detail: str = "Стандарт") -> str:
    """
    detail: 'Коротко' | 'Стандарт' | 'Подробно'
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    word_goal = {"Коротко":"3–4", "Стандарт":"4–6", "Подробно":"6–9"}[detail]
    prompt = f"""
Собери {word_goal} предложений живого разбора для инвестора. Без жаргона и индикаторов в лоб.
Тикер: {symbol}
Горизонт: {horizon_ui}
Подсказки (не зачитывай их): {json.dumps(payload, ensure_ascii=False)}
Сделай структуру: контекст движения → ключевые уровни (ключевая отметка и рабочий коридор) →
план с Entry/SL/TP1/TP2 → что считаем сигналом отмены → что внимательно отслеживать дальше.
Избегай слов RSI/MACD/pivot. Тон спокойный, уверенный, без обещаний доходности.
"""

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":"Ты рыночный аналитик. Объясняешь просто и по делу."},
                    {"role":"user","content":prompt},
                ],
                temperature=0.6 if detail!="Подробно" else 0.7,
                max_tokens=360 if detail=="Подробно" else 260,
            )
            text = rsp.choices[0].message.content.strip()
            if text:
                return text
        except Exception:
            pass

    # Fallback, если ключа нет или запрос упал
    return _human_fallback(symbol, horizon_ui, payload, detail)

