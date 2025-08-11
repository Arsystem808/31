from __future__ import annotations
import os, json, random

def _human_fallback(symbol: str, horizon_ui: str, p: dict) -> str:
    # чуть более «живой» тон без ботовщины
    fragments = [
        f"{symbol}: рынок держится уверенно, без резких перегибов.",
        f"Смотрим на реакцию у ключевой отметки {p['key_mark']}.",
        f"Рабочий план: вход {p['entry']}, стоп {p['sl']}, цели {p['tp1']} / {p['tp2']}.",
        f"Диапазон внимания {p['lower_zone']}–{p['upper_zone']}, горизонт — {horizon_ui.lower()}.",
        f"По ощущениям, дисбаланс пока на стороне движения, но риски контролируем.",
    ]
    random.shuffle(fragments)
    return " ".join(fragments[:4])

def build_rationale(symbol: str, horizon_ui: str, payload: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    prompt = f"""
Сформируй короткий, живой разбор, без канцелярита и без перечисления индикаторов.
Тикер: {symbol}
Горизонт: {horizon_ui}
Данные (не зачитывай дословно, это подсказки): {json.dumps(payload, ensure_ascii=False)}
Требования: 4–6 предложений, спокойный тон. Упомяни план (Entry/SL/TP1/TP2),
ключевую отметку и рабочий диапазон (нижняя/верхняя зона), оцени уверенность как «низкая/средняя/высокая».
Не обещай доходности, не используй жаргон. Избегай слов "RSI/MACD/pivot".
"""
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                max_tokens=220,
            )
            text = rsp.choices[0].message.content.strip()
            if text: 
                return text
        except Exception:
            pass

    # fallback, если ключа нет или запрос упал
    return _human_fallback(symbol, horizon_ui, payload)
