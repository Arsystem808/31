# core/llm.py — офлайн NLG без GPT
from __future__ import annotations
import random

def _verbal_conf(c: float) -> str:
    if c < 0.45: return "низкая"
    if c < 0.7:  return "средняя"
    return "высокая"

def _dir_word(action: str) -> str:
    return {"BUY":"покупку","SHORT":"шорт","WAIT":"ожидание"}.get(action, "ожидание")

def _ctx_intro(symbol: str, action: str, src: str, horizon_ui: str) -> str:
    variants = [
        f"{symbol}: картина аккуратная, без экстремальных скачков. Источник данных — {src.lower()}, горизонт — {horizon_ui.lower()}.",
        f"{symbol}: рынок ведёт себя ровно; работаем со свежими котировками ({src}). Горизонт — {horizon_ui.lower()}.",
        f"{symbol}: тон нейтрально-умеренный, без перекосов. Данные: {src.lower()}, режим — {horizon_ui.lower()}."
    ]
    # Чуть другой заход, если сигнал не WAIT
    if action in ("BUY","SHORT"):
        variants += [
            f"{symbol}: условия для сделки выглядят адекватно. Используем {src.lower()}, горизонт — {horizon_ui.lower()}.",
            f"{symbol}: на текущей выборке есть сетап под {_dir_word(action)}. Источник — {src}, горизонт — {horizon_ui.lower()}."
        ]
    return random.choice(variants)

def _levels_block(p: dict) -> str:
    return (f"План: вход {p['entry']}, стоп {p['sl']}, цели {p['tp1']} / {p['tp2']}. "
            f"Рабочий коридор {p['lower_zone']}–{p['upper_zone']}, опорная отметка {p['key_mark']}.")

def _pivots_block(p: dict) -> str:
    if all(k in p for k in ("pivot_P","R1","R2","R3","S1","S2","S3")):
        return (f"Контрольные уровни: P={p['pivot_P']}, R1={p['R1']}, R2={p['R2']}, R3={p['R3']}, "
                f"S1={p['S1']}, S2={p['S2']}, S3={p['S3']}.")
    return ""

def _followups(action: str) -> str:
    if action == "BUY":
        return "Если цена закрепится выше ближайшего сопротивления — держим курс на вторую цель; при слабой реакции — сокращаем риск."
    if action == "SHORT":
        return "Если импульс вниз сохранится у ближайшей поддержки — удерживаем позицию до второй цели; откат — защищаемся."
    return "Наблюдаем за реакцией у ближайших уровней; чёткий пробой задаст следующую фазу."

def build_rationale(symbol: str, horizon_ui: str, payload: dict, detail: str = "Стандарт") -> str:
    """
    Локальная генерация текста без внешних API.
    detail: 'Коротко' | 'Стандарт' | 'Подробно'
    """
    action = payload.get("action","WAIT")
    source = payload.get("source","market")
    conf_t = _verbal_conf(float(payload.get("confidence", 0.5)))

    intro = _ctx_intro(symbol, action, source, horizon_ui)
    plan  = _levels_block(payload)
    pivs  = _pivots_block(payload)
    tail  = _followups(action)

    # Три уровня детализации
    if detail == "Коротко":
        parts = [
            f"Сценарий: {_dir_word(action)}; уверенность {conf_t}.",
            plan,
            tail
        ]
    elif detail == "Подробно":
        nuances = [
            "По тонусу ленты перекос минимальный: важна реакция на пробой/отбой.",
            "Риски контролируем через стоп и частичную фиксацию у первой цели.",
            "Сильные движения часто начинаются после выхода из узкого коридора."
        ]
        parts = [intro, f"Сценарий: {_dir_word(action)}; уверенность {conf_t}.", plan, pivs or "", random.choice(nuances), tail]
    else:  # Стандарт
        parts = [intro, f"Сценарий: {_dir_word(action)}; уверенность {conf_t}.", plan, pivs or "", tail]

    # Собираем текст, чистим двойные пробелы
    text = " ".join([s for s in parts if s]).replace("  ", " ").strip()
    return text
