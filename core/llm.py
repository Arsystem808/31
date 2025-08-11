
   # core/llm.py — офлайн NLG без упоминаний "Pivot", R1/R2/S1/S2
from __future__ import annotations
import random

def _verbal_conf(c: float) -> str:
    if c < 0.45: return "низкая"
    if c < 0.7:  return "средняя"
    return "высокая"

def _dir_word(action: str) -> str:
    return {"BUY":"покупку","SHORT":"шорт","WAIT":"ожидание","CLOSE":"закрытие"}.get(action, "ожидание")

def _nearest_levels(sig: dict) -> tuple[list[float], list[float]]:
    """Возвращает 2 ориентирa сверху и 2 снизу без ярлыков уровней."""
    px = float(sig.get("entry", sig.get("Close", 0.0)))
    # соберём возможные уровни (если есть), но не называем их
    up_raw = [sig.get("R1"), sig.get("R2"), sig.get("R3"), sig.get("upper_zone"), sig.get("key_mark")]
    dn_raw = [sig.get("S1"), sig.get("S2"), sig.get("S3"), sig.get("lower_zone"), sig.get("key_mark")]
    ups = sorted([float(x) for x in up_raw if isinstance(x,(int,float)) and x and x > px])[:2]
    dns = sorted([float(x) for x in dn_raw if isinstance(x,(int,float)) and x and x < px], reverse=True)[:2]
    return ups, dns

def _intro(symbol: str, action: str, src: str, horizon_ui: str) -> str:
    starts = [
        f"{symbol}: картина спокойная, без лишней суеты. Источник — {src.lower()}, горизонт — {horizon_ui.lower()}.",
        f"{symbol}: рынок ведёт себя ровно; работаем по {src.lower()}, режим — {horizon_ui.lower()}.",
        f"{symbol}: тон умеренный, без явных перегибов. Данные — {src.lower()}, горизонт — {horizon_ui.lower()}."
    ]
    if action in ("BUY","SHORT"):
        starts += [
            f"{symbol}: условия для сделки выглядят рабочими. Источник — {src}, горизонт — {horizon_ui.lower()}.",
            f"{symbol}: есть сетап под {_dir_word(action)}. Берём котировки {src.lower()}, горизонт — {horizon_ui.lower()}."
        ]
    return random.choice(starts)

def _plan(sig: dict) -> str:
    return (f"План: вход {sig['entry']}, стоп {sig['sl']}, цели {sig['tp1']} / {sig['tp2']}. "
            f"Рабочий коридор {sig['lower_zone']}–{sig['upper_zone']}, опорная отметка {sig['key_mark']}.")

def _context_levels(sig: dict) -> str:
    ups, dns = _nearest_levels(sig)
    parts = []
    if ups:
        if len(ups) == 1:
            parts.append(f"Сверху ближайший ориентир — {ups[0]:.2f}.")
        else:
            parts.append(f"Сверху ориентиры — {ups[0]:.2f} и {ups[1]:.2f}.")
    if dns:
        if len(dns) == 1:
            parts.append(f"Снизу ближайший ориентир — {dns[0]:.2f}.")
        else:
            parts.append(f"Снизу ориентиры — {dns[0]:.2f} и {dns[1]:.2f}.")
    return " ".join(parts)

def _next_steps(action: str) -> str:
    if action == "BUY":
        return "Если цена удержится выше ближайшего сопротивления, даём ходу; при вялой реакции сокращаем риск."
    if action == "SHORT":
        return "Если давление сохранится у ближайшей поддержки, держим ход до второй цели; при отскоке — защищаемся."
    if action == "CLOSE":
        return "Логично зафиксировать результат и посмотреть на следующую расстановку сил."
    return "Наблюдаем за реакцией у ближайших ориентиров; уверенный пробой задаст следующее движение."

def build_rationale(symbol: str, horizon_ui: str, sig: dict, detail: str = "Стандарт") -> str:
    """
    Локальная генерация текста без внешних API и без раскрытия терминов методики.
    detail: 'Коротко' | 'Стандарт' | 'Подробно'
    """
    action = sig.get("action", "WAIT")
    source = sig.get("source", "market")
    conf_t = _verbal_conf(float(sig.get("confidence", 0.5)))

    intro = _intro(symbol, action, source, horizon_ui)
    plan  = _plan(sig)
    lvl   = _context_levels(sig)
    tail  = _next_steps(action)
    conf  = f"Уверенность оценки — {conf_t}."

    if detail == "Коротко":
        parts = [f"Сценарий: {_dir_word(action)}.", conf, plan, tail]
    elif detail == "Подробно":
        nuances = [
            "Следим за поведением цены в узком коридоре: импульс часто появляется на выходе.",
            "Риски контролируем через стоп и частичную фиксацию у первой цели.",
            "Фон рынка учитываем, но решения привязываем к собственным уровням."
        ]
        parts = [intro, f"Сценарий: {_dir_word(action)}.", conf, plan, lvl, random.choice(nuances), tail]
    else:  # Стандарт
        parts = [intro, f"Сценарий: {_dir_word(action)}.", conf, plan, lvl, tail]

    text = " ".join([p for p in parts if p]).replace("  ", " ").strip()
    return text
