"""Notificaciones push a Telegram (solo 'requests', sin wrappers pesados).

Lee TELEGRAM_TOKEN y TELEGRAM_CHAT_ID de os.environ o de un .env en la raiz.

Setup (una vez):
  1. En Telegram, hablale a @BotFather -> /newbot -> te da el TOKEN.
  2. Escribile algo a tu bot, despues abri:
     https://api.telegram.org/bot<TOKEN>/getUpdates  -> ahi ves tu CHAT_ID.
  3. Pone ambos en el archivo .env de la raiz (ver .env.example).

Funciones:
  enviar_mensaje(texto)                          -> manda texto crudo (Markdown).
  enviar_alerta(partido, pick, prob, cuota, ev)  -> alerta corta de value bet.
  enviar_reporte_partido(info, cuotas)           -> reporte COMPLETO del partido.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

from src import config
from src.ev_calculator import calcular_ev


def _credenciales() -> tuple[str | None, str | None]:
    config.cargar_env()  # asegura que el .env este cargado en os.environ
    return os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def enviar_mensaje(texto: str) -> bool:
    """Envia un mensaje a Telegram. Devuelve True si salio bien."""
    token, chat = _credenciales()
    if not (token and chat):
        print("[telegram] faltan TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (ver .env)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": texto, "parse_mode": "Markdown",
                  "disable_web_page_preview": True},
            timeout=20,
        )
        if r.status_code != 200:
            print(f"[telegram] error {r.status_code}: {r.text[:120]}")
        return r.status_code == 200
    except requests.RequestException as e:
        print(f"[telegram] fallo de red: {e}")
        return False


def enviar_alerta(partido: str, pick: str, probabilidad: float, cuota: float, ev: float) -> bool:
    """Alerta corta de value bet (firma del spec)."""
    return enviar_mensaje(
        f"🚨 *VALUE BET* 🚨\n"
        f"⚽ {partido}\n"
        f"📈 Pick: *{pick}*\n"
        f"Prob modelo: *{probabilidad*100:.0f}%*  |  Cuota: *{cuota:.2f}*\n"
        f"EV: *{ev:+.1f}%*"
    )


def _horas_para(fecha_iso: str) -> str:
    try:
        ko = datetime.fromisoformat(fecha_iso)
        h = (ko - datetime.now(timezone.utc)).total_seconds() / 3600.0
        return f"en ~{h:.0f}h ({ko:%H:%M} UTC)" if h > 0 else f"{ko:%H:%M} UTC"
    except Exception:
        return ""


def enviar_reporte_partido(info: dict, cuotas: dict | None) -> bool:
    """Reporte COMPLETO de un partido para decidir desde el celular.

    Manda numeros del modelo (1X2, goles/faltas/tarjetas/corners) y, si hay cuotas, la
    comparacion contra el mercado con el EV de cada mercado. NO filtra por EV: manda todo.
    """
    r = info["res"]
    loc, vis = info["local"], info["visitante"]
    pl, pe, pv = r["prob_local"], r["prob_empate"], r["prob_visitante"]
    gl, gv = r["goles_esp"]
    marc = r["marcadores_top"][0][0] if r.get("marcadores_top") else (0, 0)

    msg = [
        f"⚽ *{loc} vs {vis}*",
        f"🕐 {_horas_para(info.get('fecha',''))}  |  Arbitro: {info.get('arbitro') or 's/d'}",
        "",
        "📊 *Modelo (1X2)*",
        f"🏠 {loc}: *{pl*100:.0f}%*   🤝 Empate: *{pe*100:.0f}%*   ✈️ {vis}: *{pv*100:.0f}%*",
        f"Goles esp: *{gl:.2f} - {gv:.2f}*   |   Marcador prob: *{marc[0]}-{marc[1]}*",
        "",
        "📋 *Otros mercados (esperado)*",
        f"⚽ Over 2.5: *{r['over_2_5_goles']*100:.0f}%*   "
        f"🟨 Tarjetas: *{r['tarjetas_esp']:.1f}*   "
        f"🚩 Faltas: *{r['faltas_esp']:.0f}*   ⛳ Corners: *{r['corners_esp']:.0f}*",
    ]

    valor = []
    if cuotas:
        msg += ["", "💰 *vs Mercado (cuota | EV)*"]
        mercados = [
            (f"{loc}", pl, cuotas.get("Home")),
            ("Empate", pe, cuotas.get("Draw")),
            (f"{vis}", pv, cuotas.get("Away")),
            ("Over 2.5", r["over_2_5_goles"], cuotas.get("Over")),
        ]
        for nombre, prob, mkt in mercados:
            if not mkt:
                continue
            ev, es_valor = calcular_ev(prob, mkt["mejor"])
            flag = "  ✅" if es_valor else ""
            msg.append(f"{nombre}: @ {mkt['mejor']:.2f}  ->  EV *{ev:+.0f}%*{flag}")
            if es_valor:
                valor.append((nombre, ev, mkt["mejor"]))

    if valor:
        valor.sort(key=lambda x: -x[1])
        msg += ["", "📈 *VALOR detectado:*"]
        msg += [f"   • *{n}* (EV {ev:+.0f}% @ {c:.2f})" for n, ev, c in valor]
        msg += ["", "⚠️ _El valor es tan confiable como el modelo (selecciones aun en validacion)._"]
    elif cuotas:
        msg += ["", "📈 Sin valor claro segun el modelo. _(igual decidi vos)_"]

    return enviar_mensaje("\n".join(msg))


if __name__ == "__main__":
    # Prueba rapida de credenciales/envio
    ok = enviar_mensaje("✅ *D-Soccer* conectado a Telegram. Listo para avisarte del Mundial.")
    print("Mensaje enviado OK" if ok else "No se pudo enviar (revisa .env / credenciales)")
