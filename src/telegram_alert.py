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
import sys
from datetime import datetime, timezone

import requests

from src import config
from src.ev_calculator import calcular_ev

SUSCRIPTORES = config.RAIZ / "suscriptores.txt"


def _recipientes() -> list[str]:
    """Chat_ids a los que mandar: el/los del .env (TELEGRAM_CHAT_ID, o TELEGRAM_CHAT_IDS separados
    por coma) MAS los de suscriptores.txt (en la raiz, gitignored). Asi sumas amigos sin tocar
    codigo: cada uno le escribe al bot, sacas su chat_id con 'python -m src.telegram_alert --quien'
    y lo agregas a suscriptores.txt (una linea por persona; 'chat_id  # Nombre' permitido)."""
    config.cargar_env()
    ids: list[str] = []
    for var in ("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_IDS"):
        ids += [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]
    if SUSCRIPTORES.exists():
        for linea in SUSCRIPTORES.read_text(encoding="utf-8").splitlines():
            cid = linea.split("#", 1)[0].strip()  # corta el comentario/nombre
            if cid:
                ids.append(cid)
    return list(dict.fromkeys(ids))  # dedupe preservando orden


def enviar_mensaje(texto: str) -> bool:
    """Envia un mensaje a TODOS los destinatarios. True si llego al menos a uno."""
    config.cargar_env()
    token = os.environ.get("TELEGRAM_TOKEN")
    chats = _recipientes()
    if not token or not chats:
        print("[telegram] falta TELEGRAM_TOKEN o no hay destinatarios (.env / suscriptores.txt)")
        return False
    ok = 0
    for chat in chats:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": texto, "parse_mode": "Markdown",
                      "disable_web_page_preview": True},
                timeout=20,
            )
            if r.status_code == 200:
                ok += 1
            else:
                print(f"[telegram] error chat {chat}: {r.status_code} {r.text[:100]}")
        except requests.RequestException as e:
            print(f"[telegram] fallo de red chat {chat}: {e}")
    print(f"[telegram] enviado a {ok}/{len(chats)} destinatarios")
    return ok > 0


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
    o245, o275, o305 = (r.get("over_24_5_faltas", 0), r.get("over_27_5_faltas", 0),
                        r.get("over_30_5_faltas", 0))

    msg_arb_faltas = r.get("msg_faltas_arbitro", "")
    msg = [
        f"⚽ *{loc} vs {vis}*",
        f"🕐 {_horas_para(info.get('fecha',''))}  |  Arbitro: {info.get('arbitro') or 's/d'}",
        "",
        # FALTAS primero: el UNICO mercado con edge validado (+4.8% vs baseline).
        "⭐ *FALTAS — nuestro edge (+4.8% validado)*",
        f"Total esperado: *{r.get('faltas_esp', 0):.0f}*   (prob. over)",
        f"   24.5: *{o245*100:.0f}%*    ·    27.5: *{o275*100:.0f}%*    ·    30.5: *{o305*100:.0f}%*",
        f"_Arbitro: {msg_arb_faltas}_" if msg_arb_faltas else "",
        "_Compara con la linea de faltas de tu casa de apuestas._",
        "",
        # 1X2: el Elo ~= mercado. Soporte de decision, pero NO le ganamos al mercado aca.
        "📊 *1X2*  _(≈ mercado: soporte, sin edge propio)_",
        f"🏠 {loc}: *{pl*100:.0f}%*   🤝 *{pe*100:.0f}%*   ✈️ {vis}: *{pv*100:.0f}%*",
        f"Goles {gl:.2f}-{gv:.2f}  ·  Marcador {marc[0]}-{marc[1]}  ·  Over 2.5: {r['over_2_5_goles']*100:.0f}%",
        "",
        # Tarjetas/corners: sin senal predecible (validado). Solo informativo.
        "⚠️ *Tarjetas / Corners — sin senal, NO apostar*",
        f"Tarjetas ~{r.get('tarjetas_esp', 0):.1f}   ·   Corners ~{r.get('corners_esp', 0):.0f}   _(informativo)_",
    ]

    if cuotas:
        msg += ["", "💰 *vs Mercado (1X2 + O/U goles) — referencia*"]
        for nombre, prob, mkt in [(loc, pl, cuotas.get("Home")), ("Empate", pe, cuotas.get("Draw")),
                                  (vis, pv, cuotas.get("Away")),
                                  ("Over 2.5", r["over_2_5_goles"], cuotas.get("Over"))]:
            if not mkt:
                continue
            ev, _ = calcular_ev(prob, mkt["mejor"])
            msg.append(f"{nombre}: @ {mkt['mejor']:.2f}  ->  EV {ev:+.0f}%")
        msg += ["_El 1X2 ≈ mercado: EV cerca de 0 es lo esperado, no busques ganarle aca._"]

    return enviar_mensaje("\n".join(msg))


def enviar_reporte_props(info: dict, props: dict, top: list) -> bool:
    """Alerta de Player Props: tiros esperados por jugador, con ranking y aviso de calibracion.

    Se envia ~70-80 min antes del KO, cuando el XI ya esta confirmado.
    Solo incluye jugadores con historial real (fuente != shadow) y lam >= 1.0.
    """
    loc, vis = info.get("local", "?"), info.get("visitante", "?")
    meta = props.get("meta", {})
    lam_l = meta.get("lam_goles_l", 0)
    lam_v = meta.get("lam_goles_v", 0)

    msg = [
        f"🎯 *PLAYER PROPS — {loc} vs {vis}*",
        f"_XI confirmados. Tiros totales esp.: {loc} ~{lam_l/meta.get('conv_rate',0.091):.0f}"
        f"  |  {vis} ~{lam_v/meta.get('conv_rate',0.091):.0f}_",
        "",
        "*Mas de 1.5 tiros — ranking por λ esperado:*",
    ]

    for j in top:
        bar = "█" * int(j["lam"] + 0.5)
        fuente_icon = "🌍" if j["fuente"] == "intl" else "🏟️"
        msg.append(
            f"{fuente_icon} *{j['nombre']}* ({j['nacion']}) "
            f"λ={j['lam']:.1f}  P(>1.5)={j['p_over_1_5']*100:.0f}%  "
            f"P(SOT≥1)={j['p_sot_1']*100:.0f}%"
        )

    msg += [
        "",
        "🌍=historial intl  🏟️=datos club escalados",
        "⚠️ _Prob. ORIENTATIVAS: el modelo sobreestima en valores altos (~20pp)._",
        "_Usar λ como ranking de quien disparara mas, no como probabilidad exacta._",
        "_Comparar siempre con la linea real de tu casa antes de apostar._",
    ]

    return enviar_mensaje("\n".join(m for m in msg if m is not None))


def listar_chats() -> None:
    """Muestra los chat_ids de quienes le escribieron al bot (getUpdates) para agregarlos a
    suscriptores.txt. Cada persona primero tiene que mandarle /start a @D_SoccerBot."""
    config.cargar_env()
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("Falta TELEGRAM_TOKEN en el .env.")
        return
    try:
        data = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    except requests.RequestException as e:
        print(f"fallo de red: {e}")
        return
    vistos = {}
    for upd in data.get("result", []):
        chat = (upd.get("message") or upd.get("channel_post") or {}).get("chat", {})
        if chat.get("id"):
            vistos[chat["id"]] = (chat.get("first_name") or chat.get("title")
                                  or chat.get("username") or "")
    if not vistos:
        print("Nadie le escribio al bot ultimamente (o ya expiraron los updates de Telegram).")
        print("Pedile a la persona que le mande /start a @D_SoccerBot y volve a correr esto.")
        return
    print("Chats que le escribieron al bot (copialos a suscriptores.txt, uno por linea):\n")
    for cid, nom in vistos.items():
        print(f"  {cid}  # {nom}")
    print(f"\nDestinatarios actuales configurados: {len(_recipientes())}")


def main() -> None:
    if "--quien" in sys.argv:
        listar_chats()
        return
    # Sin args: ping de prueba a TODOS los destinatarios.
    ok = enviar_mensaje("✅ *D-Soccer* conectado. Listo para avisarte del Mundial.")
    print("Ping enviado." if ok else "No se pudo enviar (revisa .env / suscriptores.txt).")


if __name__ == "__main__":
    main()
