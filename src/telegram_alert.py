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


def enviar_mensaje(texto: str, chats: list[str] | None = None) -> bool:
    """Envia un mensaje a TODOS los destinatarios (o solo a 'chats' si se pasa, util para
    tests dirigidos). True si llego al menos a uno."""
    config.cargar_env()
    token = os.environ.get("TELEGRAM_TOKEN")
    if chats is None:
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


def enviar_reporte_props(info: dict, props: dict, top: list,
                         solo_chat: str | None = None) -> bool:
    """Alerta de Player Props: ~70-90min antes del KO, con XI confirmado.

    Muestra top 4 disparadores por equipo, separados. Para cada jugador:
    λ (tiros esperados), xG (calidad de esas posiciones), P(>1.5 tiros), P(SOT>=1).
    solo_chat: si se pasa, envia SOLO a ese chat_id (modo test, no a los suscriptores).
    """
    loc, vis = info.get("local", "?"), info.get("visitante", "?")
    meta = props.get("meta", {})
    lam_l = meta.get("lam_goles_l", 0)
    lam_v = meta.get("lam_goles_v", 0)
    conv_rate = meta.get("conv_rate", 0.121)  # xG-based (mas preciso que goles reales)

    # Minutos hasta el KO (si la fecha esta disponible)
    ko_str = info.get("fecha", "")
    min_str = ""
    if ko_str:
        try:
            ko = datetime.fromisoformat(ko_str)
            mins = (ko - datetime.now(timezone.utc)).total_seconds() / 60
            if mins > 0:
                min_str = f" · ~{mins:.0f}min"
        except Exception:
            pass

    tiros_l = round(lam_l / conv_rate) if conv_rate > 0 else "?"
    tiros_v = round(lam_v / conv_rate) if conv_rate > 0 else "?"

    msg = [
        f"🎯 *PLAYER PROPS — {loc} vs {vis}*",
        f"_XI confirmados{min_str}_",
        f"_{loc} ≈{tiros_l} tiros  ·  {vis} ≈{tiros_v} tiros esperados_",
        "",
    ]

    # Top 4 por equipo, calculado directamente desde props (independiente del top global)
    MIN_LAM = 0.8
    TOP_N = 4

    def _top_equipo(lado: str) -> list[dict]:
        datos = props.get(lado, {})
        nac = datos.get("nacion", "")
        out = [
            {"nombre": n, "nacion": nac, "lado": lado, **v}
            for n, v in datos.get("jugadores", {}).items()
            if v.get("fuente") != "shadow" and v.get("lam", 0) >= MIN_LAM
        ]
        return sorted(out, key=lambda x: x.get("p_over_1_5", 0), reverse=True)[:TOP_N]

    for lado, titulo in [("local", loc), ("visitante", vis)]:
        jugadores = _top_equipo(lado)
        if not jugadores:
            continue
        msg.append(f"*{titulo} — top disparadores:*")
        for j in jugadores:
            fuente_icon = "🌍" if j["fuente"] == "intl" else "🏟️"
            xg = j.get("xG_base", 0.0)
            msg.append(
                f"{fuente_icon} *{j['nombre']}*"
                f"  λ={j['lam']:.1f}  xG={xg:.2f}"
                f"  P(>1.5)={j['p_over_1_5']*100:.0f}%"
                f"  P(SOT≥1)={j['p_sot_1']*100:.0f}%"
            )
        msg.append("")

    msg += [
        "_λ = tiros esperados  ·  xG = goles de esas posiciones_",
        "_Probs ORIENTATIVAS: sobreestima ~20pp en valores altos_",
        "_Usar el ranking de λ/xG, no la prob. como valor exacto_",
        "_Compara la linea de tu casa antes de apostar_",
        "🌍 historial intl  ·  🏟️ datos club escalados",
    ]

    return enviar_mensaje("\n".join(msg), chats=[solo_chat] if solo_chat else None)


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
