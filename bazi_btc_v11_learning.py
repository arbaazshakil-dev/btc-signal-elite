"""BAZI BTC V11 Learning — contract-aligned 15-minute forecasting.

Run with: streamlit run bazi_btc_v9_learning.py
This is a research/decision-support tool, not an order execution system.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

APP_VERSION = "BAZI BTC V11 Learning"
DB_PATH = Path(__file__).with_name("bazi_learning.sqlite3")
CB = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
HEADERS = {"User-Agent": "BAZI-BTC-V11/1.0", "Accept": "application/json"}
FEATURES = ("gap_vol", "mom_3", "mom_10", "book", "kraken_agree", "time_frac")
SCALES = {"gap_vol": 3.0, "mom_3": 1.5, "mom_10": 1.5, "book": 1.0,
          "kraken_agree": 1.0, "time_frac": 1.0}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def next_expiry(ts: Optional[float] = None) -> int:
    """Next UTC quarter-hour boundary; exact boundaries advance 15 minutes."""
    now = int(ts or time.time())
    return ((now // 900) + 1) * 900


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
      id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, expiry_at INTEGER NOT NULL,
      target REAL NOT NULL, contract_now REAL NOT NULL, coinbase_now REAL NOT NULL,
      kraken_now REAL, predicted_side TEXT NOT NULL, probability REAL NOT NULL,
      decision TEXT NOT NULL, forecast_price REAL NOT NULL, features_json TEXT NOT NULL,
      resolved_at INTEGER, settlement_price REAL, outcome_side TEXT, correct INTEGER,
      resolution_source TEXT, learnable INTEGER DEFAULT 0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS model (
      id INTEGER PRIMARY KEY CHECK(id=1), weights_json TEXT NOT NULL,
      examples INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL)""")
    con.execute("INSERT OR IGNORE INTO model VALUES (1, ?, 0, ?)",
                (json.dumps({"bias": 0.0, **{f: 0.0 for f in FEATURES}}), int(time.time())))
    con.commit()
    return con


def get_json(url: str, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=8)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=8, show_spinner=False)
def market_snapshot():
    ticker = get_json(f"{CB}/products/BTC-USD/ticker")
    candles = get_json(f"{CB}/products/BTC-USD/candles", {"granularity": 60})
    book = get_json(f"{CB}/products/BTC-USD/book", {"level": 2})
    rows = sorted(candles, key=lambda x: x[0])[-31:]
    closes = [float(x[4]) for x in rows]
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    cb_price = float(ticker["price"])
    bid_qty = sum(float(x[1]) for x in book.get("bids", [])[:25])
    ask_qty = sum(float(x[1]) for x in book.get("asks", [])[:25])
    imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1e-9)
    try:
        kr = get_json(f"{KRAKEN}/Ticker", {"pair": "XBTUSD"})
        kraken_price = float(next(iter(kr["result"].values()))["c"][0])
    except Exception:
        kraken_price = None
    return {"cb": cb_price, "kraken": kraken_price, "closes": closes,
            "returns": returns, "book": max(-1.0, min(1.0, imbalance))}


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


# ---------------------------------------------------------------------------
# Digital/HUD-style rendering helpers (dark, monospace, neon-accented)
# ---------------------------------------------------------------------------

def render_digital_metric(label: str, value: str, color: str = "#00ff9d", sub: str = ""):
    sub_html = f'<div style="color:#7d8590; font-size:12px; font-family:monospace; margin-top:4px;">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid #21262d; border-radius:12px;
                padding:20px; margin-bottom:16px;">
        <div style="color:#7d8590; font-size:12px; text-transform:uppercase;
                    letter-spacing:1px; font-family:monospace;">{label}</div>
        <div style="color:{color}; font-size:44px; font-weight:700; font-family:monospace;
                    text-shadow:0 0 20px {color}66; line-height:1.1;">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def _format_cell(col: str, val):
    """Return (display_text, inline_style) for a table cell based on column/value."""
    if val is None:
        return "—", "color:#4b5563;"
    text = str(val)
    upper = text.upper()

    if col.lower() in ("predicted_side", "pick", "outcome", "outcome_side"):
        if upper == "UP":
            return "▲ UP", "color:#00ff9d; text-shadow:0 0 8px #00ff9d99; font-weight:600;"
        if upper == "DOWN":
            return "▼ DOWN", "color:#ff3b5c; text-shadow:0 0 8px #ff3b5c99; font-weight:600;"

    if col.lower() == "decision":
        if upper == "CONFIDENT":
            return "✅ CONFIDENT", "background:#00331f; color:#00ff9d; padding:3px 10px; border-radius:6px; font-size:11px; letter-spacing:0.5px;"
        if upper == "LEAN":
            return "🟡 LEAN", "background:#3a2f00; color:#ffcc00; padding:3px 10px; border-radius:6px; font-size:11px; letter-spacing:0.5px;"
        if upper == "WAIT":
            return "⏸ WAIT", "background:#3a3f4b; color:#9aa0ab; padding:3px 10px; border-radius:6px; font-size:11px; letter-spacing:0.5px;"

    if col.lower() == "correct":
        if val is True or upper in ("✓", "TRUE", "1"):
            return "✓", "color:#00ff9d; font-weight:700; text-align:center;"
        if val is False or upper in ("✗", "FALSE", "0"):
            return "✗", "color:#ff3b5c; font-weight:700; text-align:center;"
        return "· PENDING", "color:#7d8590;"

    if col.lower() in ("created_at",):
        try:
            dt = text
            if "T" in dt:
                dt = dt.split(".")[0].replace("T", " ")
            return dt[-8:] if len(dt) >= 8 else dt, "color:#7d8590; font-family:monospace;"
        except Exception:
            return text, "color:#7d8590;"

    return text, "font-family:monospace;"


def render_digital_table(rows: list[dict], columns: list[str], labels: Optional[dict] = None):
    """Render a list of dicts as a dark, neon-accented HTML table."""
    if not rows:
        st.info("No rows to display yet.")
        return
    labels = labels or {}
    header_html = "".join(f"<th>{labels.get(c, c)}</th>" for c in columns)
    body_html = ""
    for row in rows:
        cells = ""
        for c in columns:
            text, style = _format_cell(c, row.get(c))
            cells += f'<td style="{style}">{text}</td>'
        body_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <style>
        .digital-table-wrap {{ overflow-x:auto; border-radius:12px; border:1px solid #21262d; }}
        .digital-table {{ width:100%; border-collapse:collapse; background:#0d1117; }}
        .digital-table th {{ background:#161b22; color:#7d8590; text-align:left;
            padding:10px 14px; font-size:11px; text-transform:uppercase;
            letter-spacing:1px; font-family:monospace; border-bottom:1px solid #21262d;
            white-space:nowrap; }}
        .digital-table td {{ padding:9px 14px; font-size:13px;
            border-bottom:1px solid #161b22; white-space:nowrap; }}
        .digital-table tr:hover td {{ background:#161b22; }}
    </style>
    <div class="digital-table-wrap">
        <table class="digital-table"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>
    </div>
    """, unsafe_allow_html=True)


def load_model(con):
    row = con.execute("SELECT * FROM model WHERE id=1").fetchone()
    return json.loads(row["weights_json"]), int(row["examples"])


@dataclass
class Forecast:
    side: str
    probability: float
    decision: str
    forecast_price: float
    baseline_probability: float
    learned_probability: Optional[float]
    seconds_left: int
    gap: float
    sigma_expiry: float
    features: dict
    reasons: list
    warnings: list


def forecast(target: float, contract_now: float, snap: dict, expiry: int, con) -> Forecast:
    seconds = max(1, expiry - int(time.time()))
    closes, rets = snap["closes"], snap["returns"]
    vol_1m = statistics.stdev(rets[-20:]) if len(rets) >= 3 else 0.0006
    vol_1m = max(0.00015, min(0.004, vol_1m))
    m3 = math.log(closes[-1] / closes[-4]) if len(closes) >= 4 else 0.0
    m10 = math.log(closes[-1] / closes[-11]) if len(closes) >= 11 else m3
    # Coinbase predicts only the move. The move is applied to contract NOW, never
    # Coinbase's absolute level, preventing a cross-feed basis from flipping the side.
    momentum = 0.55 * m3 + 0.25 * m10 + 0.20 * snap["book"] * vol_1m
    damp = min(1.0, seconds / 180.0)
    expected_return = max(-0.004, min(0.004, momentum * (seconds / 60.0) * 0.35 * damp))
    predicted = contract_now * math.exp(expected_return)
    sigma_price = max(contract_now * vol_1m * math.sqrt(seconds / 60.0), contract_now * 0.00008)
    z = (predicted - target) / sigma_price
    p_above_base = max(0.05, min(0.95, normal_cdf(z)))
    kr = snap["kraken"]
    cb_move = m3
    kr_agree = 0.0 if kr is None else (1.0 if cb_move * (kr - snap["cb"]) >= 0 else -1.0)
    feats = {"gap_vol": max(-5.0, min(5.0, (contract_now-target)/sigma_price)),
             "mom_3": max(-3.0, min(3.0, m3/max(vol_1m, 1e-9))),
             "mom_10": max(-3.0, min(3.0, m10/max(vol_1m*math.sqrt(10), 1e-9))),
             "book": snap["book"], "kraken_agree": kr_agree,
             "time_frac": seconds/900.0}
    weights, n = load_model(con)
    p_learned = None
    p_above = p_above_base
    if n >= 30:
        score = weights.get("bias", 0.0) + sum(weights.get(f, 0.0)*feats[f]/SCALES[f] for f in FEATURES)
        p_learned = sigmoid(score)
        blend = min(0.35, 0.10 + (n-30)/500.0)  # learning may calibrate, never dominate
        p_above = (1-blend)*p_above_base + blend*p_learned
    p_above = max(0.08, min(0.92, p_above))  # never advertise near-certainty
    side = "ABOVE" if p_above >= 0.5 else "BELOW"
    confidence = p_above if side == "ABOVE" else 1-p_above
    warnings, reasons = [], []
    basis = contract_now - snap["cb"]
    if abs(basis) > max(15.0, 1.25*sigma_price):
        warnings.append("Large contract/Coinbase basis: Coinbase is used for movement only.")
    if kr is None:
        warnings.append("Kraken unavailable; secondary confirmation missing.")
    elif abs(kr-snap["cb"]) > max(20.0, 1.5*sigma_price):
        warnings.append("Coinbase and Kraken materially disagree.")
    margin = abs(predicted-target)
    if confidence >= 0.78 and margin >= 1.15*sigma_price and not warnings:
        decision = "CONFIDENT"
    elif confidence >= 0.62 and margin >= 0.45*sigma_price:
        decision = "LEAN"
    else:
        decision = "WAIT"
    if seconds > 900 or seconds <= 5:
        decision = "WAIT"
        warnings.append("Outside the usable 15-minute forecasting window.")
    reasons.append(f"Contract reference is ${contract_now-target:+,.2f} versus Price to Beat.")
    reasons.append(f"Coinbase-derived move forecast is ${predicted-contract_now:+,.2f} from contract NOW.")
    reasons.append(f"Expiry uncertainty is about ±${sigma_price:,.2f} (1σ).")
    return Forecast(side, confidence, decision, predicted, p_above_base, p_learned,
                    seconds, contract_now-target, sigma_price, feats, reasons, warnings)


def save_prediction(con, fc, target, contract_now, snap, expiry):
    pred_id = uuid.uuid4().hex
    con.execute("""INSERT INTO predictions
      (id,created_at,expiry_at,target,contract_now,coinbase_now,kraken_now,predicted_side,
       probability,decision,forecast_price,features_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
      (pred_id, int(time.time()), expiry, target, contract_now, snap["cb"], snap["kraken"],
       fc.side, fc.probability, fc.decision, fc.forecast_price, json.dumps(fc.features)))
    con.commit()
    return pred_id


def train_one(con, row, outcome_above: int):
    weights, n = load_model(con)
    x = json.loads(row["features_json"])
    score = weights.get("bias", 0.0) + sum(weights.get(f, 0.0)*x[f]/SCALES[f] for f in FEATURES)
    error = outcome_above - sigmoid(score)
    lr = max(0.015, 0.08/math.sqrt(1+n/25))
    l2 = 0.002
    weights["bias"] = max(-3.0, min(3.0, weights.get("bias", 0.0) + lr*error))
    for f in FEATURES:
        value = x[f]/SCALES[f]
        weights[f] = max(-3.0, min(3.0, weights.get(f, 0.0)*(1-lr*l2) + lr*error*value))
    con.execute("UPDATE model SET weights_json=?, examples=?, updated_at=? WHERE id=1",
                (json.dumps(weights), n+1, int(time.time())))


def resolve_prediction(con, pred_id: str, settlement: float, source="MANUAL_CONTRACT"):
    row = con.execute("SELECT * FROM predictions WHERE id=? AND resolved_at IS NULL", (pred_id,)).fetchone()
    if not row:
        return False
    outcome = "ABOVE" if settlement > row["target"] else "BELOW"  # tie is BELOW unless contract says otherwise
    correct = int(outcome == row["predicted_side"])
    learnable = int(source == "MANUAL_CONTRACT")
    con.execute("""UPDATE predictions SET resolved_at=?, settlement_price=?, outcome_side=?,
      correct=?, resolution_source=?, learnable=? WHERE id=?""",
      (int(time.time()), settlement, outcome, correct, source, learnable, pred_id))
    if learnable:
        train_one(con, row, int(outcome == "ABOVE"))
    con.commit()
    return True


def auto_resolve_proxy(con):
    """Resolve old records for reporting only, using basis-adjusted Coinbase candles.

    Proxy labels never train the model; only entered contract settlement values do.
    """
    rows = con.execute("SELECT * FROM predictions WHERE resolved_at IS NULL AND expiry_at < ?", (int(time.time())-75,)).fetchall()
    for row in rows:
        try:
            start, end = row["expiry_at"]-60, row["expiry_at"]+60
            candles = get_json(f"{CB}/products/BTC-USD/candles",
                               {"granularity": 60, "start": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                                "end": datetime.fromtimestamp(end, timezone.utc).isoformat()})
            close = min(candles, key=lambda x: abs(int(x[0])-row["expiry_at"]))[4]
            basis = row["contract_now"] - row["coinbase_now"]
            resolve_prediction(con, row["id"], float(close)+basis, "COINBASE_BASIS_PROXY")
        except Exception:
            pass


def history_stats(con):
    rows = con.execute("SELECT * FROM predictions WHERE resolved_at IS NOT NULL ORDER BY created_at DESC").fetchall()
    manual = [r for r in rows if r["learnable"]]
    accuracy = sum(r["correct"] for r in manual)/len(manual) if manual else None
    return rows, manual, accuracy


def cloud_history():
    """Read unattended V11 results when Supabase secrets are configured."""
    try:
        url = st.secrets["SUPABASE_URL"].rstrip("/")
        key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    except Exception:
        return None
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    result = requests.get(
        f"{url}/rest/v1/bazi_predictions",
        params={"select":"market_ticker,created_at,decision,predicted_side,probability,outcome_side,correct,resolution_source",
                "order":"created_at.desc", "limit":"100"}, headers=headers, timeout=10)
    result.raise_for_status()
    model = requests.get(f"{url}/rest/v1/bazi_model",
                         params={"id":"eq.1", "select":"examples,updated_at"},
                         headers=headers, timeout=10)
    model.raise_for_status()
    return result.json(), (model.json()[0] if model.json() else {"examples":0})


def checkpoint_history():
    """Read V12.1 13m/8m/6m/4m/2m checkpoint results from Supabase."""
    try:
        url = st.secrets["SUPABASE_URL"].rstrip("/")
        key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    except Exception:
        return None
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows = requests.get(
        f"{url}/rest/v1/bazi_checkpoints",
        params={
            "select": "market_ticker,checkpoint,seconds_remaining,created_at,target,coinbase_now,predicted_side,probability,decision,outcome_side,correct,resolved_at",
            "order": "created_at.desc",
            "limit": "200",
        },
        headers=headers, timeout=10)
    rows.raise_for_status()
    summary = requests.get(
        f"{url}/rest/v1/bazi_checkpoint_accuracy",
        params={"select": "checkpoint,resolved_predictions,correct_predictions,accuracy_pct"},
        headers=headers, timeout=10)
    summary.raise_for_status()
    return rows.json(), summary.json()


def main():
    st.set_page_config(page_title=APP_VERSION, page_icon="₿", layout="wide")
    st.title(APP_VERSION)
    st.caption("Contract-aligned research for the next 15-minute expiry. No automatic trading.")
    con = db()
    auto_resolve_proxy(con)
    expiry = next_expiry()
    left = expiry-int(time.time())
    with st.sidebar:
        st.header("Contract inputs")
        target = st.number_input("Price to Beat ($)", min_value=1.0, value=65000.0, step=0.01, format="%.2f")
        contract_now = st.number_input("Contract NOW / reference ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        st.caption("Required. Copy the live reference shown by the contract; do not substitute Coinbase.")
        st.metric("Auto expiry (UTC)", datetime.fromtimestamp(expiry, timezone.utc).strftime("%H:%M:%S"))
        st.metric("Time remaining", f"{left//60:02d}:{left%60:02d}")
        auto_refresh = st.toggle("Auto-refresh market data", value=True)
        refresh_seconds = st.select_slider(
            "Refresh every", options=[10, 15, 30, 60], value=10,
            format_func=lambda value: f"{value} seconds", disabled=not auto_refresh)
        if auto_refresh:
            st_autorefresh(interval=refresh_seconds * 1000, key="bazi_market_refresh")
        refresh = st.button("Refresh market data", use_container_width=True)
    if refresh:
        market_snapshot.clear()
    try:
        snap = market_snapshot()
    except Exception as exc:
        st.error(f"Market feeds unavailable: {exc}")
        st.stop()
    a,b,c = st.columns(3)
    a.metric("Coinbase BTC (primary features)", f"${snap['cb']:,.2f}")
    b.metric("Kraken BTC (confirmation)", "Unavailable" if snap["kraken"] is None else f"${snap['kraken']:,.2f}")
    c.metric("Contract position NOW", "Enter reference" if not contract_now else f"${contract_now-target:+,.2f}")
    st.caption(f"Market calculation updated {utc_now().strftime('%H:%M:%S UTC')}"
               + (f" · automatic refresh every {refresh_seconds}s" if auto_refresh else " · automatic refresh off"))
    if contract_now <= 0:
        st.warning("WAIT — enter the contract's live NOW/reference price. BAZI will not infer contract position from Coinbase.")
    else:
        fc = forecast(target, contract_now, snap, expiry, con)
        color = {"WAIT":"⚪", "LEAN":"🟡", "CONFIDENT":"🟢"}[fc.decision]
        st.subheader(f"{color} {fc.decision}: {fc.side} — {fc.probability:.1%}")
        x,y,z = st.columns(3)
        x.metric("Forecast at expiry", f"${fc.forecast_price:,.2f}")
        y.metric("Price to Beat", f"${target:,.2f}")
        z.metric("Forecast distance", f"${fc.forecast_price-target:+,.2f}")
        for warning in fc.warnings:
            st.warning(warning)
        with st.expander("Why BAZI says this", expanded=True):
            for reason in fc.reasons:
                st.write("• " + reason)
            st.caption("Learning calibrates at 30 verified outcomes, is capped at 35% influence, and confidence is capped at 92%.")
        if st.button("Log this prediction", type="primary"):
            pid = save_prediction(con, fc, target, contract_now, snap, expiry)
            st.success(f"Saved prediction {pid[:8]}. Resolve it after contract settlement so BAZI can learn.")
    st.divider()
    st.header("Autonomous learning V11")
    try:
        cloud = cloud_history()
    except Exception as exc:
        cloud = None
        st.warning(f"Cloud learning database is configured but unavailable: {exc}")
    if cloud:
        cloud_rows, cloud_model = cloud
        resolved = [x for x in cloud_rows if x.get("outcome_side")]
        correct = [x for x in resolved if x.get("correct")]
        ca, cb, cc = st.columns(3)
        ca.metric("Automatic predictions", len(cloud_rows))
        cb.metric("Official outcomes learned", int(cloud_model.get("examples", 0)))
        acc_val = "—" if not resolved else f"{len(correct)/len(resolved):.1%}"
        render_digital_metric("Autonomous Accuracy", acc_val,
                               sub=f"{len(correct)}/{len(resolved)} resolved correct" if resolved else "awaiting resolved outcomes")
        if cloud_rows:
            display_cols = ["created_at", "decision", "predicted_side", "correct"]
            col_labels = {"created_at": "Time", "decision": "Decision",
                          "predicted_side": "Signal", "correct": "Correct"}
            render_digital_table(cloud_rows, display_cols, col_labels)
    else:
        st.info("Autonomous cloud history will appear here after Supabase secrets and the scheduled worker are connected.")

    st.divider()
    st.header("V12.1 checkpoint predictions")
    st.caption("Shadow checkpoint tracker: independent predictions near 13, 8, 6, 4, and 2 minutes remaining. These results do not change the main V12 model weights.")
    try:
        checkpoint_cloud = checkpoint_history()
    except Exception as exc:
        checkpoint_cloud = None
        st.warning(f"Checkpoint database is configured but unavailable: {exc}")
    if checkpoint_cloud:
        checkpoint_rows, checkpoint_summary = checkpoint_cloud
        summary_map = {x.get("checkpoint"): x for x in checkpoint_summary}
        cols = st.columns(5)
        for i, label in enumerate(("13m", "8m", "6m", "4m", "2m")):
            item = summary_map.get(label, {})
            resolved_n = int(item.get("resolved_predictions") or 0)
            accuracy = item.get("accuracy_pct")
            display = "—" if accuracy is None else f"{float(accuracy):.1f}%"
            cols[i].metric(f"{label} accuracy", display, help=f"{resolved_n} officially resolved checkpoint predictions")
            cols[i].caption(f"{resolved_n} resolved")
        unresolved = [x for x in checkpoint_rows if not x.get("outcome_side")]
        if unresolved:
            latest_by_checkpoint = {}
            for row in unresolved:
                cp = row.get("checkpoint")
                if cp and cp not in latest_by_checkpoint:
                    latest_by_checkpoint[cp] = row
            latest = [latest_by_checkpoint[x] for x in ("13m","8m","6m","4m","2m") if x in latest_by_checkpoint]
            if latest:
                st.subheader("Current checkpoint signals")
                live_table=[]
                for x in latest:
                    live_table.append({
                        "Checkpoint": x.get("checkpoint"),
                        "Seconds left": x.get("seconds_remaining"),
                        "Pick": x.get("predicted_side"),
                        "Confidence": "—" if x.get("probability") is None else f"{float(x['probability']):.1%}",
                        "Decision": x.get("decision"),
                        "BTC now": "—" if x.get("coinbase_now") is None else f"${float(x['coinbase_now']):,.2f}",
                        "Price to Beat": "—" if x.get("target") is None else f"${float(x['target']):,.2f}",
                        "Contract": x.get("market_ticker"),
                    })
                render_digital_table(live_table,
                                      ["Checkpoint", "Seconds left", "Pick", "Confidence", "Decision", "BTC now", "Price to Beat", "Contract"])
        if checkpoint_rows:
            st.subheader("Checkpoint history")
            history=[]
            for x in checkpoint_rows[:100]:
                history.append({
                    "Checkpoint": x.get("checkpoint"),
                    "Pick": x.get("predicted_side"),
                    "Confidence": "—" if x.get("probability") is None else f"{float(x['probability']):.1%}",
                    "Outcome": x.get("outcome_side") or "Pending",
                    "Correct": "Pending" if x.get("correct") is None else ("✓" if x.get("correct") else "✗"),
                    "Seconds left": x.get("seconds_remaining"),
                    "Contract": x.get("market_ticker"),
                })
            render_digital_table(history,
                                  ["Checkpoint", "Pick", "Confidence", "Outcome", "Correct", "Seconds left", "Contract"])
        else:
            st.info("No checkpoint predictions have been recorded yet.")
    else:
        st.info("Checkpoint results will appear here after the V12.1 tracker records its first 13m/8m/6m/4m/2m prediction.")

    st.subheader("Manual outcomes")
    pending = con.execute("SELECT * FROM predictions WHERE resolved_at IS NULL ORDER BY expiry_at").fetchall()
    expired = [r for r in pending if r["expiry_at"] <= int(time.time())]
    if expired:
        labels = {f"{r['id'][:8]} · {datetime.fromtimestamp(r['expiry_at']).strftime('%m/%d %H:%M')} · target ${r['target']:,.2f}": r for r in expired}
        selected = st.selectbox("Expired prediction", list(labels))
        settlement = st.number_input("Official contract settlement/reference value ($)", min_value=1.0, step=0.01, format="%.2f")
        if st.button("Resolve and learn"):
            resolve_prediction(con, labels[selected]["id"], settlement)
            st.success("Outcome saved and the online calibrator updated.")
            st.rerun()
    else:
        st.info("No expired unresolved predictions. Automatic proxy outcomes are tracked but never used for training.")
    rows, manual, accuracy = history_stats(con)
    weights, learned_n = load_model(con)
    p,q,r = st.columns(3)
    p.metric("Verified training outcomes", learned_n)
    q.metric("Verified accuracy", "—" if accuracy is None else f"{accuracy:.1%}")
    r.metric("Learning active", "Yes" if learned_n >= 30 else f"After {30-learned_n} more")
    if rows:
        table = [{"Created": datetime.fromtimestamp(x["created_at"]).strftime("%Y-%m-%d %H:%M"),
                  "Decision": x["decision"], "Pick": x["predicted_side"],
                  "Confidence": f"{x['probability']:.1%}", "Outcome": x["outcome_side"],
                  "Correct": "✓" if x["correct"] else "✗", "Source": x["resolution_source"]}
                 for x in rows[:100]]
        st.dataframe(table, use_container_width=True, hide_index=True)
    with st.expander("Safety and methodology"):
        st.write("Coinbase supplies price movement, candles, volatility and order-book features. The contract NOW value anchors the target gap. Kraken is confirmation only. Large feed disagreement blocks CONFIDENT status. Only manually verified contract settlements train the model; Coinbase proxy resolutions are reporting-only. WAIT is a valid result.")
        st.write("This model is experimental and cannot guarantee outcomes. Prediction-market trading can lose money.")


if __name__ == "__main__":
    main()
