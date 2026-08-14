
import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

KRAKEN = "https://api.kraken.com/0/public"
COINGLASS = "https://open-api-v4.coinglass.com"

FEATURES = [
    "r1","r3","r5","r8",
    "ema_gap","rsi","vol_z",
    "range_pct","body_pct",
    "atr_pct","realized_vol"
]

st.set_page_config(
    page_title="BTC Signal Elite",
    page_icon="₿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {padding:1rem .8rem 2rem; max-width:780px}
h1 {font-size:1.8rem}
.signal {border-radius:22px;padding:22px;text-align:center;margin:8px 0 14px}
.up {background:#0d2b1b;border:2px solid #35c46a}
.down {background:#321414;border:2px solid #ef6262}
.wait {background:#2d2610;border:2px solid #d4a72c}
.call {font-size:2.15rem;font-weight:800}
.sub {font-size:1.05rem;font-weight:700;margin-top:7px}
.card {padding:14px;border-radius:16px;background:#17243b;margin:8px 0}
.small {opacity:.78;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HTTP
# -----------------------------
def get_json(url, params=None, headers=None, timeout=15):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def kraken_api(endpoint, params=None):
    data = get_json(f"{KRAKEN}/{endpoint}", params=params)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data["result"]

def coinglass_api(path, params, api_key):
    data = get_json(
        f"{COINGLASS}{path}",
        params=params,
        headers={"CG-API-KEY": api_key}
    )
    if str(data.get("code")) != "0":
        raise RuntimeError(data.get("msg", "CoinGlass request failed"))
    return data.get("data")

# -----------------------------
# KRAKEN SPOT DATA
# -----------------------------
@st.cache_data(ttl=10)
def get_ohlc(pair="XBTUSD"):
    result = kraken_api("OHLC", {"pair": pair, "interval": 1})
    key = [k for k in result if k != "last"][0]
    rows = result[key]

    df = pd.DataFrame(rows, columns=[
        "time","open","high","low","close","vwap","volume","count"
    ])

    for c in ["open","high","low","close","vwap","volume","count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna().reset_index(drop=True)

@st.cache_data(ttl=5)
def get_ticker(pair="XBTUSD"):
    result = kraken_api("Ticker", {"pair": pair})
    key = list(result.keys())[0]
    return float(result[key]["c"][0])

@st.cache_data(ttl=5)
def get_book(pair="XBTUSD"):
    result = kraken_api("Depth", {"pair": pair, "count": 100})
    key = list(result.keys())[0]
    book = result[key]

    bid_notional = sum(float(x[0]) * float(x[1]) for x in book["bids"])
    ask_notional = sum(float(x[0]) * float(x[1]) for x in book["asks"])
    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total else 0.0

    return imbalance, bid_notional, ask_notional

# -----------------------------
# COINGLASS DERIVATIVES DATA
# -----------------------------
@st.cache_data(ttl=15)
def get_derivatives_snapshot(api_key):
    if not api_key:
        return None

    out = {}

    # Open interest across exchanges
    oi = coinglass_api(
        "/api/futures/open-interest/exchange-list",
        {"symbol": "BTC"},
        api_key
    )
    all_oi = next((x for x in oi if x.get("exchange") == "All"), oi[0] if oi else {})
    out["oi_usd"] = float(all_oi.get("open_interest_usd", 0) or 0)
    out["oi_5m"] = float(all_oi.get("open_interest_change_percent_5m", 0) or 0)
    out["oi_15m"] = float(all_oi.get("open_interest_change_percent_15m", 0) or 0)

    # Liquidations
    liq = coinglass_api(
        "/api/futures/liquidation/exchange-list",
        {"symbol": "BTC", "range": "1h"},
        api_key
    )
    all_liq = next((x for x in liq if x.get("exchange") == "All"), liq[0] if liq else {})
    out["long_liq"] = float(all_liq.get("long_liquidation_usd", 0) or 0)
    out["short_liq"] = float(all_liq.get("short_liquidation_usd", 0) or 0)

    # Aggregated taker buy/sell
    taker = coinglass_api(
        "/api/futures/aggregated-taker-buy-sell-volume/history",
        {
            "exchange_list": "Binance,OKX,Bybit",
            "symbol": "BTC",
            "interval": "5m",
            "limit": 6,
            "unit": "usd"
        },
        api_key
    )
    if taker:
        recent = taker[-1]
        buy = float(recent.get("aggregated_buy_volume_usd", 0) or 0)
        sell = float(recent.get("aggregated_sell_volume_usd", 0) or 0)
    else:
        buy = sell = 0.0
    out["taker_buy"] = buy
    out["taker_sell"] = sell

    # OI-weighted funding
    funding = coinglass_api(
        "/api/futures/funding-rate/oi-weight-history",
        {"symbol": "BTC", "interval": "5m", "limit": 6},
        api_key
    )
    out["funding"] = float(funding[-1].get("close", 0) or 0) if funding else 0.0

    return out

# -----------------------------
# FEATURES
# -----------------------------
def add_features(df):
    x = df.copy()

    for n in [1, 3, 5, 8]:
        x[f"r{n}"] = x["close"].pct_change(n)

    ema5 = x["close"].ewm(span=5, adjust=False).mean()
    ema20 = x["close"].ewm(span=20, adjust=False).mean()
    x["ema_gap"] = ema5 / ema20 - 1

    delta = x["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    x["rsi"] = x["rsi"].fillna(50).clip(0, 100)

    vol_mean = x["volume"].rolling(30).mean()
    vol_std = x["volume"].rolling(30).std()
    x["vol_z"] = (x["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    x["range_pct"] = (x["high"] - x["low"]) / x["close"]
    x["body_pct"] = (x["close"] - x["open"]) / x["open"]

    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / x["close"]

    x["realized_vol"] = x["r1"].rolling(20).std()

    return x.replace([np.inf, -np.inf], np.nan)

def target_and_mask(x, horizon):
    future_ret = x["close"].shift(-horizon) / x["close"] - 1
    y = (future_ret > 0).astype(int)
    mask = x[FEATURES].notna().all(axis=1) & future_ret.notna()
    return y, future_ret, mask

# -----------------------------
# MODELS
# -----------------------------
def logistic_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])

def rf_model():
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced_subsample"
    )

def train_ensemble(x, horizon):
    y, _, mask = target_and_mask(x, horizon)
    idx = np.where(mask.values)[0]

    if len(idx) < 320:
        raise RuntimeError("Not enough valid candles to train elite model.")

    train_idx = idx[:-80]
    val_idx = idx[-80:]

    logit = logistic_model()
    forest = rf_model()

    logit.fit(x.iloc[train_idx][FEATURES], y.iloc[train_idx])
    forest.fit(x.iloc[train_idx][FEATURES], y.iloc[train_idx])

    p1 = logit.predict_proba(x.iloc[val_idx][FEATURES])[:, 1]
    p2 = forest.predict_proba(x.iloc[val_idx][FEATURES])[:, 1]
    p = 0.55 * p1 + 0.45 * p2

    preds = (p >= 0.5).astype(int)
    actual = y.iloc[val_idx].values

    validation = {
        "accuracy": float(accuracy_score(actual, preds)),
        "brier": float(brier_score_loss(actual, p)),
        "n": int(len(actual))
    }

    # Refit on all available historical rows
    logit.fit(x.loc[mask, FEATURES], y.loc[mask])
    forest.fit(x.loc[mask, FEATURES], y.loc[mask])

    return logit, forest, validation

def similar_setups(x, horizon, row, k=100):
    y, _, mask = target_and_mask(x, horizon)
    hist = x.loc[mask, FEATURES].copy()
    if len(hist) < 120:
        return None

    mean = hist.mean()
    std = hist.std().replace(0, 1)
    z_hist = (hist - mean) / std
    z_now = (row.iloc[0] - mean) / std
    dist = np.sqrt(((z_hist - z_now) ** 2).mean(axis=1))
    nearest = dist.nsmallest(min(k, len(dist))).index
    outcomes = y.loc[nearest]

    return {
        "p_up": float(outcomes.mean()),
        "n": int(len(outcomes))
    }

# -----------------------------
# DERIVATIVES SCORE
# -----------------------------
def derivatives_score(d):
    if not d:
        return 0.0, []

    score = 0.0
    notes = []

    # Taker flow
    total_taker = d["taker_buy"] + d["taker_sell"]
    taker_imb = (
        (d["taker_buy"] - d["taker_sell"]) / total_taker
        if total_taker else 0.0
    )
    score += np.clip(taker_imb * 0.16, -0.16, 0.16)
    notes.append(f"Taker flow {taker_imb*100:+.1f}%")

    # Open interest: rising OI amplifies current direction, but does not define it alone.
    oi_bias = np.clip(d["oi_5m"] / 5.0, -1.0, 1.0)
    score += oi_bias * 0.035
    notes.append(f"OI 5m {d['oi_5m']:+.2f}%")

    # Liquidation asymmetry
    liq_total = d["long_liq"] + d["short_liq"]
    liq_imb = (
        (d["short_liq"] - d["long_liq"]) / liq_total
        if liq_total else 0.0
    )
    score += np.clip(liq_imb * 0.06, -0.06, 0.06)
    notes.append(f"Liquidation skew {liq_imb*100:+.1f}%")

    # Funding: crowded longs = mild bearish contrarian pressure, crowded shorts = mild bullish.
    funding = d["funding"]
    funding_adj = float(np.clip(-funding * 8.0, -0.03, 0.03))
    score += funding_adj
    notes.append(f"Funding {funding:+.5f}")

    return float(np.clip(score, -0.22, 0.22)), notes

# -----------------------------
# REGIME / RISK
# -----------------------------
def regime(x, book_imb, ensemble_gap):
    rsi = float(x["rsi"].iloc[-1])
    ema_gap = float(x["ema_gap"].iloc[-1])
    rv = float(x["realized_vol"].iloc[-1])
    r5 = float(x["r5"].iloc[-1])

    choppy = (
        abs(ema_gap) < 0.00025
        and abs(r5) < 0.0008
        and 42 <= rsi <= 58
    )

    if choppy or ensemble_gap > 0.16:
        risk = "HIGH"
    elif abs(book_imb) < 0.025 or ensemble_gap > 0.09:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "rsi": rsi,
        "ema_gap": ema_gap,
        "rv": rv,
        "r5": r5,
        "choppy": choppy,
        "risk": risk
    }

# -----------------------------
# UI
# -----------------------------
st.title("₿ BTC Short-Term Signal Elite")
st.caption("Ensemble model + validation + similar setups + optional CoinGlass derivatives confirmation")

with st.sidebar:
    pair = st.text_input("Kraken pair", "XBTUSD").upper()
    horizon = st.slider("Prediction window (minutes)", 1, 30, 8)
    min_conf = st.slider("Minimum elite confidence", 0.52, 0.80, 0.62, 0.01)
    refresh = st.slider("Refresh seconds", 15, 90, 30)

# Read CoinGlass key from Streamlit secrets if configured.
try:
    cg_key = st.secrets.get("COINGLASS_API_KEY", "")
except Exception:
    cg_key = ""

try:
    df = get_ohlc(pair)
    x = add_features(df)
    row = x[FEATURES].iloc[[-1]]

    logit, forest, validation = train_ensemble(x, horizon)

    p_logit = float(logit.predict_proba(row)[0, 1])
    p_forest = float(forest.predict_proba(row)[0, 1])
    p_spot = 0.55 * p_logit + 0.45 * p_forest

    ensemble_gap = abs(p_logit - p_forest)

    price = get_ticker(pair)
    book_imb, bid_depth, ask_depth = get_book(pair)

    # Spot order-book adjustment
    book_adj = float(np.clip(book_imb * 0.07, -0.07, 0.07))

    similar = similar_setups(x, horizon, row)
    if similar:
        hist_adj = float(np.clip((similar["p_up"] - 0.5) * 0.18, -0.07, 0.07))
    else:
        hist_adj = 0.0

    derivatives = None
    derivative_error = None
    if cg_key:
        try:
            derivatives = get_derivatives_snapshot(cg_key)
        except Exception as e:
            derivative_error = str(e)

    d_adj, d_notes = derivatives_score(derivatives)

    p_final = float(np.clip(
        p_spot + book_adj + hist_adj + d_adj,
        0.01, 0.99
    ))

    direction = "UP" if p_final >= 0.5 else "DOWN"
    confidence = max(p_final, 1 - p_final)

    reg = regime(x, book_imb, ensemble_gap)

    models_agree = (
        (p_logit >= 0.5 and p_forest >= 0.5)
        or (p_logit < 0.5 and p_forest < 0.5)
    )

    validation_ok = validation["accuracy"] >= 0.52
    derivatives_required = bool(cg_key)
    derivatives_ok = (not derivatives_required) or (derivatives is not None)

    eligible = (
        confidence >= min_conf
        and models_agree
        and validation_ok
        and not reg["choppy"]
        and reg["risk"] != "HIGH"
        and derivatives_ok
    )

    if eligible:
        emoji = "🟢" if direction == "UP" else "🔴"
        css = "up" if direction == "UP" else "down"
        status = f"{direction} — {confidence*100:.1f}%"
        grade = "ELITE CONFIRMED"
    else:
        emoji = "🟡"
        css = "wait"
        status = "WAIT — NO TRADE"
        grade = "FILTERED"

    st.markdown(f"""
    <div class="signal {css}">
      <div class="small">ELITE STATUS</div>
      <div class="call">{emoji} {status}</div>
      <div class="sub">{grade} · {horizon}-minute horizon</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("BTC", f"${price:,.2f}")
    c2.metric("Order Book", f"{book_imb*100:+.2f}%")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Elite model stack")
    st.write(f"**Logistic model:** {'UP' if p_logit >= .5 else 'DOWN'} {max(p_logit,1-p_logit)*100:.1f}%")
    st.write(f"**Random forest:** {'UP' if p_forest >= .5 else 'DOWN'} {max(p_forest,1-p_forest)*100:.1f}%")
    st.write(f"**Model agreement:** {'YES' if models_agree else 'NO'}")
    st.write(f"**Validation accuracy:** {validation['accuracy']*100:.1f}% · n={validation['n']}")
    st.write(f"**Brier score:** {validation['brier']:.3f} (lower is better)")
    if similar:
        hdir = "UP" if similar["p_up"] >= .5 else "DOWN"
        hconf = max(similar["p_up"], 1-similar["p_up"])
        st.write(f"**Similar historical setups:** {hdir} {hconf*100:.1f}% · n={similar['n']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Risk filter")
    st.write(f"**Market regime:** {'CHOPPY' if reg['choppy'] else 'ACTIVE / TRENDING'}")
    st.write(f"**Flip risk:** {reg['risk']}")
    st.write(f"**RSI:** {reg['rsi']:.1f}")
    st.write(f"**5-min return:** {reg['r5']*100:+.3f}%")
    st.write(f"**EMA gap:** {reg['ema_gap']*100:+.4f}%")
    st.write(f"**Realized volatility:** {reg['rv']*100:.3f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Derivatives confirmation")
    if derivatives is None:
        if cg_key and derivative_error:
            st.warning("CoinGlass key is configured, but derivatives data could not be loaded.")
            st.caption(derivative_error)
        else:
            st.info("CORE MODE — add a CoinGlass API key in Streamlit Secrets to activate OI, liquidations, funding, and taker-flow confirmation.")
    else:
        st.write(f"**Open interest:** ${derivatives['oi_usd']/1e9:.2f}B")
        st.write(f"**OI change 5m:** {derivatives['oi_5m']:+.2f}%")
        st.write(f"**OI change 15m:** {derivatives['oi_15m']:+.2f}%")
        st.write(f"**Long liquidations 1h:** ${derivatives['long_liq']/1e6:.2f}M")
        st.write(f"**Short liquidations 1h:** ${derivatives['short_liq']/1e6:.2f}M")
        taker_total = derivatives['taker_buy'] + derivatives['taker_sell']
        taker_imb = ((derivatives['taker_buy'] - derivatives['taker_sell']) / taker_total) if taker_total else 0
        st.write(f"**Taker buy/sell imbalance:** {taker_imb*100:+.1f}%")
        st.write(f"**OI-weighted funding:** {derivatives['funding']:+.5f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Order-book depth")
    st.write(f"Bid depth: **${bid_depth:,.0f}**")
    st.write(f"Ask depth: **${ask_depth:,.0f}**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.line_chart(df.tail(180).set_index("time")["close"], height=260)

    st.caption("Kraken supplies spot OHLC and L2 order-book data. CoinGlass derivatives confirmation is optional and requires an API key.")
    st.caption("Elite confidence is a model diagnostic, not a guaranteed probability of profit.")

except Exception as e:
    st.error(f"Elite model error: {e}")

time.sleep(refresh)
st.rerun()
