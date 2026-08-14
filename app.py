
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss

KRAKEN = "https://api.kraken.com/0/public"
OKX = "https://www.okx.com"

FEATURES = [
    "r1","r3","r5","r8",
    "ema_gap","rsi","vol_z",
    "range_pct","body_pct",
    "atr_pct","realized_vol"
]

st.set_page_config(
    page_title="BTC Signal Elite Free",
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
.call {font-size:2.1rem;font-weight:800}
.sub {font-size:1.0rem;font-weight:700;margin-top:7px}
.card {padding:14px;border-radius:16px;background:#17243b;margin:8px 0}
.small {opacity:.78;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def get_json(url, params=None, timeout=15):
    r = requests.get(
        url,
        params=params or {},
        timeout=timeout,
        headers={"User-Agent": "btc-signal-elite-free/1.0"}
    )
    r.raise_for_status()
    return r.json()

def kraken_api(endpoint, params=None):
    data = get_json(f"{KRAKEN}/{endpoint}", params=params)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data["result"]

def okx_api(path, params=None):
    data = get_json(f"{OKX}{path}", params=params)
    if str(data.get("code")) != "0":
        raise RuntimeError(data.get("msg", "OKX request failed"))
    return data.get("data", [])

# -----------------------------
# Kraken spot data
# -----------------------------
@st.cache_data(ttl=10)
def get_kraken_ohlc(pair="XBTUSD"):
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
def get_kraken_ticker(pair="XBTUSD"):
    result = kraken_api("Ticker", {"pair": pair})
    key = list(result.keys())[0]
    return float(result[key]["c"][0])

@st.cache_data(ttl=5)
def get_kraken_book(pair="XBTUSD"):
    result = kraken_api("Depth", {"pair": pair, "count": 100})
    key = list(result.keys())[0]
    book = result[key]
    bid_notional = sum(float(x[0]) * float(x[1]) for x in book["bids"])
    ask_notional = sum(float(x[0]) * float(x[1]) for x in book["asks"])
    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total else 0.0
    return imbalance, bid_notional, ask_notional

# -----------------------------
# OKX free public derivatives
# -----------------------------
@st.cache_data(ttl=10)
def get_okx_swap_snapshot(inst_id="BTC-USDT-SWAP"):
    out = {
        "available": False,
        "price": None,
        "book_imb": 0.0,
        "bid_depth": 0.0,
        "ask_depth": 0.0,
        "oi": None,
        "funding": None,
        "premium": None,
        "volume_24h": None,
    }

    # Public swap ticker
    ticker = okx_api("/api/v5/market/ticker", {"instId": inst_id})
    if ticker:
        t = ticker[0]
        out["price"] = float(t.get("last", 0) or 0)
        out["volume_24h"] = float(t.get("volCcy24h", 0) or 0)

    # Public order book
    books = okx_api("/api/v5/market/books", {"instId": inst_id, "sz": "100"})
    if books:
        b = books[0]
        bids = b.get("bids", [])
        asks = b.get("asks", [])
        bid_notional = sum(float(x[0]) * float(x[1]) for x in bids)
        ask_notional = sum(float(x[0]) * float(x[1]) for x in asks)
        total = bid_notional + ask_notional
        out["book_imb"] = (bid_notional - ask_notional) / total if total else 0.0
        out["bid_depth"] = bid_notional
        out["ask_depth"] = ask_notional

    # Public open interest
    oi = okx_api(
        "/api/v5/public/open-interest",
        {"instType": "SWAP", "instId": inst_id}
    )
    if oi:
        out["oi"] = float(oi[0].get("oiCcy", 0) or oi[0].get("oi", 0) or 0)

    # Public funding
    funding = okx_api("/api/v5/public/funding-rate", {"instId": inst_id})
    if funding:
        f = funding[0]
        out["funding"] = float(f.get("fundingRate", 0) or 0)
        out["premium"] = float(f.get("premium", 0) or 0)

    out["available"] = True
    return out

# -----------------------------
# Feature engineering
# -----------------------------
def add_features(df):
    x = df.copy()

    for n in [1,3,5,8]:
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
    x["rsi"] = 100 - 100/(1+rs)
    x["rsi"] = x["rsi"].fillna(50).clip(0,100)

    vm = x["volume"].rolling(30).mean()
    vs = x["volume"].rolling(30).std()
    x["vol_z"] = (x["volume"] - vm) / vs.replace(0, np.nan)

    x["range_pct"] = (x["high"] - x["low"]) / x["close"]
    x["body_pct"] = (x["close"] - x["open"]) / x["open"]

    prev = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs()
    ], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / x["close"]

    x["realized_vol"] = x["r1"].rolling(20).std()

    return x.replace([np.inf,-np.inf], np.nan)

def target_and_mask(x, horizon):
    future_ret = x["close"].shift(-horizon)/x["close"] - 1
    y = (future_ret > 0).astype(int)
    mask = x[FEATURES].notna().all(axis=1) & future_ret.notna()
    return y, future_ret, mask

# -----------------------------
# Model stack
# -----------------------------
def make_logit():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])

def make_rf():
    return RandomForestClassifier(
        n_estimators=240,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced_subsample"
    )

def train_ensemble(x, horizon):
    y, _, mask = target_and_mask(x, horizon)
    idx = np.where(mask.values)[0]

    if len(idx) < 320:
        raise RuntimeError("Not enough valid candles to train model.")

    val_n = min(100, max(60, len(idx)//5))
    train_idx = idx[:-val_n]
    val_idx = idx[-val_n:]

    logit = make_logit()
    rf = make_rf()

    logit.fit(x.iloc[train_idx][FEATURES], y.iloc[train_idx])
    rf.fit(x.iloc[train_idx][FEATURES], y.iloc[train_idx])

    p1 = logit.predict_proba(x.iloc[val_idx][FEATURES])[:,1]
    p2 = rf.predict_proba(x.iloc[val_idx][FEATURES])[:,1]
    p = 0.55*p1 + 0.45*p2

    validation = {
        "accuracy": float(accuracy_score(y.iloc[val_idx].values, (p>=0.5).astype(int))),
        "brier": float(brier_score_loss(y.iloc[val_idx].values, p)),
        "n": int(len(val_idx)),
    }

    logit.fit(x.loc[mask, FEATURES], y.loc[mask])
    rf.fit(x.loc[mask, FEATURES], y.loc[mask])

    return logit, rf, validation

def similar_setups(x, horizon, row, k=100):
    y, _, mask = target_and_mask(x, horizon)
    hist = x.loc[mask, FEATURES].copy()
    if len(hist) < 120:
        return None

    mean = hist.mean()
    std = hist.std().replace(0,1)
    z_hist = (hist-mean)/std
    z_now = (row.iloc[0]-mean)/std
    dist = np.sqrt(((z_hist-z_now)**2).mean(axis=1))
    nearest = dist.nsmallest(min(k,len(dist))).index
    outcomes = y.loc[nearest]

    return {
        "p_up": float(outcomes.mean()),
        "n": int(len(outcomes))
    }

# -----------------------------
# Free derivatives confirmation
# -----------------------------
def free_derivatives_adjustment(okx, spot_price):
    if not okx or not okx.get("available"):
        return 0.0, []

    score = 0.0
    notes = []

    # Futures order-book imbalance
    ob = float(okx.get("book_imb",0) or 0)
    score += float(np.clip(ob*0.08, -0.08, 0.08))
    notes.append(f"OKX swap order book {ob*100:+.1f}%")

    # Funding used lightly as a contrarian crowding measure.
    funding = float(okx.get("funding",0) or 0)
    score += float(np.clip(-funding*8.0, -0.025, 0.025))
    notes.append(f"Funding {funding:+.5f}")

    # Futures/spot basis
    fut_price = okx.get("price")
    if fut_price and spot_price:
        basis = fut_price/spot_price - 1
        score += float(np.clip(basis*10.0, -0.025, 0.025))
        notes.append(f"Swap/spot basis {basis*100:+.3f}%")

    return float(np.clip(score,-0.12,0.12)), notes

def regime(x, spot_book, model_gap):
    rsi = float(x["rsi"].iloc[-1])
    ema = float(x["ema_gap"].iloc[-1])
    r5 = float(x["r5"].iloc[-1])
    rv = float(x["realized_vol"].iloc[-1])

    choppy = (
        abs(ema) < 0.00025 and
        abs(r5) < 0.0008 and
        42 <= rsi <= 58
    )

    if choppy or model_gap > 0.16:
        risk = "HIGH"
    elif abs(spot_book) < 0.025 or model_gap > 0.09:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "rsi":rsi,
        "ema":ema,
        "r5":r5,
        "rv":rv,
        "choppy":choppy,
        "risk":risk
    }

# -----------------------------
# App
# -----------------------------
st.title("₿ BTC Signal Elite — Free")
st.caption("Kraken spot + OKX public futures + ensemble model + validation + no-trade filtering")

with st.sidebar:
    pair = st.text_input("Kraken pair","XBTUSD").upper()
    okx_inst = st.text_input("OKX swap","BTC-USDT-SWAP").upper()
    horizon = st.slider("Prediction window (minutes)",1,30,8)
    min_conf = st.slider("Minimum signal confidence",0.52,0.80,0.62,0.01)
    refresh = st.slider("Refresh seconds",15,90,30)

try:
    df = get_kraken_ohlc(pair)
    x = add_features(df)
    row = x[FEATURES].iloc[[-1]]

    logit, rf, validation = train_ensemble(x,horizon)

    p_logit = float(logit.predict_proba(row)[0,1])
    p_rf = float(rf.predict_proba(row)[0,1])
    p_base = 0.55*p_logit + 0.45*p_rf
    model_gap = abs(p_logit-p_rf)

    spot_price = get_kraken_ticker(pair)
    spot_ob, spot_bids, spot_asks = get_kraken_book(pair)

    okx = None
    okx_error = None
    try:
        okx = get_okx_swap_snapshot(okx_inst)
    except Exception as e:
        okx_error = str(e)

    # Kraken spot book adjustment
    spot_adj = float(np.clip(spot_ob*0.06,-0.06,0.06))

    similar = similar_setups(x,horizon,row)
    hist_adj = 0.0
    if similar:
        hist_adj = float(np.clip((similar["p_up"]-0.5)*0.18,-0.07,0.07))

    fut_adj, fut_notes = free_derivatives_adjustment(okx,spot_price)

    p_final = float(np.clip(
        p_base + spot_adj + hist_adj + fut_adj,
        0.01,0.99
    ))

    direction = "UP" if p_final >= 0.5 else "DOWN"
    confidence = max(p_final,1-p_final)

    models_agree = (
        (p_logit>=0.5 and p_rf>=0.5) or
        (p_logit<0.5 and p_rf<0.5)
    )

    reg = regime(x,spot_ob,model_gap)
    validation_ok = validation["accuracy"] >= 0.52

    futures_available = bool(okx and okx.get("available"))

    eligible = (
        confidence >= min_conf and
        models_agree and
        validation_ok and
        not reg["choppy"] and
        reg["risk"] != "HIGH"
    )

    if eligible:
        emoji = "🟢" if direction=="UP" else "🔴"
        css = "up" if direction=="UP" else "down"
        status = f"{direction} — {confidence*100:.1f}%"
        grade = "FREE ELITE CONFIRMED"
    else:
        emoji = "🟡"
        css = "wait"
        status = "WAIT — NO TRADE"
        grade = "FILTERED"

    st.markdown(f"""
    <div class="signal {css}">
      <div class="small">ELITE FREE STATUS</div>
      <div class="call">{emoji} {status}</div>
      <div class="sub">{grade} · {horizon}-minute horizon</div>
    </div>
    """,unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    c1.metric("BTC Spot",f"${spot_price:,.2f}")
    c2.metric("Kraken Book",f"{spot_ob*100:+.2f}%")

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.write("### Model stack")
    st.write(f"**Logistic:** {'UP' if p_logit>=.5 else 'DOWN'} {max(p_logit,1-p_logit)*100:.1f}%")
    st.write(f"**Random forest:** {'UP' if p_rf>=.5 else 'DOWN'} {max(p_rf,1-p_rf)*100:.1f}%")
    st.write(f"**Model agreement:** {'YES' if models_agree else 'NO'}")
    st.write(f"**Validation accuracy:** {validation['accuracy']*100:.1f}% · n={validation['n']}")
    st.write(f"**Brier score:** {validation['brier']:.3f}")
    if similar:
        hdir = "UP" if similar["p_up"]>=.5 else "DOWN"
        hconf = max(similar["p_up"],1-similar["p_up"])
        st.write(f"**Similar setups:** {hdir} {hconf*100:.1f}% · n={similar['n']}")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.write("### Free futures confirmation")
    if futures_available:
        st.write(f"**OKX BTC swap:** ${okx['price']:,.2f}")
        st.write(f"**OKX swap order book:** {okx['book_imb']*100:+.2f}%")
        if okx.get("oi") is not None:
            st.write(f"**Open interest:** {okx['oi']:,.2f} BTC")
        if okx.get("funding") is not None:
            st.write(f"**Funding rate:** {okx['funding']*100:+.5f}%")
        if okx.get("premium") is not None:
            st.write(f"**Premium:** {okx['premium']*100:+.5f}%")
        if okx.get("volume_24h") is not None:
            st.write(f"**24h swap volume:** {okx['volume_24h']:,.2f} BTC")
    else:
        st.warning("OKX futures confirmation unavailable right now; core Kraken model is still running.")
        if okx_error:
            st.caption(okx_error)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.write("### Risk filter")
    st.write(f"**Market regime:** {'CHOPPY' if reg['choppy'] else 'ACTIVE / TRENDING'}")
    st.write(f"**Flip risk:** {reg['risk']}")
    st.write(f"**RSI:** {reg['rsi']:.1f}")
    st.write(f"**5-min return:** {reg['r5']*100:+.3f}%")
    st.write(f"**EMA gap:** {reg['ema']*100:+.4f}%")
    st.write(f"**Realized volatility:** {reg['rv']*100:.3f}%")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.write("### Kraken spot depth")
    st.write(f"Bid depth: **${spot_bids:,.0f}**")
    st.write(f"Ask depth: **${spot_asks:,.0f}**")
    st.markdown('</div>',unsafe_allow_html=True)

    st.line_chart(df.tail(180).set_index("time")["close"],height=260)

    st.caption("No paid API key required. Uses Kraken spot market data and OKX public futures endpoints when available.")
    st.caption("Signal confidence is a model diagnostic, not a guaranteed probability of profit.")

except Exception as e:
    st.error(f"Elite Free model error: {e}")

time.sleep(refresh)
st.rerun()
