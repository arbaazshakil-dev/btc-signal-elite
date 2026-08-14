import math
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title="BTC Predictor Elite V5 Precision", page_icon="₿", layout="centered",
                   initial_sidebar_state="collapsed")

KRAKEN = "https://api.kraken.com/0/public"
OKX = "https://www.okx.com/api/v5"
HORIZON_MIN = 8

def get_json(url, params=None):
    r = requests.get(url, params=params or {}, timeout=15,
                     headers={"User-Agent": "btc-elite-free-v3/1.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=20)
def kraken_ohlc(interval=1):
    data = get_json(f"{KRAKEN}/OHLC", {"pair":"XBTUSD","interval":interval})
    if data.get("error"): raise RuntimeError(str(data["error"]))
    result = data["result"]
    key = next(k for k in result if k != "last")
    cols = ["time","open","high","low","close","vwap","volume","count"]
    df = pd.DataFrame(result[key], columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.dropna().sort_values("time").reset_index(drop=True)

@st.cache_data(ttl=10)
def kraken_book():
    d = get_json(f"{KRAKEN}/Depth", {"pair":"XBTUSD","count":50})
    if d.get("error"): raise RuntimeError(str(d["error"]))
    key = next(iter(d["result"]))
    bids, asks = d["result"][key]["bids"], d["result"][key]["asks"]
    bv = sum(float(p)*float(q) for p,q,*_ in bids)
    av = sum(float(p)*float(q) for p,q,*_ in asks)
    return 100*(bv-av)/(bv+av) if bv+av else 0.0, bv, av

@st.cache_data(ttl=10)
def okx_futures():
    inst = "BTC-USDT-SWAP"
    ticker = get_json(f"{OKX}/market/ticker", {"instId":inst})["data"][0]
    book = get_json(f"{OKX}/market/books", {"instId":inst,"sz":"50"})["data"][0]
    oi = get_json(f"{OKX}/public/open-interest", {"instId":inst})["data"][0]
    funding = get_json(f"{OKX}/public/funding-rate", {"instId":inst})["data"][0]
    bids, asks = book["bids"], book["asks"]
    bv = sum(float(x[0])*float(x[1]) for x in bids)
    av = sum(float(x[0])*float(x[1]) for x in asks)
    imb = 100*(bv-av)/(bv+av) if bv+av else 0.0
    last = float(ticker["last"])
    index_px = float(ticker.get("idxPx") or 0)
    # ticker idxPx may be absent; fetch index ticker as fallback
    if not index_px:
        try:
            idx = get_json(f"{OKX}/market/index-tickers", {"instId":"BTC-USDT"})["data"][0]
            index_px = float(idx["idxPx"])
        except Exception:
            index_px = last
    premium = 100*(last-index_px)/index_px if index_px else 0.0
    return {
        "last": last, "book_imb": imb, "oi": float(oi["oi"]),
        "funding": 100*float(funding["fundingRate"]),
        "premium": premium, "vol24": float(ticker.get("vol24h") or 0)
    }

def rsi(s, n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)

def add_indicators(df):
    x=df.copy()
    c=x["close"]
    x["r1"]=c.pct_change()
    x["r3"]=c.pct_change(3)
    x["r5"]=c.pct_change(5)
    x["ema9"]=c.ewm(span=9,adjust=False).mean()
    x["ema21"]=c.ewm(span=21,adjust=False).mean()
    x["ema50"]=c.ewm(span=50,adjust=False).mean()
    x["ema_gap"]=(x["ema9"]-x["ema21"])/c
    x["trend_gap"]=(x["ema21"]-x["ema50"])/c
    x["rsi"]=rsi(c)
    e12=c.ewm(span=12,adjust=False).mean()
    e26=c.ewm(span=26,adjust=False).mean()
    x["macd"]=e12-e26
    x["macd_signal"]=x["macd"].ewm(span=9,adjust=False).mean()
    x["macd_hist"]=(x["macd"]-x["macd_signal"])/c
    prev=c.shift(1)
    tr=pd.concat([(x.high-x.low),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x["atr_pct"]=tr.rolling(14).mean()/c
    vm=x["volume"].rolling(30).mean()
    vs=x["volume"].rolling(30).std().replace(0,np.nan)
    x["vol_z"]=((x["volume"]-vm)/vs).fillna(0)
    x["range_pct"]=(x["high"]-x["low"])/c
    x["body_pct"]=(x["close"]-x["open"])/x["open"]
    x["resistance"]=x["high"].rolling(30).max().shift(1)
    x["support"]=x["low"].rolling(30).min().shift(1)
    x["to_res"]=(x["resistance"]-c)/c
    x["to_sup"]=(c-x["support"])/c
    return x.replace([np.inf,-np.inf],np.nan)

FEATURES=["r1","r3","r5","ema_gap","trend_gap","rsi","macd_hist","atr_pct",
          "vol_z","range_pct","body_pct","to_res","to_sup"]

def train_models(df):
    x=add_indicators(df)
    # 8-minute forward target on 1m candles
    x["target"]=(x["close"].shift(-HORIZON_MIN)>x["close"]).astype(float)
    x.loc[x["close"].shift(-HORIZON_MIN).isna(),"target"]=np.nan
    z=x.dropna(subset=FEATURES+["target"]).copy()
    if len(z)<180: raise RuntimeError("Not enough clean history yet.")
    split=max(120,int(len(z)*0.78))
    tr, va=z.iloc[:split], z.iloc[split:]
    log=Pipeline([("scale",StandardScaler()),
                  ("model",LogisticRegression(max_iter=1000,C=0.7))])
    rf=RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=8,
                              random_state=42,class_weight="balanced")
    log.fit(tr[FEATURES],tr["target"].astype(int))
    rf.fit(tr[FEATURES],tr["target"].astype(int))
    lp=log.predict_proba(va[FEATURES])[:,1]
    rp=rf.predict_proba(va[FEATURES])[:,1]
    ep=(lp+rp)/2
    pred=(ep>=.5).astype(int)
    acc=accuracy_score(va["target"].astype(int),pred)
    brier=brier_score_loss(va["target"].astype(int),ep)
    latest=x.dropna(subset=FEATURES).iloc[[-1]]
    l=float(log.predict_proba(latest[FEATURES])[:,1][0])
    r=float(rf.predict_proba(latest[FEATURES])[:,1][0])
    return x, l, r, acc, brier, len(va)


def confidence_bucket_stats(df):
    x=add_indicators(df)
    x["target"]=(x["close"].shift(-HORIZON_MIN)>x["close"]).astype(float)
    x.loc[x["close"].shift(-HORIZON_MIN).isna(),"target"]=np.nan
    z=x.dropna(subset=FEATURES+["target"]).copy()
    if len(z)<220:
        return None

    split=max(140,int(len(z)*0.70))
    tr, va=z.iloc[:split], z.iloc[split:]

    log=Pipeline([("scale",StandardScaler()),
                  ("model",LogisticRegression(max_iter=1000,C=0.7))])
    rf=RandomForestClassifier(n_estimators=240,max_depth=5,min_samples_leaf=8,
                              random_state=123,class_weight="balanced")

    log.fit(tr[FEATURES],tr["target"].astype(int))
    rf.fit(tr[FEATURES],tr["target"].astype(int))

    lp=log.predict_proba(va[FEATURES])[:,1]
    rp=rf.predict_proba(va[FEATURES])[:,1]
    p=(lp+rp)/2
    pred=(p>=0.5).astype(int)
    conf=np.maximum(p,1-p)
    actual=va["target"].astype(int).values

    rows=[]
    for lo,hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.01)]:
        m=(conf>=lo)&(conf<hi)
        if m.sum():
            hit=(pred[m]==actual[m]).mean()
            rows.append({"lo":lo,"hi":hi,"n":int(m.sum()),"hit":float(hit)})
    return rows

def timeframe_vote(df):
    x=add_indicators(df).dropna(subset=["ema9","ema21","ema50","rsi","macd_hist"])
    q=x.iloc[-1]
    score=0
    score += 1 if q.ema9>q.ema21 else -1
    score += 1 if q.ema21>q.ema50 else -1
    score += 1 if q.macd_hist>0 else -1
    score += 1 if q.rsi>52 else (-1 if q.rsi<48 else 0)
    return score, q

def pct(v): return f"{v:+.2f}%"


if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

def settle_predictions(current_price):
    now = pd.Timestamp.now(tz="UTC")
    for p in st.session_state.prediction_log:
        if not p["settled"] and (now - p["time"]).total_seconds() >= HORIZON_MIN*60:
            p["exit_price"] = float(current_price)
            actual = "UP" if p["exit_price"] > p["entry_price"] else "DOWN"
            p["actual"] = actual
            p["win"] = actual == p["direction"]
            p["settled"] = True

def record_prediction(direction, confidence, price):
    now = pd.Timestamp.now(tz="UTC")
    # Avoid duplicate records from Streamlit reruns: at most one issued prediction per 8-minute block.
    block = int(now.timestamp() // (HORIZON_MIN*60))
    for p in st.session_state.prediction_log:
        if p.get("block") == block:
            return
    st.session_state.prediction_log.append({
        "time": now,
        "block": block,
        "direction": direction,
        "confidence": float(confidence),
        "entry_price": float(price),
        "exit_price": None,
        "actual": None,
        "win": None,
        "settled": False
    })
    # Keep phone session lightweight.
    st.session_state.prediction_log = st.session_state.prediction_log[-300:]


refresh_seconds = st.sidebar.slider(
    "Live refresh (seconds)",
    min_value=5,
    max_value=60,
    value=10,
    step=5
)
st.sidebar.caption("The dashboard reruns automatically while this page is open.")

try:
    d1=kraken_ohlc(1)

    live_now = pd.Timestamp.now(tz="UTC")
    st.caption(f"🟢 LIVE • Auto-refresh every {refresh_seconds}s • Last update {live_now.strftime('%H:%M:%S UTC')}")
    d5=kraken_ohlc(5)
    d15=kraken_ohlc(15)
    kb, bid_depth, ask_depth=kraken_book()
    fut=okx_futures()
    feat, lp, rp, acc, brier, nval=train_models(d1)
    bucket_stats=confidence_bucket_stats(d1)
    s1,q1=timeframe_vote(d1); s5,q5=timeframe_vote(d5); s15,q15=timeframe_vote(d15)

    base=(lp+rp)/2
    # Confirmation adjustment is intentionally modest; model probability remains dominant.
    confirm=0.0
    confirm += 0.015*np.sign(s1)
    confirm += 0.020*np.sign(s5)
    confirm += 0.025*np.sign(s15)
    confirm += 0.020*np.clip(kb/20,-1,1)
    confirm += 0.025*np.clip(fut["book_imb"]/20,-1,1)
    confirm += -0.010*np.clip(fut["funding"]/0.03,-1,1)
    confirm += 0.010*np.clip(fut["premium"]/0.10,-1,1)
    p_up=float(np.clip(base+confirm,0.05,0.95))
    p_dn=1-p_up

    model_agree=(lp>=.5)==(rp>=.5)
    tf_agree=(np.sign(s5)==np.sign(s15)) and s5!=0 and s15!=0
    books_agree=np.sign(kb)==np.sign(fut["book_imb"])
    atr=float(q1.atr_pct)
    choppy=(abs(float(q1.ema_gap))<0.00035 and abs(float(q1.macd_hist))<0.00012)
    flip_high=choppy or not tf_agree or not model_agree
    validation_ok=acc>=0.52 and brier<=0.255

    confidence=max(p_up,p_dn)
    direction="UP" if p_up>=.5 else "DOWN"
    # Precision-first gate: fewer calls, stronger agreement.
    strong_tf = abs(s5) >= 2 and abs(s15) >= 2
    strong_model = abs(lp-rp) <= 0.12
    book_supports_direction = (
        (direction=="UP" and (kb>0 or fut["book_imb"]>0)) or
        (direction=="DOWN" and (kb<0 or fut["book_imb"]<0))
    )
    trade_ok=(confidence>=0.64 and model_agree and tf_agree and strong_tf
              and strong_model and validation_ok and not flip_high
              and book_supports_direction)

    if trade_ok:
        status=f"{'🟢' if direction=='UP' else '🔴'} {direction} — {confidence*100:.1f}%"
        subtitle="PREDICTION CONFIRMED • 8-minute horizon"
        border="#25a55f" if direction=="UP" else "#d9534f"
        bg="#10281b" if direction=="UP" else "#2b1717"
    else:
        status="🟡 WAIT — NO TRADE"
        subtitle="NO PREDICTION • 8-minute horizon"
        border="#d7aa2b"; bg="#342d0d"

    st.title("₿ BTC Predictor Elite V5.3 — Live")
    st.markdown(
        '<div style="font-size:18px;font-weight:700;margin-top:-8px;margin-bottom:8px;">BAZI</div>',
        unsafe_allow_html=True
    )
    st.caption("Prediction-focused 8-minute BTC model with multi-timeframe confirmation, abstention, and validation")
    st.markdown(f"""<div style="padding:28px 16px;border:3px solid {border};border-radius:34px;
    background:{bg};text-align:center"><div style="font-size:18px">PREDICTION STATUS</div>
    <div style="font-size:38px;font-weight:800;margin:10px 0">{status}</div>
    <div style="font-size:20px;font-weight:700">{subtitle}</div></div>""",unsafe_allow_html=True)

    st.metric("BTC Spot",f"${float(d1.close.iloc[-1]):,.2f}")
    st.metric("Kraken Book",pct(kb))
    st.progress(int(round(confidence*100))/100)
    st.subheader("Model stack")
    st.write(f"**Logistic:** UP {lp*100:.1f}%")
    st.write(f"**Random forest:** UP {rp*100:.1f}%")
    st.write(f"**Adjusted ensemble:** UP {p_up*100:.1f}% • DOWN {p_dn*100:.1f}%")
    st.write(f"**Model agreement:** {'YES' if model_agree else 'NO'}")
    st.write(f"**Walk-forward validation:** {acc*100:.1f}% accuracy • Brier {brier:.3f} • n={nval}")

    st.subheader("Multi-timeframe confirmation")
    def vote_text(s): return "BULLISH" if s>0 else ("BEARISH" if s<0 else "NEUTRAL")
    st.write(f"**1m:** {vote_text(s1)} ({s1:+d})")
    st.write(f"**5m:** {vote_text(s5)} ({s5:+d})")
    st.write(f"**15m:** {vote_text(s15)} ({s15:+d})")
    st.write(f"**5m/15m agreement:** {'YES' if tf_agree else 'NO'}")

    st.subheader("Prediction quality")
    st.write(f"**Current prediction:** {direction}")
    st.write(f"**UP probability:** {p_up*100:.1f}%")
    st.write(f"**DOWN probability:** {p_dn*100:.1f}%")
    st.write(f"**Prediction issued:** {'YES' if trade_ok else 'NO — confidence filters not passed'}")
    st.write(f"**Validation accuracy:** {acc*100:.1f}% over {nval} held-out examples")
    st.write(f"**Brier score:** {brier:.3f} (lower is better)")
    if bucket_stats:
        current_conf=max(p_up,p_dn)
        chosen=None
        for b in bucket_stats:
            if b["lo"] <= current_conf < b["hi"]:
                chosen=b
                break
        if chosen:
            st.write(f"**Historical hit rate near this confidence:** {chosen['hit']*100:.1f}% · n={chosen['n']}")
        with st.expander("Confidence bucket history"):
            for b in bucket_stats:
                label=f"{int(b['lo']*100)}–{int(min(b['hi'],1)*100)}%"
                st.write(f"**{label}:** {b['hit']*100:.1f}% hit rate · n={b['n']}")

    st.subheader("Free futures confirmation")
    st.write(f"**OKX BTC swap:** ${fut['last']:,.2f}")
    st.write(f"**OKX swap order book:** {pct(fut['book_imb'])}")
    st.write(f"**Open interest:** {fut['oi']:,.2f} BTC")
    st.write(f"**Funding rate:** {fut['funding']:+.5f}%")
    st.write(f"**Premium:** {fut['premium']:+.5f}%")
    st.write(f"**24h swap volume:** {fut['vol24']:,.2f} BTC")

    st.subheader("Technical / risk filter")
    st.write(f"**Market regime:** {'CHOPPY' if choppy else 'TRENDING'}")
    st.write(f"**Flip risk:** {'HIGH' if flip_high else 'LOWER'}")
    st.write(f"**RSI:** {float(q1.rsi):.1f}")
    st.write(f"**ATR:** {atr*100:.3f}%")
    st.write(f"**5-min return:** {float(q1.r5)*100:+.3f}%")
    st.write(f"**Volume z-score:** {float(q1.vol_z):+.2f}")
    if pd.notna(q1.support) and pd.notna(q1.resistance):
        st.write(f"**30m support:** ${float(q1.support):,.2f}")
        st.write(f"**30m resistance:** ${float(q1.resistance):,.2f}")


    st.subheader("Live BTC chart")
    st.caption("Kraken BTC/USD • 1-minute candles • latest 180 minutes")

    chart_df = d1.tail(180)[["time", "close"]].copy()
    chart_df = chart_df.set_index("time")
    st.line_chart(
        chart_df,
        y="close",
        height=320,
        use_container_width=True
    )

    chart_cols = st.columns(3)
    chart_cols[0].metric("Last", f"${float(d1['close'].iloc[-1]):,.2f}")
    chart_cols[1].metric("30m High", f"${float(d1['high'].tail(30).max()):,.2f}")
    chart_cols[2].metric("30m Low", f"${float(d1['low'].tail(30).min()):,.2f}")

    st.subheader("Live 8-minute prediction tracker")
    settled=[p for p in st.session_state.prediction_log if p["settled"]]
    pending=[p for p in st.session_state.prediction_log if not p["settled"]]
    if settled:
        wins=sum(1 for p in settled if p["win"])
        st.write(f"**Session verified:** {wins}/{len(settled)} correct ({wins/len(settled)*100:.1f}%)")
        for window in [20,50,100]:
            sample=settled[-window:]
            if sample:
                w=sum(1 for p in sample if p["win"])
                st.write(f"**Last {len(sample)}:** {w}/{len(sample)} ({w/len(sample)*100:.1f}%)")
    else:
        st.write("No settled V5 predictions yet. The tracker scores a confirmed call after 8 minutes.")
    if pending:
        st.write(f"**Pending:** {len(pending)} prediction(s) waiting for the 8-minute outcome.")

    with st.expander("Prediction history"):
        if st.session_state.prediction_log:
            rows=[]
            for p in reversed(st.session_state.prediction_log[-50:]):
                rows.append({
                    "time": p["time"].strftime("%H:%M:%S"),
                    "call": p["direction"],
                    "confidence": f"{p['confidence']*100:.1f}%",
                    "entry": f"${p['entry_price']:,.2f}",
                    "exit": "" if p["exit_price"] is None else f"${p['exit_price']:,.2f}",
                    "result": "PENDING" if not p["settled"] else ("WIN" if p["win"] else "LOSS")
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.write("No confirmed predictions recorded in this app session.")

    st.caption("Prediction research only. V5 prioritizes precision by abstaining more often. Session tracking resets if the Streamlit app restarts; short-horizon BTC remains uncertain and no hit rate guarantees profit.")

    st.markdown(
        '<div style="text-align:center;opacity:.65;margin-top:24px;font-size:13px;">BAZI • BTC Predictor Elite</div>',
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(f"Market/model error: {e}")
    st.caption("Refresh once. If it persists, check Streamlit logs; public exchange APIs can occasionally rate-limit or change response fields.")

# Native Streamlit live loop: no third-party browser component required.
time.sleep(refresh_seconds)
st.rerun()
