"""BAZI V11 unattended KXBTC15M prediction, resolution, and online learning.

Read-only market access: this worker contains no order endpoint and cannot trade.
Required environment: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone

import requests

CB = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
KALSHI = "https://external-api.kalshi.com/trade-api/v2"
SERIES = os.getenv("KALSHI_SERIES", "KXBTC15M")
PASSES = max(1, int(os.getenv("BAZI_PASSES", "4")))
PASS_INTERVAL = max(15, int(os.getenv("BAZI_PASS_INTERVAL", "45")))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
HEADERS = {"User-Agent": "BAZI-V11-Research/1.0", "Accept": "application/json"}
FEATURES = ("gap_vol", "mom_3", "mom_10", "book", "kraken_agree", "kalshi_market", "time_frac")
SCALES = {"gap_vol":3.0, "mom_3":1.5, "mom_10":1.5, "book":1.0,
          "kraken_agree":1.0, "kalshi_market":1.0, "time_frac":1.0}


def get(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()


def sb_headers(prefer=None):
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
         "Content-Type": "application/json"}
    if prefer:
        h["Prefer"] = prefer
    return h


def sb_get(table, params=None):
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", params=params,
                     headers=sb_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def sb_post(table, payload, prefer="return=representation"):
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", json=payload,
                      headers=sb_headers(prefer), timeout=15)
    if r.status_code == 409:  # unique market ticker: another run already logged it
        return []
    r.raise_for_status()
    return r.json() if r.content else []


def sb_patch(table, params, payload):
    r = requests.patch(f"{SUPABASE_URL}/rest/v1/{table}", params=params, json=payload,
                       headers=sb_headers("return=representation"), timeout=15)
    r.raise_for_status()
    return r.json() if r.content else []


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def market_expiry(market):
    return parse_time(market.get("close_time") or market.get("latest_expiration_time"))


def market_target(market):
    for key in ("floor_strike", "cap_strike"):
        value = market.get(key)
        if value not in (None, ""):
            try:
                value = float(value)
                if value > 1000:
                    return value
            except (TypeError, ValueError):
                pass
    text = " ".join(str(market.get(k, "")) for k in
                    ("functional_strike", "subtitle", "yes_sub_title", "title"))
    nums = re.findall(r"\$?([0-9]{2,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", text)
    return float(nums[-1].replace(",", "")) if nums else None


def yes_probability(market):
    vals = []
    for key in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"):
        try:
            value = float(market.get(key))
            if 0 <= value <= 1:
                vals.append(value)
        except (TypeError, ValueError):
            pass
    return sum(vals[:2])/len(vals[:2]) if vals else None


def market_snapshot():
    ticker = get(f"{CB}/products/BTC-USD/ticker")
    candles = get(f"{CB}/products/BTC-USD/candles", {"granularity":60})
    book = get(f"{CB}/products/BTC-USD/book", {"level":2})
    rows = sorted(candles, key=lambda x: x[0])[-31:]
    closes = [float(x[4]) for x in rows]
    rets = [math.log(b/a) for a,b in zip(closes, closes[1:]) if a>0 and b>0]
    bid = sum(float(x[1]) for x in book.get("bids", [])[:25])
    ask = sum(float(x[1]) for x in book.get("asks", [])[:25])
    try:
        k = get(f"{KRAKEN}/Ticker", {"pair":"XBTUSD"})
        kraken = float(next(iter(k["result"].values()))["c"][0])
    except Exception:
        kraken = None
    return {"cb":float(ticker["price"]), "kraken":kraken, "closes":closes,
            "rets":rets, "book":max(-1,min(1,(bid-ask)/max(bid+ask,1e-9)))}


def sigmoid(x):
    return 1/(1+math.exp(-max(-30,min(30,x))))


def normal_cdf(x):
    return .5*(1+math.erf(x/math.sqrt(2)))


def load_model():
    rows = sb_get("bazi_model", {"id":"eq.1", "select":"weights,examples"})
    if not rows:
        raise RuntimeError("bazi_model is missing; run bazi_supabase_schema.sql")
    return rows[0]["weights"], int(rows[0]["examples"])


def make_prediction(market, snap, seconds):
    closes, rets = snap["closes"], snap["rets"]
    vol = statistics.stdev(rets[-20:]) if len(rets)>=3 else .0006
    vol = max(.00015, min(.004, vol))
    m3 = math.log(closes[-1]/closes[-4]) if len(closes)>=4 else 0
    m10 = math.log(closes[-1]/closes[-11]) if len(closes)>=11 else m3
    target, kp = market_target(market), yes_probability(market)
    sigma = max(snap["cb"]*vol*math.sqrt(max(seconds,1)/60), snap["cb"]*.00008)
    expected = snap["cb"]*math.exp(max(-.004,min(.004,(.55*m3+.25*m10+.2*snap["book"]*vol)*(seconds/60)*.35)))
    if target:
        base = normal_cdf((expected-target)/sigma)
        gap_vol = max(-5,min(5,(snap["cb"]-target)/sigma))
    else:
        # Kalshi does not document a public live CFB reference. Without a target,
        # use direction features and enforce conservative confidence.
        base = sigmoid((.75*m3+.25*m10)/max(vol,1e-9)+.35*snap["book"])
        gap_vol = 0
    if kp is not None:
        base = .75*base + .25*kp  # contract alignment check, never primary feed
    kr_agree = 0 if snap["kraken"] is None else (1 if m3*(snap["kraken"]-snap["cb"])>=0 else -1)
    feats = {"gap_vol":gap_vol, "mom_3":max(-3,min(3,m3/max(vol,1e-9))),
             "mom_10":max(-3,min(3,m10/max(vol*math.sqrt(10),1e-9))),
             "book":snap["book"], "kraken_agree":kr_agree,
             "kalshi_market":0 if kp is None else 2*kp-1, "time_frac":seconds/900}
    weights,n = load_model()
    if n>=30:
        learned=sigmoid(weights.get("bias",0)+sum(weights.get(f,0)*feats[f]/SCALES[f] for f in FEATURES))
        blend=min(.35,.10+(n-30)/500)
        base=(1-blend)*base+blend*learned
    # Missing exact target or Kraken is a data-quality penalty, not hidden certainty.
    cap=.92 if target is not None and snap["kraken"] is not None else .82
    p_up=max(1-cap,min(cap,base))
    side="UP" if p_up>=.5 else "DOWN"
    confidence=p_up if side=="UP" else 1-p_up
    decision="CONFIDENT" if confidence>=.78 and target is not None and snap["kraken"] is not None else ("LEAN" if confidence>=.62 else "WAIT")
    return side,confidence,decision,feats,target,kp


def already_logged(ticker):
    rows = sb_get("bazi_predictions", {
        "market_ticker": f"eq.{ticker}",
        "select": "id",
        "limit": "1",
    })
    return bool(rows)


def discover_and_predict():
    """Capture each open KXBTC15M contract once while 1–15 minutes remain."""
    data = get(f"{KALSHI}/markets", {
        "series_ticker": SERIES,
        "status": "open",
        "limit": 100,
    })
    now = datetime.now(timezone.utc)
    candidates = []

    for market in data.get("markets", []):
        expiry = market_expiry(market)
        if not expiry:
            continue

        seconds = int((expiry - now).total_seconds())
        if 60 <= seconds <= 900:
            candidates.append((seconds, market, expiry))

    candidates.sort(key=lambda item: item[0])

    if not candidates:
        print("No KXBTC15M contract currently in the 1–15 minute capture window.")
        return

    # Avoid market-feed work when every eligible contract was already recorded.
    pending = [
        item for item in candidates
        if not already_logged(item[1].get("ticker", ""))
    ]
    if not pending:
        print("Eligible KXBTC15M contract(s) already logged.")
        return

    snap = market_snapshot()

    for seconds, market, expiry in pending:
        ticker = market.get("ticker")
        if not ticker:
            continue

        side, p, decision, features, target, kp = make_prediction(
            market, snap, seconds
        )
        payload = {
            "market_ticker": ticker,
            "series_ticker": SERIES,
            "expiry_at": expiry.isoformat(),
            "target": target,
            "coinbase_now": snap["cb"],
            "kraken_now": snap["kraken"],
            "kalshi_yes_probability": kp,
            "predicted_side": side,
            "probability": round(p, 6),
            "decision": decision,
            "features": features,
            "model_version": "v11",
        }
        created = sb_post("bazi_predictions", payload)
        print(
            ("Logged" if created else "Already logged"),
            ticker,
            decision,
            side,
            f"{p:.1%}",
            f"{seconds}s_to_expiry",
        )


def train(row, outcome_up):
    weights,n=load_model()
    x=row["features"]
    score=weights.get("bias",0)+sum(weights.get(f,0)*float(x.get(f,0))/SCALES[f] for f in FEATURES)
    error=int(outcome_up)-sigmoid(score)
    lr=max(.015,.08/math.sqrt(1+n/25)); l2=.002
    weights["bias"]=max(-3,min(3,weights.get("bias",0)+lr*error))
    for f in FEATURES:
        value=float(x.get(f,0))/SCALES[f]
        weights[f]=max(-3,min(3,weights.get(f,0)*(1-lr*l2)+lr*error*value))
    sb_patch("bazi_model",{"id":"eq.1"},{"weights":weights,"examples":n+1,
             "updated_at":datetime.now(timezone.utc).isoformat()})


def resolve_finished():
    now=datetime.now(timezone.utc).isoformat()
    rows=sb_get("bazi_predictions",{"resolved_at":"is.null","expiry_at":f"lt.{now}",
                "select":"id,market_ticker,predicted_side,features"})
    for row in rows:
        try:
            market=get(f"{KALSHI}/markets/{row['market_ticker']}").get("market",{})
            result=str(market.get("result","")).lower()
            status=str(market.get("status","")).lower()
            if result not in ("yes","no"):
                print("Awaiting official result",row["market_ticker"],status)
                continue
            outcome="UP" if result=="yes" else "DOWN"
            changed=sb_patch("bazi_predictions",{"id":f"eq.{row['id']}","resolved_at":"is.null"},
                {"outcome_side":outcome,"correct":outcome==row["predicted_side"],
                 "resolution_source":"KALSHI_OFFICIAL_RESULT",
                 "resolved_at":datetime.now(timezone.utc).isoformat()})
            if changed:
                train(row,outcome=="UP")
                print("Resolved and learned",row["market_ticker"],outcome)
        except Exception as exc:
            print("Resolution retry later",row["market_ticker"],exc)


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    failures = 0
    for pass_no in range(1, PASSES + 1):
        print(
            f"BAZI V11 pass {pass_no}/{PASSES} "
            f"at {datetime.now(timezone.utc).isoformat()}"
        )
        try:
            resolve_finished()
            discover_and_predict()
        except Exception as exc:
            failures += 1
            print(f"Pass {pass_no} failed: {exc}", file=sys.stderr)

        if pass_no < PASSES:
            time.sleep(PASS_INTERVAL)

    if failures == PASSES:
        raise RuntimeError("All BAZI V11 passes failed")


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print(f"BAZI worker failed: {exc}",file=sys.stderr)
        raise

