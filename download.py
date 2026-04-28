import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SPOT = "https://api.binance.com/api/v3/klines"
PERP = "https://fapi.binance.com/fapi/v1/klines"
FUND = "https://fapi.binance.com/fapi/v1/fundingRate"

SPOT_START = "2019-01-01"
SYMBOLS = {
    "BTCUSDT": "2019-09-08",
    "ETHUSDT": "2019-11-27",
}

LIMIT = 1000
SLEEP = 0.1
TIMEOUT = 20
TRAIN, VAL = 0.70, 0.15

OUT = Path(__file__).parent / "raw"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades",
    "taker_buy_vol", "taker_buy_quote_vol", "ignore",
]
OHLCV = ["open", "high", "low", "close", "volume"]
NUM_COLS = OHLCV + ["quote_vol", "n_trades", "taker_buy_vol", "taker_buy_quote_vol"]


def date_to_ms(d):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(url, symbol, start_ms, interval="1h"):
    rows, start = [], start_ms
    while True:
        r = requests.get(url, params={
            "symbol": symbol, "interval": interval,
            "startTime": start, "limit": LIMIT,
        }, timeout=TIMEOUT)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        start = chunk[-1][0] + 1
        if len(chunk) < LIMIT:
            break
        time.sleep(SLEEP)

    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df[NUM_COLS] = df[NUM_COLS].astype(float)
    return df


def fetch_funding(symbol, start_ms):
    rows, start = [], start_ms
    while True:
        r = requests.get(FUND, params={
            "symbol": symbol, "startTime": start, "limit": LIMIT,
        }, timeout=TIMEOUT)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        start = chunk[-1]["fundingTime"] + 1
        if len(chunk) < LIMIT:
            break
        time.sleep(SLEEP)
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df.set_index("fundingTime")[["fundingRate"]].sort_index()


def with_taker(perp):
    df = perp.copy()
    df["taker_buy_ratio"] = df["taker_buy_vol"] / df["volume"].replace(0, np.nan)
    df["taker_imbalance"] = df["taker_buy_ratio"] - 0.5
    return df[OHLCV + ["quote_vol", "n_trades", "taker_buy_vol",
                       "taker_buy_ratio", "taker_imbalance"]]


def split_counts(n):
    n_tr = int(n * TRAIN)
    n_va = int(n * VAL)
    return n_tr, n_va, n - n_tr - n_va


def main():
    OUT.mkdir(exist_ok=True)
    spot_ms = date_to_ms(SPOT_START)
    for sym, listed in SYMBOLS.items():
        ms = date_to_ms(listed)

        spot = fetch_klines(SPOT, sym, spot_ms)[OHLCV]
        spot.to_parquet(OUT / f"{sym}_spot_1h.parquet")

        perp = fetch_klines(PERP, sym, ms)
        with_taker(perp).to_parquet(OUT / f"{sym}_klines_1h.parquet")
        perp[OHLCV].to_parquet(OUT / f"{sym}_perp_1h.parquet")

        fund = fetch_funding(sym, ms)
        fund.to_parquet(OUT / f"{sym}_funding.parquet")

        n = len(perp)
        n_tr, n_va, n_te = split_counts(n)
        t_tr = perp.index[n_tr - 1].date()
        t_va = perp.index[n_tr + n_va - 1].date()
        t_end = perp.index[-1].date()
        print(f"{sym}  listed={listed}  end={t_end}  n={n}")
        print(f"        train={n_tr} (..{t_tr})  val={n_va} (..{t_va})  test={n_te}")


if __name__ == "__main__":
    main()
