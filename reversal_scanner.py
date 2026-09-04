# -*- coding: utf-8 -*-
"""
اسکنر بازگشت از حمایت (Value at Support Scanner)
====================================================
این اسکریپت دنبال سهامی می‌گردد که:
  ۱) نسبت به اوج اخیرشان افت معناداری کرده‌اند (نه در روند صعودی)
  ۲) به یک سطح حمایت فنی نزدیک‌اند (نزدیک کف ۵۲ هفته یا میانگین ۲۰۰ روزه)
  ۳) نشانه اشباع فروش دارند ولی علائم تثبیت/توقف افت دیده می‌شود
  ۴) از نظر ارزندگی (P/E، EV/EBITDA) نسبت به هم‌صنعتی‌ها ارزان‌اند
  ۵) و مهم‌تر از همه: بنیاد مالی‌شان نشان نمی‌دهد در حال فروپاشی واقعی‌اند

⚠️ تفاوت کلیدی با short_term_scanner.py:
آن اسکریپت دنبال سهامی می‌گردد که در حال رشد و هم‌جهت با روند صعودی‌اند
(استراتژی «دنبال‌کردن روند»). این اسکریپت دقیقاً برعکس: دنبال سهامی
می‌گردد که افت کرده‌اند ولی ارزنده به‌نظر می‌رسند (استراتژی «بازگشت از
حمایت»). این دو فلسفه معامله‌گری متضادند و عمداً در دو گزارش جدا نگه
داشته شده‌اند - ترکیب‌شان در یک امتیاز، سیگنال را بی‌معنی می‌کند.

⚠️⚠️ هشدار مهم درباره «تله ارزشی» (Value Trap):
یک سهم گاهی ارزان است چون بازار به‌درستی پیش‌بینی کرده که آینده بدی در
پیش دارد (افول ساختاری کسب‌وکار، از دست‌دادن بازار، بدهی غیرقابل‌کنترل).
افت قیمت به‌تنهایی دلیل خرید نیست. این اسکریپت یک فیلتر «ایمنی بنیادی»
دارد تا حداقل موارد آشکارا در حال فروپاشی را حذف کند، ولی این تضمین
نمی‌کند سهم واقعاً بازخواهد گشت - قبل از خرید، دلیل افت قیمت را حتماً
بررسی کنید (اخبار، تغییر رهبری، از دست‌دادن مشتری کلیدی، و غیره).

نصب پیش‌نیازها:
    pip install yfinance pandas numpy lxml requests

اجرا:
    python reversal_scanner.py
    python reversal_scanner.py --universe sp500 --top 15
"""

import argparse
import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# تنظیمات
# ----------------------------------------------------------------------

WEIGHT_VALUE = 0.35          # ارزندگی نسبت به هم‌صنعتی‌ها (P/E، EV/EBITDA، PEG پایین)
WEIGHT_SUPPORT = 0.25        # نزدیکی به سطح حمایت (کف ۵۲ هفته / میانگین ۲۰۰ روزه)
WEIGHT_STABILIZING = 0.20    # نشانه توقف افت (اشباع فروش + تثبیت اخیر)
WEIGHT_SAFETY = 0.20         # فیلتر ایمنی بنیادی - رد تله ارزشی/شرکت در حال فروپاشی

HISTORY_PERIOD = "1y"
MAX_WORKERS = int(os.environ.get("SCANNER_MAX_WORKERS", "8"))
BATCH_DELAY_SECONDS = float(os.environ.get("SCANNER_BATCH_DELAY", "0"))
REQUEST_TIMEOUT_RETRIES = 3
MAX_PER_SECTOR = 4
MIN_DOLLAR_VOLUME = 3_000_000

# محدوده افت قیمت "جالب": کمتر از این افت نکرده (پس دیگر واقعاً "افتاده" نیست)
# و بیشتر از این افت کرده (احتمالاً یک مشکل ساختاری جدی دارد، نه یک اصلاح موقت)
MIN_DRAWDOWN_FROM_HIGH = 0.15   # حداقل ۱۵٪ افت از اوج ۵۲ هفته
MAX_DRAWDOWN_FROM_HIGH = 0.60   # بیش از ۶۰٪ افت = ریسک خیلی بالا، این اسکنر مناسبش نیست

DISPLAY_CURRENCY = "EUR"


def get_usd_to_eur_rate() -> float:
    try:
        rate_hist = yf.Ticker("USDEUR=X").history(period="5d")
        if not rate_hist.empty:
            rate = float(rate_hist["Close"].iloc[-1])
            if pd.notna(rate) and 0.5 < rate < 1.5:
                return rate
    except Exception:
        pass
    print("  Could not fetch live USD/EUR rate, using fallback ~0.86")
    return 0.86


# ----------------------------------------------------------------------
# ۱) یونیورس (همان S&P 500 + یورواستوکس ۵۰)
# ----------------------------------------------------------------------

def _fetch_html_tables(url: str) -> list:
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))


def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = _fetch_html_tables(url)[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


EUROSTOXX50_TICKERS = [
    "MC.PA", "TTE.PA", "SAN.PA", "OR.PA", "AI.PA", "SU.PA", "BNP.PA", "CS.PA",
    "DG.PA", "AIR.PA", "BN.PA", "KER.PA", "RI.PA", "SAF.PA", "ENGI.PA",
    "ACA.PA", "GLE.PA", "EL.PA", "RMS.PA", "STLAP.PA",
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE", "VOW3.DE",
    "MBG.DE", "BMW.DE", "IFX.DE", "DBK.DE", "MUV2.DE", "DHL.DE", "EOAN.DE",
    "RWE.DE", "ADS.DE", "DB1.DE",
    "ASML.AS", "ADYEN.AS", "PRX.AS", "INGA.AS", "AD.AS", "WKL.AS",
    "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "RACE.MI", "STMPA.MI",
    "IBE.MC", "SAN.MC", "BBVA.MC", "ITX.MC",
    "NOKIA.HE", "KNEBV.HE",
    "KBC.BR", "ABI.BR",
]


def get_eurostoxx50_tickers() -> list[str]:
    return EUROSTOXX50_TICKERS


def get_combined_tickers() -> list[str]:
    return get_sp500_tickers() + get_eurostoxx50_tickers()


UNIVERSE_LOADERS = {
    "sp500": get_sp500_tickers,
    "eurostoxx50": get_eurostoxx50_tickers,
    "sp500_eurostoxx50": get_combined_tickers,
}


# ----------------------------------------------------------------------
# ۲) دریافت داده هر سهم
# ----------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan


def fetch_one(ticker: str) -> dict:
    row = {"ticker": ticker}
    for attempt in range(REQUEST_TIMEOUT_RETRIES + 1):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            row.update({
                "short_name": info.get("shortName"),
                "sector": info.get("sector") or "Unknown",
                "country": info.get("country") or "Unknown",
                "currency": info.get("currency") or "USD",
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "peg_ratio": info.get("pegRatio"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "roe": info.get("returnOnEquity"),
                "free_cashflow": info.get("freeCashflow"),
            })

            hist = tk.history(period=HISTORY_PERIOD)
            if hist.empty or len(hist) < 60:
                row["error"] = "insufficient_history"
                return row

            close = hist["Close"]
            volume = hist["Volume"]
            price = close.iloc[-1]

            high_52w = close.max()
            low_52w = close.min()
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

            pct_from_high = (price / high_52w) - 1          # منفی = میزان افت از اوج
            pct_above_low = (price / low_52w) - 1            # چقدر بالای کف ۵۲ هفته‌ایم (نزدیک صفر = روی حمایت)
            pct_vs_sma200 = (price / sma200 - 1) if pd.notna(sma200) else np.nan

            rsi = compute_rsi(close)
            # تثبیت: آیا قیمت اخیراً کف جدید ۲۰ روزه نساخته؟ (نشانه توقف سقوط)
            recent_low_20d = close.iloc[-20:].min()
            made_new_low_last_3d = bool(close.iloc[-3:].min() <= recent_low_20d * 1.001)
            mom_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else np.nan

            avg_dollar_volume = float((close * volume).rolling(20).mean().iloc[-1])

            row.update({
                "current_price": price,
                "price_date": hist.index[-1].strftime("%Y-%m-%d"),
                "pct_from_52w_high": pct_from_high,
                "pct_above_52w_low": pct_above_low,
                "pct_vs_sma200": pct_vs_sma200,
                "rsi": rsi,
                "made_new_low_last_3d": made_new_low_last_3d,
                "mom_5d": mom_5d,
                "avg_dollar_volume": avg_dollar_volume,
            })
            return row
        except Exception as e:
            if attempt == REQUEST_TIMEOUT_RETRIES:
                row["error"] = str(e)
                return row
            wait = 8 * (attempt + 1) if "429" in str(e) else 1.5 * (attempt + 1)
            time.sleep(wait)
    return row


def fetch_universe_data(tickers: list[str], max_workers: int = MAX_WORKERS,
                         batch_delay: float = BATCH_DELAY_SECONDS) -> pd.DataFrame:
    records, total = [], len(tickers)
    for start in range(0, total, max_workers):
        batch = tickers[start:start + max_workers]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, t): t for t in batch}
            for future in as_completed(futures):
                records.append(future.result())
        done = min(start + max_workers, total)
        if done % 25 < max_workers or done == total:
            print(f"  Progress: {done}/{total} tickers processed...")
        if batch_delay and done < total:
            time.sleep(batch_delay)
    return pd.DataFrame(records).set_index("ticker")


# ----------------------------------------------------------------------
# ۳) فیلترها
# ----------------------------------------------------------------------

def apply_quality_filters(df: pd.DataFrame, min_market_cap: float, fx_rate_usd_eur: float = None,
                           min_data_fields: int = 4) -> pd.DataFrame:
    before = len(df)
    if "error" in df.columns:
        error_count = df["error"].notna().sum()
        if error_count:
            print(f"  Note: {error_count} of {before} tickers had a fetch error.")
        df = df[df["error"].isna()]
    print(f"  After error filter: {len(df)} remain.")

    if "market_cap" in df.columns and min_market_cap:
        market_cap_usd = df["market_cap"].copy()
        if fx_rate_usd_eur and "currency" in df.columns:
            is_eur = df["currency"] == "EUR"
            market_cap_usd.loc[is_eur] = df.loc[is_eur, "market_cap"] / fx_rate_usd_eur
        df = df[market_cap_usd.fillna(0) >= min_market_cap]
    print(f"  After market-cap filter: {len(df)} remain.")

    if "avg_dollar_volume" in df.columns:
        df = df[df["avg_dollar_volume"].fillna(0) >= MIN_DOLLAR_VOLUME]
    print(f"  After liquidity filter: {len(df)} remain.")

    # هسته اصلی این اسکنر: فقط سهامی که واقعاً در محدوده افت "جالب" هستند
    # - نه خیلی کم افتاده (دیگر "فرصت بازگشت" نیست)
    # - نه خیلی زیاد افتاده (احتمالاً یک مشکل ساختاری جدی دارد)
    if "pct_from_52w_high" in df.columns:
        drawdown = -df["pct_from_52w_high"]  # مثبت می‌کنیم برای خوانایی
        if not drawdown.dropna().empty:
            print(f"  Drawdown stats: min={drawdown.min():.3f}, median={drawdown.median():.3f}, "
                  f"max={drawdown.max():.3f}, non-null={drawdown.notna().sum()}")
        df = df[(drawdown >= MIN_DRAWDOWN_FROM_HIGH) & (drawdown <= MAX_DRAWDOWN_FROM_HIGH)]
    print(f"  After drawdown-range filter ({MIN_DRAWDOWN_FROM_HIGH}-{MAX_DRAWDOWN_FROM_HIGH}): {len(df)} remain.")

    key_fields = [c for c in ["pe_ratio", "ev_to_ebitda", "revenue_growth", "profit_margin",
                               "roe", "rsi"] if c in df.columns]
    df = df[df[key_fields].notna().sum(axis=1) >= min_data_fields]
    print(f"  After data-completeness filter (>={min_data_fields} of {len(key_fields)} fields): {len(df)} remain.")

    print(f"  Quality filter: {len(df)} of {before} stocks passed.")
    return df


# ----------------------------------------------------------------------
# ۴) امتیازدهی
# ----------------------------------------------------------------------

def normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.astype(float)
    if s.dropna().empty:
        return pd.Series([0.5] * len(s), index=s.index)
    ranks = s.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranks = 1 - ranks
    return ranks


def sector_relative_normalize(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    result = pd.Series(0.5, index=df.index)
    for sector, group in df.groupby("sector"):
        result.loc[group.index] = normalize(group[col], higher_is_better)
    return result


def score_value(df: pd.DataFrame) -> pd.Series:
    score, n = pd.Series(0.0, index=df.index), 0
    if "pe_ratio" in df.columns:
        score += sector_relative_normalize(df, "pe_ratio", False).fillna(0.5); n += 1
    if "ev_to_ebitda" in df.columns:
        score += sector_relative_normalize(df, "ev_to_ebitda", False).fillna(0.5); n += 1
    if "peg_ratio" in df.columns:
        score += sector_relative_normalize(df, "peg_ratio", False).fillna(0.5); n += 1
    return score / n if n else score


def score_support(df: pd.DataFrame) -> pd.Series:
    """هرچه به کف ۵۲ هفته یا میانگین ۲۰۰ روزه نزدیک‌تر (بدون شکستن آن)،
    امتیاز حمایت بالاتر - یعنی سهم روی یک سطح فنی مهم "نشسته" است."""
    score, n = pd.Series(0.0, index=df.index), 0
    if "pct_above_52w_low" in df.columns:
        # نزدیک‌تر به کف = بهتر؛ پس معکوس می‌کنیم
        score += normalize(df["pct_above_52w_low"], False).fillna(0.5); n += 1
    if "pct_vs_sma200" in df.columns:
        # نزدیک صفر (روی خط میانگین ۲۰۰ روزه) بهترین حالت است؛ خیلی بالا یا
        # خیلی پایین امتیاز کمتری می‌گیرد
        closeness = -df["pct_vs_sma200"].abs()
        score += normalize(closeness, True).fillna(0.5); n += 1
    return score / n if n else score


def score_stabilizing(df: pd.DataFrame) -> pd.Series:
    """اشباع فروش به‌تنهایی کافی نیست - باید نشانه‌ای از توقف افت هم باشد،
    وگرنه ممکن است در وسط یک سقوط آزاد بخریم."""
    score, n = pd.Series(0.0, index=df.index), 0
    if "rsi" in df.columns:
        def rsi_score(r):
            if pd.isna(r): return 0.5
            if r < 30: return 1.0    # اشباع فروش شدید
            if r < 40: return 0.8
            if r < 50: return 0.5
            return 0.2               # دیگر اشباع فروش نیست، فرصت کمتر
        score += df["rsi"].map(rsi_score); n += 1
    if "made_new_low_last_3d" in df.columns:
        score += df["made_new_low_last_3d"].map({True: 0.2, False: 0.9}).fillna(0.5); n += 1
    if "mom_5d" in df.columns:
        # افت شدید و ادامه‌دار در ۵ روز اخیر = هنوز در سقوط آزاد؛ کمی مثبت یا خنثی بهتر است
        def mom_score(m):
            if pd.isna(m): return 0.5
            if m < -0.08: return 0.1   # هنوز به‌شدت در حال سقوط
            if m < 0: return 0.5
            return 0.9                  # نشانه اولیه بازگشت
        score += df["mom_5d"].map(mom_score); n += 1
    return score / n if n else score


def score_safety(df: pd.DataFrame) -> pd.Series:
    """فیلتر ایمنی در برابر «تله ارزشی»: آیا این واقعاً یک فرصت است یا یک
    شرکت در حال فروپاشی؟ سودآوری، رشد نه‌چندان منفی، بدهی قابل‌مدیریت و
    جریان نقدی مثبت را می‌سنجد - نسبت به هم‌صنعتی‌ها."""
    score, n = pd.Series(0.0, index=df.index), 0
    if "profit_margin" in df.columns:
        score += sector_relative_normalize(df, "profit_margin", True).fillna(0.5); n += 1
    if "roe" in df.columns:
        score += sector_relative_normalize(df, "roe", True).fillna(0.5); n += 1
    if "debt_to_equity" in df.columns and "sector" in df.columns:
        leverage_exempt = {"Financial Services", "Financials", "Utilities"}
        is_exempt = df["sector"].isin(leverage_exempt)
        de_score = sector_relative_normalize(df, "debt_to_equity", False).fillna(0.5)
        de_score.loc[is_exempt] = 0.5  # برای بانک/بیمه بی‌طرف - بدهی بالا طبیعی است
        score += de_score; n += 1
    if "earnings_growth" in df.columns:
        # رشد سود به‌شدت منفی (نه فقط افت جزئی) نشانه هشدار جدی‌تری است
        def earn_score(e):
            if pd.isna(e): return 0.5
            if e < -0.30: return 0.1
            if e < 0: return 0.4
            return 0.8
        score += df["earnings_growth"].map(earn_score); n += 1
    if "free_cashflow" in df.columns:
        score += (df["free_cashflow"].fillna(0) > 0).astype(float); n += 1
    return score / n if n else score


def explain_pick(row: pd.Series) -> str:
    reasons = []
    if pd.notna(row.get("pct_from_52w_high")):
        reasons.append(f"{abs(row['pct_from_52w_high'])*100:.0f}٪ پایین‌تر از اوج ۵۲ هفته")
    if pd.notna(row.get("pct_above_52w_low")) and row["pct_above_52w_low"] < 0.10:
        reasons.append("خیلی نزدیک به کف ۵۲ هفته (سطح حمایت)")
    if pd.notna(row.get("rsi")) and row["rsi"] < 35:
        reasons.append(f"RSI در محدوده اشباع فروش ({row['rsi']:.0f})")
    if row.get("made_new_low_last_3d") is False:
        reasons.append("نشانه اولیه توقف افت")
    if pd.notna(row.get("profit_margin")) and row["profit_margin"] > 0:
        reasons.append("همچنان سودآور")
    if pd.notna(row.get("free_cashflow")) and row["free_cashflow"] > 0:
        reasons.append("جریان نقدی مثبت")
    return "، ".join(reasons) if reasons else "امتیاز متعادل در چند معیار"


def diversify_top_picks(df: pd.DataFrame, top_n: int, max_per_sector: int = MAX_PER_SECTOR) -> pd.DataFrame:
    selected, sector_counts = [], {}
    for ticker, row in df.iterrows():
        sector = row.get("sector", "Unknown")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= top_n:
            break
    return df.loc[selected]


def revalidate_prices(df: pd.DataFrame) -> pd.DataFrame:
    for ticker in df.index:
        try:
            fresh_hist = yf.Ticker(ticker).history(period="5d")
            if fresh_hist.empty:
                continue
            new_price = float(fresh_hist["Close"].iloc[-1])
            new_date = fresh_hist.index[-1].strftime("%Y-%m-%d")
            old_price = df.loc[ticker, "current_price"]
            if old_price and abs(new_price - old_price) / old_price > 0.03:
                df.loc[ticker, "price_was_stale"] = True
            df.loc[ticker, "current_price"] = new_price
            df.loc[ticker, "price_date"] = new_date
        except Exception:
            continue
    return df


# ----------------------------------------------------------------------
# ۵) اجرای اصلی
# ----------------------------------------------------------------------

def run(universe: str, top_n: int, min_market_cap: float, custom_tickers: list[str] | None):
    if custom_tickers:
        tickers = custom_tickers
    else:
        loader = UNIVERSE_LOADERS[universe]
        print(f"Fetching {universe} ticker list...")
        tickers = loader()
        print(f"  Found {len(tickers)} tickers.")

    fx_rate = get_usd_to_eur_rate() if DISPLAY_CURRENCY == "EUR" else None

    print("Fetching data for value/support screening...")
    df = fetch_universe_data(tickers)
    df = apply_quality_filters(df, min_market_cap=min_market_cap, fx_rate_usd_eur=fx_rate)
    if df.empty:
        print("No stocks passed the quality filters.")
        return df

    df["value_score"] = score_value(df)
    df["support_score"] = score_support(df)
    df["stabilizing_score"] = score_stabilizing(df)
    df["safety_score"] = score_safety(df)

    df["total_score"] = (
        df["value_score"] * WEIGHT_VALUE
        + df["support_score"] * WEIGHT_SUPPORT
        + df["stabilizing_score"] * WEIGHT_STABILIZING
        + df["safety_score"] * WEIGHT_SAFETY
    )
    df = df.sort_values("total_score", ascending=False)
    picks = diversify_top_picks(df, top_n)
    picks = revalidate_prices(picks)

    if DISPLAY_CURRENCY == "EUR" and pd.notna(fx_rate):
        original_prices = picks["current_price"].copy()
        is_usd = picks.get("currency", "USD") != "EUR"
        picks.loc[is_usd, "current_price"] = picks.loc[is_usd, "current_price"] * fx_rate
        broken = picks["current_price"].isna() & original_prices.notna() & is_usd
        if broken.any():
            picks.loc[broken, "current_price"] = original_prices[broken] * 0.86

    csv_file = "reversal_scan_results.csv"
    df.to_csv(csv_file, encoding="utf-8-sig")

    html_file = "reversal_scan_report.html"
    write_html_report(picks, html_file, currency=DISPLAY_CURRENCY, fx_rate=fx_rate or 1.0)

    print(f"Done. Full data for {len(df)} stocks saved to {csv_file}")
    try:
        import webbrowser
        webbrowser.open(html_file)
    except Exception:
        pass
    return picks


def write_html_report(df: pd.DataFrame, path: str, currency: str = "USD", fx_rate: float = 1.0):
    symbol = "€" if currency == "EUR" else "$"
    if currency == "EUR":
        currency_note = (
            f'💶 قیمت‌ها به یورو نمایش داده شده‌اند (نرخ لحظه‌ای تقریباً '
            f'{1/fx_rate:.4f} دلار به ازای هر یورو).'
        )
    else:
        currency_note = "💵 همه قیمت‌ها به دلار آمریکا (USD) هستند."

    rows_html = []
    for ticker, row in df.iterrows():
        price = row.get("current_price")
        price_str = f"{symbol}{price:.2f}" if pd.notna(price) else "—"
        stale_flag = ""
        if row.get("price_was_stale"):
            stale_flag = ' <span class="warn-inline">⚠️ قیمت اولیه کهنه بود؛ اصلاح شد</span>'
        rows_html.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="ticker">{ticker}</span>
            <span class="name">{row.get('short_name', '')}</span>
            <span class="sector">{row.get('sector', 'نامشخص')} · {row.get('country', '')}</span>
          </div>
          <div class="metrics">
            <span>امتیاز: <b>{row['total_score']:.2f}</b>/1.00</span>
            <span>قیمت: <b>{price_str}</b></span>
          </div>
          <div class="reason">چرا؟ {explain_pick(row)}{stale_flag}</div>
          <div class="datestamp">آخرین داده تاریخی: {row.get('price_date', '')}</div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<title>گزارش بازگشت از حمایت</title>
<style>
  body {{ font-family: Tahoma, Arial, sans-serif; background:#0f1115; color:#e6e6e6; padding:24px; max-width:820px; margin:auto; }}
  h1 {{ font-size:20px; }}
  .warning {{ background:#1a2a3a; border:1px solid #2b6cb0; padding:12px 16px; border-radius:8px; margin-bottom:20px; font-size:14px; line-height:1.9; }}
  .card {{ background:#1b1e26; border:1px solid #2c303a; border-radius:10px; padding:16px; margin-bottom:14px; }}
  .card-header {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:8px; }}
  .ticker {{ font-weight:bold; font-size:18px; color:#4dd0e1; direction:ltr; }}
  .name {{ color:#ccc; }}
  .sector {{ color:#888; font-size:13px; margin-right:auto; }}
  .metrics {{ display:flex; gap:20px; font-size:14px; color:#bbb; margin-bottom:8px; }}
  .metrics b {{ color:#fff; }}
  .reason {{ font-size:14px; color:#a8d8a8; }}
  .warn-inline {{ color:#ffb74d; font-weight:bold; }}
  .datestamp {{ font-size:11px; color:#666; margin-top:6px; }}
</style>
</head>
<body>
  <h1>🔄 بازگشت از حمایت (سهام افتاده ولی ارزنده)</h1>
  <p style="color:#888;font-size:13px;">{currency_note}</p>
  <div class="warning">
    ℹ️ این گزارش دقیقاً برعکس گزارش «فرصت‌های کوتاه‌مدت» عمل می‌کند: به‌جای
    سهام قوی و صعودی، دنبال سهام افتاده‌ای می‌گردد که نزدیک سطح حمایت و از
    نظر ارزندگی جذاب‌اند. ⚠️ افت قیمت همیشه فرصت نیست - گاهی دلیل واقعی و
    جدی دارد ("تله ارزشی"). قبل از خرید، حتماً دلیل افت را بررسی کنید. این
    گزارش توصیه مالی نیست.
  </div>
  {''.join(rows_html)}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="اسکنر بازگشت از حمایت")
    parser.add_argument("--universe", choices=list(UNIVERSE_LOADERS.keys()), default="sp500_eurostoxx50")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--min-market-cap", type=float, default=2e9)
    parser.add_argument("--tickers", type=str, default=None)
    args = parser.parse_args()

    custom = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    run(universe=args.universe, top_n=args.top, min_market_cap=args.min_market_cap, custom_tickers=custom)
