# -*- coding: utf-8 -*-
"""
اسکنر فرصت‌های کوتاه‌مدت (Swing Trading Scanner)
====================================================
برخلاف market_scanner.py که برای سرمایه‌گذاری بلندمدت طراحی شده،
این اسکریپت دنبال سیگنال‌های کوتاه‌مدت (افق چند روز تا چند هفته)
می‌گردد: مومنتوم اخیر، افزایش حجم معاملات، شکست سطوح قیمتی،
و تقاطع MACD.

⚠️⚠️ هشدار جدی - قبل از استفاده حتماً بخوانید:
------------------------------------------------
معامله‌گری کوتاه‌مدت (Swing/Day Trading) ریسک بسیار بالاتری از
سرمایه‌گذاری بلندمدت دارد:
  - نوسان قیمت در بازه کوتاه بسیار غیرقابل پیش‌بینی‌تر است
  - این ابزار هیچ خبر یا رویداد آینده (گزارش مالی، اخبار شرکت) را
    پیش‌بینی نمی‌کند - فقط الگوی قیمتی گذشته را می‌بیند
  - سیگنال‌های این اسکریپت می‌توانند به‌سرعت (در عرض ساعت‌ها) باطل شوند
  - این ابزار **حد ضرر پیشنهادی** می‌دهد؛ رعایت آن برای مدیریت ریسک
    حیاتی است، چون در معاملات کوتاه‌مدت ضررهای کنترل‌نشده سریع
    جمع می‌شوند
این ابزار به هیچ عنوان توصیه مالی نیست و صرفاً یک نقطه شروع برای
تحقیق شخصی شماست.

نصب پیش‌نیازها:
    pip install yfinance pandas numpy lxml requests

اجرا (پیش‌فرض: ترکیب S&P 500 + یورواستوکس ۵۰، قیمت‌ها به یورو):
    python short_term_scanner.py
    python short_term_scanner.py --universe sp500 --top 15
یونیورس‌های موجود: sp500, nasdaq100, eurostoxx50, sp500_eurostoxx50
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

WEIGHT_MOMENTUM = 0.30      # مومنتوم اخیر قیمت (۵ و ۲۰ روزه)
WEIGHT_VOLUME = 0.15        # افزایش غیرعادی حجم معاملات (نشانه توجه بازار)
WEIGHT_MACD = 0.25          # تقاطع صعودی MACD
WEIGHT_BREAKOUT = 0.30      # نزدیکی به سقف/شکست سطح مقاومت

HISTORY_PERIOD = "6mo"
# روی سرورهای ابری (مثل GitHub Actions) یاهو فایننس درخواست‌های پرحجم و
# موازی را rate-limit یا مسدود می‌کند. با متغیر محیطی SCANNER_MAX_WORKERS
# و SCANNER_BATCH_DELAY می‌توان این مقادیر را برای اجرای ابری کاهش داد
# بدون این‌که روی اجرای محلی (سریع‌تر) تأثیر بگذارد.
MAX_WORKERS = int(os.environ.get("SCANNER_MAX_WORKERS", "6"))
BATCH_DELAY_SECONDS = float(os.environ.get("SCANNER_BATCH_DELAY", "0"))
REQUEST_TIMEOUT_RETRIES = 3
MAX_PER_SECTOR = 4
ATR_STOP_MULTIPLIER = 2.0   # حد ضرر پیشنهادی = قیمت - ۲ برابر ATR

DISPLAY_CURRENCY = "EUR"    # واحد پول نمایشی گزارش. "USD" یا "EUR"
                             # توجه: خود سهام همچنان در بورس آمریکا و به دلار
                             # معامله می‌شوند؛ این فقط برای راحتی نمایش شماست.


def get_usd_to_eur_rate() -> float:
    """نرخ لحظه‌ای تبدیل دلار به یورو را می‌گیرد. در صورت خطا یک نرخ
    تقریبی ثابت برمی‌گرداند تا اسکریپت متوقف نشود."""
    try:
        rate_hist = yf.Ticker("USDEUR=X").history(period="5d")
        if not rate_hist.empty:
            return float(rate_hist["Close"].iloc[-1])
    except Exception:
        pass
    print("  ⚠️ Could not fetch live USD/EUR rate, using fallback ~0.86")
    return 0.86


# ----------------------------------------------------------------------
# ۱) لیست نمادهای یونیورس (مشترک با اسکنر بلندمدت)
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


def get_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    for t in _fetch_html_tables(url):
        if "Ticker" in t.columns:
            return t["Ticker"].tolist()
        if "Symbol" in t.columns:
            return t["Symbol"].tolist()
    raise RuntimeError("جدول نمادهای Nasdaq-100 پیدا نشد.")


# ترکیب فعلی یورواستوکس ۵۰ (تقریبی، بر اساس بازنگری سپتامبر ۲۰۲۵).
# ⚠️ این شاخص هر سال (معمولاً سپتامبر) بازنگری می‌شود.
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
    "nasdaq100": get_nasdaq100_tickers,
    "eurostoxx50": get_eurostoxx50_tickers,
    "sp500_eurostoxx50": get_combined_tickers,
}


# ----------------------------------------------------------------------
# ۲) اندیکاتورهای کوتاه‌مدت
# ----------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan


def compute_macd(close: pd.Series):
    """MACD و خط سیگنال را برمی‌گرداند. تقاطع صعودی = فرصت ورود احتمالی."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Average True Range - برای تخمین نوسان‌پذیری و تعیین حد ضرر."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1]) if not atr.empty else np.nan


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
            })

            hist = tk.history(period=HISTORY_PERIOD)
            if hist.empty or len(hist) < 30:
                row["error"] = "insufficient_history"
                return row

            close = hist["Close"]
            volume = hist["Volume"]
            price = close.iloc[-1]

            # مومنتوم کوتاه‌مدت
            mom_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else np.nan
            mom_20d = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan

            # حجم معاملات نسبت به میانگین ۲۰ روزه
            avg_vol_20 = volume.rolling(20).mean().iloc[-1]
            vol_ratio = (volume.iloc[-1] / avg_vol_20) if avg_vol_20 and avg_vol_20 > 0 else np.nan

            # MACD
            macd_line, signal_line, hist_macd = compute_macd(close)
            macd_bullish_cross = bool(
                hist_macd.iloc[-1] > 0 and hist_macd.iloc[-2] <= 0
            ) if len(hist_macd) >= 2 else False
            macd_positive = bool(hist_macd.iloc[-1] > 0)

            # فاصله تا سقف/کف ۲۰ روزه (شکست سطح)
            high_20d = close.rolling(20).max().iloc[-1]
            low_20d = close.rolling(20).min().iloc[-1]
            pct_from_high = (price / high_20d - 1) if high_20d else np.nan
            near_breakout = bool(pct_from_high > -0.02)  # کمتر از ۲٪ تا سقف ۲۰ روزه

            rsi = compute_rsi(close)
            atr = compute_atr(hist)
            suggested_stop = price - ATR_STOP_MULTIPLIER * atr if pd.notna(atr) else np.nan

            row.update({
                "current_price": price,
                "price_date": hist.index[-1].strftime("%Y-%m-%d"),
                "mom_5d": mom_5d,
                "mom_20d": mom_20d,
                "volume_ratio": vol_ratio,
                "macd_bullish_cross": macd_bullish_cross,
                "macd_positive": macd_positive,
                "pct_from_20d_high": pct_from_high,
                "near_breakout": near_breakout,
                "rsi": rsi,
                "atr": atr,
                "suggested_stop": suggested_stop,
            })

            # نزدیکی به تاریخ گزارش مالی - ریسک نوسان ناگهانی
            try:
                cal = tk.calendar
                if isinstance(cal, dict) and cal.get("Earnings Date"):
                    next_earn = pd.to_datetime(cal["Earnings Date"][0])
                    days_to_earnings = (next_earn - pd.Timestamp.now()).days
                    row["days_to_earnings"] = days_to_earnings
            except Exception:
                pass

            return row
        except Exception as e:
            if attempt == REQUEST_TIMEOUT_RETRIES:
                row["error"] = str(e)
                return row
            # خطای ۴۲۹ (rate limit) نیاز به مکث بیشتر دارد تا یاهو IP را باز کند
            wait = 8 * (attempt + 1) if "429" in str(e) else 1.5 * (attempt + 1)
            time.sleep(wait)
    return row


def fetch_universe_data(tickers: list[str], max_workers: int = MAX_WORKERS,
                         batch_delay: float = BATCH_DELAY_SECONDS) -> pd.DataFrame:
    """داده هر نماد را می‌گیرد. برای جلوگیری از مسدودشدن توسط یاهو
    (که در سرورهای ابری خیلی حساس‌تر است)، به‌جای شلیک همه درخواست‌ها
    همزمان، آن‌ها را در دسته‌های کوچک با مکث بین هر دسته پردازش می‌کند."""
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
# ۳) فیلتر کیفیت
# ----------------------------------------------------------------------

def apply_quality_filters(df: pd.DataFrame, min_market_cap: float, fx_rate_usd_eur: float = None,
                           min_price: float = 5.0) -> pd.DataFrame:
    before = len(df)
    if "error" in df.columns:
        error_count = df["error"].notna().sum()
        if error_count:
            print(f"  Note: {error_count} of {before} tickers had a fetch error "
                  f"(often Yahoo Finance rate-limiting on cloud IPs).")
            # نمونه پیام خطا برای دیباگ - بدون این کار، دیدن دلیل واقعی سخت است
            sample = df.loc[df["error"].notna(), "error"].dropna().iloc[:1]
            if not sample.empty:
                print(f"  Sample error: {sample.iloc[0]}")
        df = df[df["error"].isna()]

    if "market_cap" in df.columns and min_market_cap:
        # نکته: در یونیورس ترکیبی، ارزش بازار شرکت‌های اروپایی به یورو و
        # آمریکایی به دلار است؛ برای مقایسه عادلانه هر دو را به دلار می‌بریم.
        market_cap_usd = df["market_cap"].copy()
        if fx_rate_usd_eur and "currency" in df.columns:
            is_eur = df["currency"] == "EUR"
            market_cap_usd.loc[is_eur] = df.loc[is_eur, "market_cap"] / fx_rate_usd_eur
        df = df[market_cap_usd.fillna(0) >= min_market_cap]

    # حذف سهام خیلی ارزان (penny-stock-like) که نوسان و ریسک غیرعادی دارند
    if "current_price" in df.columns:
        df = df[df["current_price"].fillna(0) >= min_price]
    print(f"  Quality filter: {len(df)} of {before} stocks passed.")
    return df


# ----------------------------------------------------------------------
# ۴) امتیازدهی کوتاه‌مدت
# ----------------------------------------------------------------------

def normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.astype(float)
    if s.dropna().empty:
        return pd.Series([0.5] * len(s), index=s.index)
    mn, mx = s.min(), s.max()
    if mn == mx:
        return pd.Series([0.5] * len(s), index=s.index)
    norm = (s - mn) / (mx - mn)
    return norm if higher_is_better else 1 - norm


def score_momentum(df: pd.DataFrame) -> pd.Series:
    score, n = pd.Series(0.0, index=df.index), 0
    if "mom_5d" in df.columns:
        score += normalize(df["mom_5d"], True).fillna(0.5); n += 1
    if "mom_20d" in df.columns:
        score += normalize(df["mom_20d"], True).fillna(0.5); n += 1
    if "rsi" in df.columns:
        # برای کوتاه‌مدت، RSI متوسط رو به بالا (۵۰-۷۰) بهتر از اشباع خرید/فروش است
        def rsi_score(r):
            if pd.isna(r): return 0.5
            if 50 <= r <= 70: return 0.9
            if r > 70: return 0.4  # اشباع خرید - ریسک برگشت
            if r < 40: return 0.3
            return 0.6
        score += df["rsi"].map(rsi_score); n += 1
    return score / n if n else score


def score_volume(df: pd.DataFrame) -> pd.Series:
    if "volume_ratio" not in df.columns:
        return pd.Series(0.5, index=df.index)
    # حجم بالاتر از میانگین = توجه بیشتر بازار؛ ولی خیلی افراطی (>5x) مشکوک/پرریسک است
    def vol_score(v):
        if pd.isna(v): return 0.5
        if 1.3 <= v <= 4: return 0.9
        if v > 4: return 0.6
        return 0.4
    return df["volume_ratio"].map(vol_score)


def score_macd(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.5, index=df.index)
    if "macd_bullish_cross" in df.columns:
        score = df["macd_bullish_cross"].map({True: 1.0, False: np.nan}).fillna(
            df.get("macd_positive", pd.Series(False, index=df.index)).map({True: 0.7, False: 0.3})
        )
    return score


def score_breakout(df: pd.DataFrame) -> pd.Series:
    score, n = pd.Series(0.0, index=df.index), 0
    if "pct_from_20d_high" in df.columns:
        score += normalize(df["pct_from_20d_high"], True).fillna(0.5); n += 1
    if "near_breakout" in df.columns:
        score += df["near_breakout"].map({True: 1.0, False: 0.4}); n += 1
    return score / n if n else score


def explain_pick(row: pd.Series) -> str:
    reasons = []
    if row.get("near_breakout"):
        reasons.append("نزدیک به شکست سقف ۲۰ روزه")
    if row.get("macd_bullish_cross"):
        reasons.append("تقاطع تازه و صعودی MACD")
    elif row.get("macd_positive"):
        reasons.append("مومنتوم MACD مثبت")
    if pd.notna(row.get("volume_ratio")) and row["volume_ratio"] > 1.3:
        reasons.append(f"حجم معاملات {row['volume_ratio']:.1f} برابر میانگین")
    if pd.notna(row.get("mom_5d")) and row["mom_5d"] > 0.03:
        reasons.append(f"رشد {row['mom_5d']*100:.1f}٪ در ۵ روز اخیر")
    if pd.notna(row.get("days_to_earnings")) and 0 <= row["days_to_earnings"] <= 5:
        reasons.append("⚠️ گزارش مالی نزدیک - نوسان غیرمنتظره محتمل")
    return "، ".join(reasons) if reasons else "امتیاز متعادل در چند سیگنال کوتاه‌مدت"


def revalidate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """قیمت لحظه‌ای واقعی را برای لیست نهایی (تعداد کم) دوباره، جداگانه و
    به‌صورت کاملاً ترتیبی می‌گیرد - نه با fast_info (که در نسخه‌های مختلف
    yfinance کلیدهای ناسازگار دارد و می‌تواند ساکت شکست بخورد)، بلکه با
    همان history() که مطمئن است. اجرای ترتیبی (بدون ThreadPoolExecutor)
    همچنین از یک باگ شناخته‌شده yfinance جلوگیری می‌کند که در آن درخواست‌های
    موازی گاهی داده یک نماد را با نماد دیگر قاطی می‌کنند."""
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

    # نرخ ارز را زود می‌گیریم چون هم برای فیلتر ارزش بازار (در یونیورس
    # ترکیبی اروپا+آمریکا) و هم برای نمایش نهایی لازم است.
    fx_rate = get_usd_to_eur_rate() if DISPLAY_CURRENCY == "EUR" else None

    print("Fetching short-term price/volume data...")
    df = fetch_universe_data(tickers)
    df = apply_quality_filters(df, min_market_cap=min_market_cap, fx_rate_usd_eur=fx_rate)
    if df.empty:
        print("No stocks passed the quality filters.")
        return df

    df["momentum_score"] = score_momentum(df)
    df["volume_score"] = score_volume(df)
    df["macd_score"] = score_macd(df)
    df["breakout_score"] = score_breakout(df)

    df["total_score"] = (
        df["momentum_score"] * WEIGHT_MOMENTUM
        + df["volume_score"] * WEIGHT_VOLUME
        + df["macd_score"] * WEIGHT_MACD
        + df["breakout_score"] * WEIGHT_BREAKOUT
    )
    df = df.sort_values("total_score", ascending=False)
    picks = diversify_top_picks(df, top_n)
    picks = revalidate_prices(picks)

    # تبدیل به ارز نمایشی؛ فقط سهام دلاری تبدیل می‌شوند، سهام اروپایی از
    # قبل به یورو هستند (هم قیمت و هم حد ضرر که از همان قیمت مشتق شده)
    if DISPLAY_CURRENCY == "EUR":
        is_usd = picks.get("currency", "USD") != "EUR"
        for col in ["current_price", "suggested_stop"]:
            if col in picks.columns:
                picks.loc[is_usd, col] = picks.loc[is_usd, col] * fx_rate

    csv_file = "short_term_scan_results.csv"
    df.to_csv(csv_file, encoding="utf-8-sig")

    html_file = "short_term_scan_report.html"
    write_html_report(picks, html_file, currency=DISPLAY_CURRENCY, fx_rate=fx_rate or 1.0)

    print(f"Done. Full data for {len(df)} stocks saved to {csv_file}")
    print(f"Readable Persian report saved to {html_file}")
    try:
        import webbrowser
        webbrowser.open(html_file)
    except Exception:
        pass

    return picks


def write_html_report(df: pd.DataFrame, path: str, currency: str = "USD", fx_rate: float = 1.0):
    symbol = "€" if currency == "EUR" else "$"
    rows_html = []
    for ticker, row in df.iterrows():
        price = row.get("current_price")
        price_str = f"{symbol}{price:.2f}" if pd.notna(price) else "—"
        stop = row.get("suggested_stop")
        stop_str = f"{symbol}{stop:.2f}" if pd.notna(stop) else "—"
        earnings_flag = ""
        if pd.notna(row.get("days_to_earnings")) and 0 <= row["days_to_earnings"] <= 5:
            earnings_flag = '<span class="earnings-warning">⚠️ گزارش مالی نزدیک</span>'
        stale_flag = ""
        if row.get("price_was_stale"):
            stale_flag = '<span class="earnings-warning">⚠️ قیمت اولیه دریافتی کهنه بود؛ اصلاح شد</span>'
        price_date = row.get("price_date", "")

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
            <span class="stop">حد ضرر پیشنهادی: <b>{stop_str}</b></span>
          </div>
          <div class="reason">چرا؟ {explain_pick(row)} {earnings_flag} {stale_flag}</div>
          <div class="datestamp">آخرین داده تاریخی: {price_date}</div>
        </div>""")

    if currency == "EUR":
        currency_note = (
            f'💶 قیمت‌ها به یورو نمایش داده شده‌اند (نرخ لحظه‌ای تقریباً '
            f'{1/fx_rate:.4f} دلار به ازای هر یورو). خود سهام همچنان در '
            f'بورس آمریکا و به دلار معامله می‌شوند؛ این فقط برای راحتی نمایش شماست.'
        )
    else:
        currency_note = "💵 همه قیمت‌ها به دلار آمریکا (USD) هستند."

    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<title>گزارش فرصت‌های کوتاه‌مدت</title>
<style>
  body {{ font-family: Tahoma, Arial, sans-serif; background:#0f1115; color:#e6e6e6; padding:24px; max-width:820px; margin:auto; }}
  h1 {{ font-size:20px; }}
  .warning {{ background:#3a1414; border:1px solid #c0392b; padding:12px 16px; border-radius:8px; margin-bottom:20px; font-size:14px; line-height:1.9; }}
  .card {{ background:#1b1e26; border:1px solid #2c303a; border-radius:10px; padding:16px; margin-bottom:14px; }}
  .card-header {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:8px; }}
  .ticker {{ font-weight:bold; font-size:18px; color:#ff9f43; direction:ltr; }}
  .name {{ color:#ccc; }}
  .sector {{ color:#888; font-size:13px; margin-right:auto; }}
  .metrics {{ display:flex; gap:20px; font-size:14px; color:#bbb; margin-bottom:8px; flex-wrap:wrap; }}
  .metrics b {{ color:#fff; }}
  .stop {{ color:#e57373; }}
  .reason {{ font-size:14px; color:#a8d8a8; }}
  .earnings-warning {{ color:#ffb74d; font-weight:bold; }}
  .datestamp {{ font-size:11px; color:#666; margin-top:6px; }}
</style>
</head>
<body>
  <h1>⚡ فرصت‌های کوتاه‌مدت (افق چند روز تا چند هفته)</h1>
  <p style="color:#888;font-size:13px;">{currency_note}</p>
  <div class="warning">
    ⚠️⚠️ معامله کوتاه‌مدت ریسک بالایی دارد. سیگنال‌های این گزارش می‌توانند
    به‌سرعت باطل شوند و هیچ خبر/رویداد آینده را پیش‌بینی نمی‌کنند.
    «حد ضرر پیشنهادی» بر اساس نوسان‌پذیری اخیر هر سهم (ATR) محاسبه شده -
    رعایت آن برای مدیریت ریسک توصیه می‌شود. این گزارش توصیه مالی نیست.
  </div>
  {''.join(rows_html)}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="اسکنر فرصت‌های کوتاه‌مدت")
    parser.add_argument("--universe", choices=list(UNIVERSE_LOADERS.keys()), default="sp500_eurostoxx50")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--min-market-cap", type=float, default=2e9)
    parser.add_argument("--tickers", type=str, default=None)
    args = parser.parse_args()

    custom = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    run(universe=args.universe, top_n=args.top, min_market_cap=args.min_market_cap, custom_tickers=custom)
