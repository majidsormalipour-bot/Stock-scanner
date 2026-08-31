# -*- coding: utf-8 -*-
"""
اسکنر خودکار بازار سهام - نسخه پیشرفته
==========================================
این نسخه طوری طراحی شده که حتی اگر با اصطلاحات مالی آشنا نیستید،
خروجی قابل فهم و تصمیم‌های طراحی آن منطقی و محافظه‌کارانه باشد.

⚠️ قبل از هر چیز بخوانید:
------------------------------------------------
این اسکریپت یک "پیش‌بین بازار" نیست. کاری که می‌کند این است:
داده‌های عمومی مالی هر شرکت را می‌گیرد، چند فرمول ریاضی رویشان
اجرا می‌کند، و سهامی که طبق آن فرمول‌ها از بقیه بهتر به نظر
می‌رسند را نشان می‌دهد. این با "می‌دانم چه سهمی بالا می‌رود" خیلی
فرق دارد. هیچ ابزاری - حتی حرفه‌ای‌ترین صندوق‌های سرمایه‌گذاری -
نمی‌تواند آینده بازار را با قطعیت پیش‌بینی کند.
از این خروجی به‌عنوان "نقطه شروع تحقیق"، نه "حکم نهایی"، استفاده کنید.

نصب پیش‌نیازها:
    pip install yfinance pandas numpy lxml requests

اجرا (پیش‌فرض: ترکیب S&P 500 + یورواستوکس ۵۰، قیمت‌ها به یورو):
    python market_scanner.py
    python market_scanner.py --universe sp500 --top 15
    python market_scanner.py --universe eurostoxx50
یونیورس‌های موجود: sp500, nasdaq100, eurostoxx50, sp500_eurostoxx50
"""

import argparse
import time
import warnings
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# تنظیمات پیش‌فرض
# ----------------------------------------------------------------------

# وزن هر بُعد در امتیاز نهایی (جمعاً = ۱)
WEIGHT_GROWTH_VALUE = 0.35     # رشد و ارزندگی (P/E، EV/EBITDA، PEG، FCF yield نسبت به هم‌صنعتی‌ها)
WEIGHT_QUALITY = 0.30           # سلامت مالی نسبت به هم‌صنعتی‌ها (بدهی، سودآوری، جریان نقدی)
WEIGHT_TECHNICAL = 0.20         # روند قیمت + مومنتوم ۱۲-۱ ماهه (فاکتور آکادمیک)
WEIGHT_ANALYST = 0.15           # نظر تحلیل‌گران - وزن محدودتر چون این معیار سابقه سوگیری خوش‌بینانه دارد

HISTORY_PERIOD = "1y"
MAX_WORKERS = int(os.environ.get("SCANNER_MAX_WORKERS", "12"))
BATCH_DELAY_SECONDS = float(os.environ.get("SCANNER_BATCH_DELAY", "0"))
REQUEST_TIMEOUT_RETRIES = 3
MAX_PER_SECTOR = 3              # حداکثر چند سهم از یک صنعت در لیست نهایی

DISPLAY_CURRENCY = "EUR"    # واحد پول نمایشی گزارش. "USD" یا "EUR"


def get_usd_to_eur_rate() -> float:
    """نرخ لحظه‌ای تبدیل دلار به یورو را می‌گیرد؛ در صورت خطا نرخ تقریبی ثابت می‌دهد."""
    try:
        rate_hist = yf.Ticker("USDEUR=X").history(period="5d")
        if not rate_hist.empty:
            return float(rate_hist["Close"].iloc[-1])
    except Exception:
        pass
    print("  Could not fetch live USD/EUR rate, using fallback ~0.86")
    return 0.86


# ----------------------------------------------------------------------
# ۱) لیست نمادهای یونیورس
# ----------------------------------------------------------------------

def _fetch_html_tables(url: str) -> list:
    """صفحه را با هدر مرورگر واقعی می‌گیرد تا ویکی‌پدیا درخواست را مسدود نکند،
    سپس جدول‌های آن را با pandas می‌خواند."""
    import io
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    # نسخه‌های جدید pandas دیگر رشته HTML خام را قبول نمی‌کنند؛
    # باید آن را داخل StringIO بپیچیم.
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


# ترکیب فعلی یورواستوکس ۵۰ (تقریبی، بر اساس بازنگری سپتامبر ۲۰۲۵) با نمادهای
# صحیح Yahoo Finance برای هر بورس اروپایی. چون STOXX جدول آزاد و قابل
# scrape ساده منتشر نمی‌کند، این لیست را دستی و از منابع عمومی تهیه کرده‌ایم.
# ⚠️ ترکیب این شاخص هر سال (معمولاً سپتامبر) بازنگری می‌شود؛ اگر مدتی از
# تاریخ این کد گذشته، بهتر است لیست را با stoxx.com مقایسه کنید.
EUROSTOXX50_TICKERS = [
    # فرانسه (.PA)
    "MC.PA", "TTE.PA", "SAN.PA", "OR.PA", "AI.PA", "SU.PA", "BNP.PA", "CS.PA",
    "DG.PA", "AIR.PA", "BN.PA", "KER.PA", "RI.PA", "SAF.PA", "ENGI.PA",
    "ACA.PA", "GLE.PA", "EL.PA", "RMS.PA", "STLAP.PA",
    # آلمان (.DE)
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE", "VOW3.DE",
    "MBG.DE", "BMW.DE", "IFX.DE", "DBK.DE", "MUV2.DE", "DHL.DE", "EOAN.DE",
    "RWE.DE", "ADS.DE", "DB1.DE",
    # هلند (.AS)
    "ASML.AS", "ADYEN.AS", "PRX.AS", "INGA.AS", "AD.AS", "WKL.AS",
    # ایتالیا (.MI)
    "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "RACE.MI", "STMPA.MI",
    # اسپانیا (.MC)
    "IBE.MC", "SAN.MC", "BBVA.MC", "ITX.MC",
    # فنلاند (.HE)
    "NOKIA.HE", "KNEBV.HE",
    # بلژیک (.BR)
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
# ۲) دریافت داده هر سهم
# ----------------------------------------------------------------------

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
                "peg_ratio": info.get("pegRatio"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),  # نسبت‌ ارزندگی مستقل از ساختار سرمایه - بهتر از P/E برای مقایسه بین شرکت‌ها
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "roe": info.get("returnOnEquity"),
                "free_cashflow": info.get("freeCashflow"),
                "dividend_yield": info.get("dividendYield"),
                # نظر تحلیل‌گران حرفه‌ای وال‌استریت
                "target_mean_price": info.get("targetMeanPrice"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
                "recommendation": info.get("recommendationKey"),
            })
            if row.get("free_cashflow") and row.get("market_cap"):
                row["fcf_yield"] = row["free_cashflow"] / row["market_cap"]  # جریان نقدی آزاد نسبت به ارزش بازار - معیار ارزندگی مورد علاقه سرمایه‌گذاران ارزشی

            hist = tk.history(period=HISTORY_PERIOD)
            if not hist.empty:
                close = hist["Close"]
                sma50 = close.rolling(50).mean().iloc[-1]
                sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
                price = close.iloc[-1]
                # مومنتوم ۱۲-۱ ماهه: یک فاکتور شناخته‌شده در ادبیات مالی
                # (Jegadeesh & Titman) - بازده ۱۲ ماه اخیر بدون آخرین ماه،
                # چون ماه آخر معمولاً بازگشت کوتاه‌مدت (mean-reversion) نشان می‌دهد
                momentum_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else np.nan
                momentum_12_1 = (
                    (close.iloc[-22] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan
                )
                volatility = close.pct_change().std() * np.sqrt(252)

                row.update({
                    "current_price": price,
                    "price_date": hist.index[-1].strftime("%Y-%m-%d"),
                    "above_sma50": bool(price > sma50) if pd.notna(sma50) else None,
                    "above_sma200": bool(price > sma200) if pd.notna(sma200) else None,
                    "golden_cross": bool(sma50 > sma200) if pd.notna(sma50) and pd.notna(sma200) else None,
                    "momentum_3m": momentum_3m,
                    "momentum_12_1": momentum_12_1,
                    "volatility_annualized": volatility,
                })

                if row.get("target_mean_price") and price:
                    row["analyst_upside"] = (row["target_mean_price"] / price) - 1

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
    """داده هر نماد را می‌گیرد. برای جلوگیری از مسدودشدن توسط یاهو (که در
    سرورهای ابری خیلی حساس‌تر است)، درخواست‌ها را در دسته‌های کوچک با
    مکث بین هر دسته پردازش می‌کند، نه همه را همزمان."""
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
# ۳) فیلتر کیفیت - حذف داده ناقص و شرکت‌های خیلی کوچک/پرریسک
# ----------------------------------------------------------------------

def apply_quality_filters(df: pd.DataFrame, min_market_cap: float, fx_rate_usd_eur: float = None,
                           min_data_fields: int = 5) -> pd.DataFrame:
    before = len(df)
    if "error" in df.columns:
        error_count = df["error"].notna().sum()
        if error_count:
            print(f"  Note: {error_count} of {before} tickers had a fetch error "
                  f"(often Yahoo Finance rate-limiting on cloud IPs).")
            sample = df.loc[df["error"].notna(), "error"].dropna().iloc[:1]
            if not sample.empty:
                print(f"  Sample error: {sample.iloc[0]}")
        df = df[df["error"].isna()]

    if "market_cap" in df.columns and min_market_cap:
        # نکته مهم: در یونیورس ترکیبی، ارزش بازار شرکت‌های اروپایی به یورو و
        # شرکت‌های آمریکایی به دلار برمی‌گردد. برای مقایسه عادلانه، همه را
        # موقتاً به دلار تبدیل می‌کنیم؛ در غیر این صورت آستانه فیلتر برای
        # یک گروه سخت‌گیرانه‌تر یا آسان‌تر از گروه دیگر می‌شود.
        market_cap_usd = df["market_cap"].copy()
        if fx_rate_usd_eur and "currency" in df.columns:
            is_eur = df["currency"] == "EUR"
            market_cap_usd.loc[is_eur] = df.loc[is_eur, "market_cap"] / fx_rate_usd_eur
        df = df[market_cap_usd.fillna(0) >= min_market_cap]

    # حذف شرکت‌هایی با بدهی افراطی - اما نه بانک‌ها/بیمه‌ها که ذاتاً
    # به‌خاطر ماهیت کسب‌وکار (سپرده مشتریان = بدهی در ترازنامه) بدهی/سرمایه
    # بسیار بالایی دارند؛ این طبیعی است و نشانه ریسک نیست. اعمال یک آستانه
    # واحد روی همه صنایع، عملاً کل بخش مالی را حذف می‌کند - این یک خطای
    # روش‌شناسی رایج در غربالگری‌های ساده است.
    if "debt_to_equity" in df.columns and "sector" in df.columns:
        leverage_exempt_sectors = {"Financial Services", "Financials", "Utilities"}
        is_exempt = df["sector"].isin(leverage_exempt_sectors)
        safe_leverage = df["debt_to_equity"].isna() | (df["debt_to_equity"] < 300)
        df = df[is_exempt | safe_leverage]

    key_fields = [c for c in ["pe_ratio", "ev_to_ebitda", "revenue_growth", "profit_margin",
                               "momentum_3m", "roe", "target_mean_price"] if c in df.columns]
    df = df[df[key_fields].notna().sum(axis=1) >= min_data_fields]

    print(f"  Quality filter: {len(df)} of {before} stocks passed.")
    return df


# ----------------------------------------------------------------------
# ۴) امتیازدهی
# ----------------------------------------------------------------------

def normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """رتبه‌بندی صدکی (۰ تا ۱) به‌جای min-max ساده. دلیل: min-max به‌شدت
    نسبت به داده پرت حساس است - یک شرکت با رشد سود ۹۰۰٪ (مثلاً بازگشت از
    ضرر یک‌باره) کل مقیاس را کشیده و بقیه سهام را مصنوعاً فشرده نشان
    می‌دهد. رتبه‌بندی صدکی (percentile rank) استاندارد رایج در ساخت
    فاکتورهای کمّی (مثلاً چارچوب‌های AQR و MSCI Factor) است چون به داده‌های
    پرت حساس نیست."""
    s = series.astype(float)
    if s.dropna().empty:
        return pd.Series([0.5] * len(s), index=s.index)
    ranks = s.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranks = 1 - ranks
    return ranks


def sector_relative_normalize(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    """به‌جای مقایسه با کل بازار، هر سهم را با هم‌صنعتی‌های خودش مقایسه می‌کند.
    این کار جلوی مقایسه ناعادلانه (مثلاً بانک با شرکت فناوری) را می‌گیرد."""
    result = pd.Series(0.5, index=df.index)
    for sector, group in df.groupby("sector"):
        result.loc[group.index] = normalize(group[col], higher_is_better)
    return result


def score_growth_value(df: pd.DataFrame) -> pd.Series:
    score, n = pd.Series(0.0, index=df.index), 0
    if "pe_ratio" in df.columns:
        score += sector_relative_normalize(df, "pe_ratio", higher_is_better=False).fillna(0.5)
        n += 1
    if "ev_to_ebitda" in df.columns:
        # مستقل از ساختار سرمایه (بدهی) است؛ برای مقایسه بین شرکت‌ها از P/E
        # قابل‌اعتمادتر است چون تفاوت در میزان بدهی/مالیات را خنثی می‌کند
        score += sector_relative_normalize(df, "ev_to_ebitda", higher_is_better=False).fillna(0.5)
        n += 1
    if "peg_ratio" in df.columns:
        score += sector_relative_normalize(df, "peg_ratio", higher_is_better=False).fillna(0.5)
        n += 1
    if "fcf_yield" in df.columns:
        # جریان نقدی آزاد نسبت به ارزش بازار - معیار محبوب سرمایه‌گذاران
        # ارزشی (مشابه مفهوم "owner earnings" وارن بافت)
        score += normalize(df["fcf_yield"], True).fillna(0.5)
        n += 1
    if "revenue_growth" in df.columns:
        score += normalize(df["revenue_growth"], True).fillna(0.5)
        n += 1
    if "earnings_growth" in df.columns:
        score += normalize(df["earnings_growth"], True).fillna(0.5)
        n += 1
    return score / n if n else score


def score_quality(df: pd.DataFrame) -> pd.Series:
    """توجه: این معیارها را نسبت به هم‌صنعتی‌ها می‌سنجیم، نه کل بازار.
    حاشیه سود نرم‌افزار ذاتاً بالاتر از خرده‌فروشی است؛ این تفاوت صنعتی
    است، نه برتری واقعی یک شرکت بر دیگری."""
    score, n = pd.Series(0.0, index=df.index), 0
    if "profit_margin" in df.columns:
        score += sector_relative_normalize(df, "profit_margin", True).fillna(0.5); n += 1
    if "roe" in df.columns:
        score += sector_relative_normalize(df, "roe", True).fillna(0.5); n += 1
    if "debt_to_equity" in df.columns:
        score += sector_relative_normalize(df, "debt_to_equity", False).fillna(0.5); n += 1
    if "current_ratio" in df.columns:
        score += sector_relative_normalize(df, "current_ratio", True).fillna(0.5); n += 1
    if "free_cashflow" in df.columns:
        score += (df["free_cashflow"].fillna(0) > 0).astype(float); n += 1
    return score / n if n else score


def score_technical(df: pd.DataFrame) -> pd.Series:
    """توجه روش‌شناسی: RSI (اندیکاتور نوسان‌گیری کوتاه‌مدت) عمداً از این
    امتیاز بلندمدت حذف شده. به‌جایش از مومنتوم ۱۲-۱ ماهه استفاده می‌شود -
    یک فاکتور شناخته‌شده در ادبیات مالی آکادمیک (Jegadeesh & Titman, 1993)
    که برای افق سرمایه‌گذاری چندماهه مناسب‌تر است."""
    score, n = pd.Series(0.0, index=df.index), 0
    if "momentum_12_1" in df.columns:
        score += normalize(df["momentum_12_1"], True).fillna(0.5); n += 1
    elif "momentum_3m" in df.columns:
        score += normalize(df["momentum_3m"], True).fillna(0.5); n += 1
    if "above_sma50" in df.columns:
        score += df["above_sma50"].map({True: 1.0, False: 0.0}).fillna(0.5); n += 1
    if "above_sma200" in df.columns:
        score += df["above_sma200"].map({True: 1.0, False: 0.0}).fillna(0.5); n += 1
    if "golden_cross" in df.columns:
        score += df["golden_cross"].map({True: 1.0, False: 0.0}).fillna(0.5); n += 1
    # نوسان‌پذیری با وزن کمتر لحاظ می‌شود: نوسان بالا لزوماً بد نیست (خیلی
    # از سهام رشدی خوب نوسان بالایی دارند)؛ این فقط یک گرایش محافظه‌کارانه
    # خفیف است، نه جریمه سنگین.
    if "volatility_annualized" in df.columns:
        score += 0.5 * normalize(df["volatility_annualized"], False).fillna(0.5) + 0.25; n += 0.75
    return score / n if n else score


def score_analyst(df: pd.DataFrame) -> pd.Series:
    """توجه: قیمت هدف تحلیل‌گران در ادبیات مالی سابقه سوگیری خوش‌بینانه و
    واکنشی (نه پیش‌بینانه) دارد؛ به همین دلیل وزن این بخش در امتیاز کل
    نسبتاً محدود نگه داشته شده و نباید تنها معیار تصمیم‌گیری باشد."""
    score, n = pd.Series(0.0, index=df.index), 0
    if "analyst_upside" in df.columns:
        score += normalize(df["analyst_upside"], True).fillna(0.5); n += 1
    if "recommendation" in df.columns:
        rec_map = {"strong_buy": 1.0, "buy": 0.8, "hold": 0.5, "underperform": 0.2, "sell": 0.0}
        score += df["recommendation"].map(rec_map).fillna(0.5); n += 1
    return score / n if n else pd.Series(0.5, index=df.index)


def explain_pick(row: pd.Series) -> str:
    """یک توضیح ساده و انسانی درباره چرایی امتیاز بالای این سهم می‌سازد."""
    reasons = []
    if pd.notna(row.get("revenue_growth")) and row["revenue_growth"] > 0.10:
        reasons.append("رشد درآمد خوب")
    if pd.notna(row.get("profit_margin")) and row["profit_margin"] > 0.15:
        reasons.append("سودآوری بالا")
    if pd.notna(row.get("debt_to_equity")) and row["debt_to_equity"] < 50:
        reasons.append("بدهی کم")
    if pd.notna(row.get("fcf_yield")) and row["fcf_yield"] > 0.05:
        reasons.append("جریان نقدی آزاد قوی نسبت به ارزش بازار")
    if row.get("above_sma200"):
        reasons.append("روند قیمتی صعودی بلندمدت")
    if pd.notna(row.get("momentum_12_1")) and row["momentum_12_1"] > 0.15:
        reasons.append(f"بازده {row['momentum_12_1']*100:.0f}٪ در ۱۲ ماه اخیر")
    if pd.notna(row.get("analyst_upside")) and row["analyst_upside"] > 0.10:
        reasons.append(f"تحلیل‌گران حدود {row['analyst_upside']*100:.0f}٪ رشد بیشتر پیش‌بینی کرده‌اند")
    return "، ".join(reasons) if reasons else "امتیاز متعادل در چند معیار مختلف"


# ----------------------------------------------------------------------
# ۵) تنوع‌بخشی به لیست نهایی (جلوگیری از تمرکز در یک صنعت)
# ----------------------------------------------------------------------

def revalidate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """قیمت را برای لیست نهایی (تعداد کم) دوباره، جداگانه و ترتیبی با
    همان روش history() می‌گیرد (نه fast_info که در نسخه‌های مختلف
    yfinance ناسازگار است) تا از داده کهنه جلوگیری شود."""
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
# ۶) اجرای اصلی
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
    # ترکیبی که شرکت‌های اروپایی و آمریکایی دارد) و هم برای نمایش نهایی لازم است.
    fx_rate = get_usd_to_eur_rate() if DISPLAY_CURRENCY == "EUR" else None

    # توجه: پیام‌های ترمینال عمداً انگلیسی هستند، چون کنسول ویندوز متن
    # فارسی راست‌به‌چپ مخلوط با اعداد/انگلیسی را به‌هم‌ریخته نشان می‌دهد.
    # گزارش کامل فارسی در فایل HTML قابل مشاهده صحیح خواهد بود.
    print("Fetching price & fundamental data (this can take a few minutes)...")
    df = fetch_universe_data(tickers)
    df = apply_quality_filters(df, min_market_cap=min_market_cap, fx_rate_usd_eur=fx_rate)
    if df.empty:
        print("No stocks passed the quality filters. Try lowering --min-market-cap.")
        return df

    df["growth_value_score"] = score_growth_value(df)
    df["quality_score"] = score_quality(df)
    df["technical_score"] = score_technical(df)
    df["analyst_score"] = score_analyst(df)

    df["total_score"] = (
        df["growth_value_score"] * WEIGHT_GROWTH_VALUE
        + df["quality_score"] * WEIGHT_QUALITY
        + df["technical_score"] * WEIGHT_TECHNICAL
        + df["analyst_score"] * WEIGHT_ANALYST
    )
    df = df.sort_values("total_score", ascending=False)

    picks = diversify_top_picks(df, top_n)
    picks = revalidate_prices(picks)

    # تبدیل قیمت به ارز نمایشی، با توجه به ارز اصلی هر سهم (سهام اروپایی
    # از قبل به یورو هستند و نیازی به تبدیل ندارند؛ فقط سهام دلاری تبدیل می‌شوند)
    if DISPLAY_CURRENCY == "EUR" and "current_price" in picks.columns:
        is_usd = picks.get("currency", "USD") != "EUR"
        picks.loc[is_usd, "current_price"] = picks.loc[is_usd, "current_price"] * fx_rate

    csv_file = "market_scan_results.csv"
    df.to_csv(csv_file, encoding="utf-8-sig")

    html_file = "market_scan_report.html"
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
    """گزارش نهایی را به‌صورت یک صفحه HTML راست‌به‌چپ می‌سازد تا فارسی
    درست نمایش داده شود (برخلاف ترمینال ویندوز که این مشکل را دارد)."""
    symbol = "€" if currency == "EUR" else "$"
    rows_html = []
    for ticker, row in df.iterrows():
        price = row.get("current_price")
        price_str = f"{symbol}{price:.2f}" if pd.notna(price) else "—"
        stale_flag = ""
        if row.get("price_was_stale"):
            stale_flag = ' <span style="color:#ffb74d;font-weight:bold;">⚠️ قیمت اولیه کهنه بود؛ اصلاح شد</span>'
        price_date = row.get("price_date", "")
        rows_html.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="ticker">{ticker}</span>
            <span class="name">{row.get('short_name', '')}</span>
            <span class="sector">{row.get('sector', 'نامشخص')} · {row.get('country', '')}</span>
          </div>
          <div class="metrics">
            <span>امتیاز کل: <b>{row['total_score']:.2f}</b>/1.00</span>
            <span>قیمت فعلی: <b>{price_str}</b></span>
          </div>
          <div class="reason">چرا؟ {explain_pick(row)}{stale_flag}</div>
          <div style="font-size:11px;color:#666;margin-top:6px;">آخرین داده تاریخی: {price_date}</div>
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
<title>گزارش اسکن بازار سهام</title>
<style>
  body {{ font-family: Tahoma, Arial, sans-serif; background:#0f1115; color:#e6e6e6; padding:24px; max-width:820px; margin:auto; }}
  h1 {{ font-size:20px; }}
  .warning {{ background:#3a2a0f; border:1px solid #b8860b; padding:12px 16px; border-radius:8px; margin-bottom:20px; font-size:14px; line-height:1.8; }}
  .card {{ background:#1b1e26; border:1px solid #2c303a; border-radius:10px; padding:16px; margin-bottom:14px; }}
  .card-header {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:8px; }}
  .ticker {{ font-weight:bold; font-size:18px; color:#4fc3f7; direction:ltr; }}
  .name {{ color:#ccc; }}
  .sector {{ color:#888; font-size:13px; margin-right:auto; }}
  .metrics {{ display:flex; gap:20px; font-size:14px; color:#bbb; margin-bottom:8px; }}
  .metrics b {{ color:#fff; }}
  .reason {{ font-size:14px; color:#a8d8a8; }}
</style>
</head>
<body>
  <h1>🏆 پیشنهادهای برتر (حداکثر {MAX_PER_SECTOR} سهم از هر صنعت، برای تنوع)</h1>
  <p style="color:#888;font-size:13px;">{currency_note}</p>
  <div class="warning">
    ⚠️ این رتبه‌بندی صرفاً بر پایه فرمول‌های عددی روی داده گذشته/فعلی است و
    قطعیتی درباره آینده ندارد. جایگزین تحقیق شخصی یا مشورت با مشاور مالی
    دارای مجوز نیست. لطفاً قبل از خرید واقعی بررسی بیشتری کنید.
  </div>
  {''.join(rows_html)}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="اسکنر خودکار بازار سهام - نسخه پیشرفته")
    parser.add_argument("--universe", choices=list(UNIVERSE_LOADERS.keys()), default="sp500_eurostoxx50")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--min-market-cap", type=float, default=2e9)
    parser.add_argument("--tickers", type=str, default=None,
                        help="لیست نماد دستی جدا شده با کاما")
    args = parser.parse_args()

    custom = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    run(universe=args.universe, top_n=args.top, min_market_cap=args.min_market_cap, custom_tickers=custom)
