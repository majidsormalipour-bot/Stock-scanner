"""
market_regime.py
ماژول شاخص رژیم بازار - برای *زمینه تفسیر*، نه فیلتر کردن سهام.

نکته طراحی مهم: این ماژول عمدا هیچ سهمی را حذف یا فیلتر نمی‌کند. هدف این
است که در بالای داشبورد یک برچسب وضعیت کلی بازار نمایش داده شود تا کاربر
بداند سیگنال‌های هر سه اسکنر را با چه میزان احتیاط بخواند - چون:

  - استراتژی دنبال‌کردن روند (short_term_scanner) در بازارهای پرنوسان/
    رنج‌زده معمولا ضربه می‌خورد (whipsaw): سیگنال شکست سطح صادر می‌شود،
    قیمت برمی‌گردد، ضرر محقق می‌شود.

  - استراتژی بازگشتی (reversal_scanner) در روندهای نزولی قوی و پایدار
    می‌تواند مدام «چاقوی در حال سقوط» بگیرد، حتی با فیلتر تله ارزشی که
    از قبل در پروژه هست - چون آن فیلتر فقط سلامت مالی را چک می‌کند، نه
    قدرت روند نزولی کلی بازار.

  - استراتژی بلندمدت (market_scanner) کمتر از این دو حساس است، ولی در
    روندهای نزولی شدید، حتی سهام باکیفیت هم می‌توانند مدتی همراه بازار
    افت کنند (correlation در ریزش‌ها معمولا افزایش می‌یابد).

این ماژول عمدا فیلتر نمی‌کند چون تصمیم این‌که «آیا این بار متفاوت است»
باید دست کاربر باشد، نه یک آستانه ثابت در کد.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("market_regime")


# ===========================================================================
# آستانه‌های طبقه‌بندی - بر اساس قواعد سرانگشتی رایج، نه بهینه‌سازی‌شده
# روی داده خاص؛ اگر لازم دیدید بر اساس تجربه خودتان تنظیم کنید.
# ===========================================================================

# سطوح VIX (شاخص نوسان ضمنی CBOE) - این آستانه‌ها سال‌هاست نسبتا پایدارند
VIX_LOW = 15.0
VIX_NORMAL_HIGH = 20.0
VIX_ELEVATED_HIGH = 30.0
# بالای ۳۰ = "high"، بالای ۴۰ عملا بحران (۲۰۰۸، مارس ۲۰۲۰) است

DRAWDOWN_CORRECTION = -0.10   # افت ۱۰٪+ از سقف = اصلاح
DRAWDOWN_BEAR = -0.20         # افت ۲۰٪+ از سقف = تعریف رایج بازار نزولی


@dataclass
class MarketRegime:
    trend_label: str              # strong_uptrend / uptrend / sideways / downtrend / strong_downtrend
    volatility_label: str         # low / normal / elevated / high / unknown
    price_vs_sma50_pct: float
    price_vs_sma200_pct: float
    drawdown_from_high_pct: float
    vix_level: Optional[float]
    as_of_date: str


# ===========================================================================
# محاسبه روند بر اساس قیمت نسبت به میانگین‌های متحرک
# ===========================================================================

def compute_trend_regime(prices: pd.Series, sma_short: int = 50, sma_long: int = 200) -> dict:
    """
    prices: سری زمانی قیمت پایانی شاخص مرجع (مثلا S&P 500)، مرتب بر اساس تاریخ صعودی.
    نیاز به حداقل sma_long روز داده دارد؛ در غیر این صورت 'insufficient_data' برمی‌گرداند.
    """
    if len(prices) < sma_long:
        return {"trend_label": "insufficient_data", "price_vs_sma50_pct": np.nan,
                "price_vs_sma200_pct": np.nan}

    last_price = prices.iloc[-1]
    sma50 = prices.rolling(sma_short, min_periods=sma_short).mean().iloc[-1]
    sma200 = prices.rolling(sma_long, min_periods=sma_long).mean().iloc[-1]

    pct_vs_50 = (last_price - sma50) / sma50
    pct_vs_200 = (last_price - sma200) / sma200

    above_50 = last_price > sma50
    above_200 = last_price > sma200
    sma50_above_sma200 = sma50 > sma200  # "golden cross" state

    if above_50 and above_200 and sma50_above_sma200 and pct_vs_200 > 0.05:
        label = "strong_uptrend"
    elif above_200:
        label = "uptrend" if above_50 else "sideways"
    elif not above_200 and not sma50_above_sma200 and pct_vs_200 < -0.05:
        label = "strong_downtrend"
    else:
        label = "downtrend" if not above_50 else "sideways"

    return {
        "trend_label": label,
        "price_vs_sma50_pct": round(float(pct_vs_50), 4),
        "price_vs_sma200_pct": round(float(pct_vs_200), 4),
    }


def compute_drawdown(prices: pd.Series, lookback_days: int = 252) -> float:
    """درصد افت قیمت فعلی نسبت به سقف lookback_days روز اخیر (۵۲ هفته تقریبی)."""
    window = prices.iloc[-lookback_days:] if len(prices) >= lookback_days else prices
    peak = window.max()
    last = prices.iloc[-1]
    return round(float((last - peak) / peak), 4)


def classify_volatility(vix_level: Optional[float]) -> str:
    """طبقه‌بندی ساده رژیم نوسان بر اساس سطح VIX."""
    if vix_level is None or (isinstance(vix_level, float) and np.isnan(vix_level)):
        return "unknown"
    if vix_level < VIX_LOW:
        return "low"
    if vix_level < VIX_NORMAL_HIGH:
        return "normal"
    if vix_level < VIX_ELEVATED_HIGH:
        return "elevated"
    return "high"


# ===========================================================================
# ترکیب همه سیگنال‌ها در یک شیء MarketRegime
# ===========================================================================

def build_market_regime(
    index_prices: pd.Series, vix_level: Optional[float] = None, as_of_date: Optional[str] = None
) -> MarketRegime:
    """
    نسخه قابل‌تست بدون شبکه: قیمت‌های شاخص و سطح VIX را مستقیم می‌گیرد
    (به‌جای فراخوانی مستقیم yfinance درون این تابع)، تا هم تست‌پذیر باشد
    و هم بتوان آن را با هر منبع داده‌ای (yfinance/Stooq) تغذیه کرد.
    """
    trend = compute_trend_regime(index_prices)
    drawdown = compute_drawdown(index_prices)
    vol_label = classify_volatility(vix_level)

    return MarketRegime(
        trend_label=trend["trend_label"],
        volatility_label=vol_label,
        price_vs_sma50_pct=trend["price_vs_sma50_pct"],
        price_vs_sma200_pct=trend["price_vs_sma200_pct"],
        drawdown_from_high_pct=drawdown,
        vix_level=vix_level,
        as_of_date=as_of_date or "",
    )


def fetch_market_regime_via_yfinance(
    index_ticker: str = "^GSPC", vix_ticker: str = "^VIX", lookback_days: int = 400
) -> Optional[MarketRegime]:
    """
    نسخه‌ای که واقعا از yfinance می‌گیرد (برای استفاده در اجرای روزانه).
    اگر yfinance نصب نباشد یا شبکه در دسترس نباشد، None برمی‌گرداند و
    اسکنرها باید graceful این حالت را مدیریت کنند (مثلا برچسب رژیم را
    در داشبورد نشان ندهند، نه این‌که کل اجرا fail شود).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance نصب نیست - رژیم بازار محاسبه نمی‌شود.")
        return None

    try:
        index_hist = yf.Ticker(index_ticker).history(period=f"{lookback_days}d")
        vix_hist = yf.Ticker(vix_ticker).history(period="5d")
        if index_hist.empty:
            logger.warning("داده شاخص %s خالی برگشت.", index_ticker)
            return None

        vix_level = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else None
        as_of = str(index_hist.index[-1].date())
        return build_market_regime(index_hist["Close"], vix_level=vix_level, as_of_date=as_of)
    except Exception as e:
        logger.warning("محاسبه رژیم بازار شکست خورد: %s", e)
        return None


# ===========================================================================
# متن تفسیری برای هر اسکنر - این‌ها فیلتر نیستند، فقط پیام هشدار/زمینه‌اند
# ===========================================================================

def regime_interpretation(regime: MarketRegime) -> dict[str, str]:
    """
    برای هر سه اسکنر یک جمله زمینه‌ای کوتاه تولید می‌کند که می‌توانید
    مستقیم بالای هر بخش در داشبورد HTML نمایش دهید.
    """
    notes = {}

    # --- market_scanner (بلندمدت) ---
    if regime.trend_label in ("strong_downtrend", "downtrend") and regime.drawdown_from_high_pct < DRAWDOWN_BEAR:
        notes["market_scanner"] = (
            "بازار در یک روند نزولی قابل‌توجه است؛ حتی سهام باکیفیت ممکن است "
            "موقتا همراه بازار افت کنند. امتیاز بالا لزوما به معنای ثبات کوتاه‌مدت نیست."
        )
    else:
        notes["market_scanner"] = "شرایط کلی بازار مانع خاصی برای این استراتژی ایجاد نمی‌کند."

    # --- short_term_scanner (دنبال‌کردن روند) ---
    if regime.trend_label == "sideways" or regime.volatility_label in ("elevated", "high"):
        notes["short_term_scanner"] = (
            "بازار رنج‌زده یا پرنوسان است - سیگنال‌های شکست سطح در این شرایط "
            "نرخ شکست بالاتری (whipsaw) دارند. احتیاط بیشتر در مدیریت حد ضرر توصیه می‌شود."
        )
    else:
        notes["short_term_scanner"] = "شرایط روند فعلی با منطق دنبال‌کردن روند این اسکنر همسو است."

    # --- reversal_scanner (بازگشتی/ارزش) ---
    if regime.trend_label == "strong_downtrend":
        notes["reversal_scanner"] = (
            "بازار در روند نزولی قوی است. فیلتر تله ارزشی موجود فقط سلامت مالی "
            "را چک می‌کند، نه قدرت روند نزولی کلی - ریسک «چاقوی در حال سقوط» در این شرایط بالاتر است."
        )
    else:
        notes["reversal_scanner"] = "روند کلی بازار مانع خاصی برای منطق بازگشت از حمایت ایجاد نمی‌کند."

    return notes


# ===========================================================================
# نمونه استفاده (در generate_index.py یا هر سه اسکنر، فقط برای نمایش)
# ===========================================================================
"""
from market_regime import fetch_market_regime_via_yfinance, regime_interpretation

regime = fetch_market_regime_via_yfinance()
if regime:
    notes = regime_interpretation(regime)
    # این را در بالای صفحه HTML به‌عنوان یک بنر زمینه‌ای نمایش دهید،
    # نه به‌عنوان فیلتری که سهام را حذف می‌کند:
    print(f"وضعیت بازار: {regime.trend_label} | نوسان: {regime.volatility_label} "
          f"| افت از سقف: {regime.drawdown_from_high_pct:.1%}")
    print(notes["short_term_scanner"])
"""
