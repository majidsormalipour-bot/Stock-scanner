"""
data_validation.py
ماژول اعتبارسنجی و راستی‌آزمایی متقاطع داده برای اسکنرهای بازار سهام.

هدف: کاهش خطای ناشی از داده نادرست/ناقص yfinance، از طریق دو مکانیزم مستقل:

  1) راستی‌آزمایی متقاطع نمونه‌ای قیمت/حجم با Stooq (منبع دوم رایگان،
     بدون نیاز به API key) — چون نمی‌توان همه سهام را چک کرد بدون
     تشدید مشکل rate-limiting یاهو که در README مستند شده.

  2) پاک‌سازی آماری فیلدهای بنیادی قبل از ورود به فرمول امتیازدهی:
     - بازه منطقی فیزیکی (sanity bounds) برای هر فیلد
     - تشخیص outlier با MAD (Median Absolute Deviation) که در برابر
       چند مقدار پرت شدید مقاوم‌تر از z-score معمولی است

طراحی شده برای الحاق به هر سه اسکنر (market_scanner.py /
short_term_scanner.py / reversal_scanner.py) بدون تغییر منطق امتیازدهی
موجود — فقط قبل از مرحله رتبه‌بندی صدکی فراخوانی می‌شود.
"""

from __future__ import annotations

import io
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("data_validation")


# ===========================================================================
# بخش ۱: راستی‌آزمایی متقاطع قیمت/حجم با Stooq
# ===========================================================================

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


def _to_stooq_symbol(ticker: str) -> str:
    """
    تبدیل نماد یاهو به فرمت Stooq (فقط برای سهام آمریکایی قابل اعتماد است؛
    Stooq پوشش محدودتری برای سهام اروپایی دارد، پس این تابع را فقط برای
    یونیورس sp500 به‌کار ببرید، نه eurostoxx50).
    """
    return f"{ticker.lower().replace('.', '-')}.us"


def fetch_stooq_last_close(
    ticker: str, session: Optional[requests.Session] = None, timeout: int = 8
) -> Optional[dict]:
    """آخرین قیمت پایانی/حجم را از Stooq می‌گیرد؛ در صورت شکست None برمی‌گرداند."""
    sess = session or requests
    symbol = _to_stooq_symbol(ticker)
    url = STOOQ_URL.format(symbol=symbol)
    try:
        resp = sess.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or "Close" not in df.columns:
            return None
        last_row = df.iloc[-1]
        return {
            "date": last_row.get("Date"),
            "close": float(last_row["Close"]),
            "volume": float(last_row.get("Volume", np.nan)),
        }
    except Exception as e:
        logger.warning("Stooq fetch failed for %s: %s", ticker, e)
        return None


@dataclass
class ValidationResult:
    ticker: str
    field: str
    yf_value: float
    reference_value: float
    relative_diff: float
    flagged: bool
    reason: str = ""


def cross_validate_prices(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    price_col: str = "price",
    volume_col: Optional[str] = "volume",
    sample_frac: float = 0.15,
    min_sample: int = 10,
    price_tolerance: float = 0.03,
    volume_tolerance: float = 0.5,
    batch_delay: float = 1.0,
    random_seed: Optional[int] = None,
) -> list[ValidationResult]:
    """
    یک نمونه تصادفی از سهام (نه همه) را با Stooq راستی‌آزمایی می‌کند، تا خطر
    rate-limit اضافه‌ای روی همان مشکل مستندشده در README ایجاد نشود.

    price_tolerance=0.03 عمدا سخت‌گیر نیست، چون یاهو و Stooq گاهی چند دقیقه
    اختلاف تاخیر گزارش دارند و این طبیعی است، نه خطا.
    """
    n = max(min_sample, int(len(df) * sample_frac))
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=random_seed)

    results: list[ValidationResult] = []
    session = requests.Session()

    for i, (_, row) in enumerate(sample.iterrows()):
        ticker = row[ticker_col]
        ref = fetch_stooq_last_close(ticker, session=session)

        if ref is None:
            results.append(
                ValidationResult(
                    ticker=ticker,
                    field=price_col,
                    yf_value=row[price_col],
                    reference_value=np.nan,
                    relative_diff=np.nan,
                    flagged=False,
                    reason="reference_unavailable",
                )
            )
        else:
            yf_price = float(row[price_col])
            ref_price = ref["close"]
            if ref_price and not np.isnan(ref_price):
                rel_diff = abs(yf_price - ref_price) / ref_price
                flagged = rel_diff > price_tolerance
                results.append(
                    ValidationResult(
                        ticker=ticker,
                        field=price_col,
                        yf_value=yf_price,
                        reference_value=ref_price,
                        relative_diff=rel_diff,
                        flagged=flagged,
                        reason="price_mismatch" if flagged else "",
                    )
                )

            ref_vol = ref.get("volume", np.nan)
            if volume_col and volume_col in row and not np.isnan(ref_vol) and ref_vol > 0:
                yf_vol = float(row[volume_col])
                rel_diff_v = abs(yf_vol - ref_vol) / ref_vol
                flagged_v = rel_diff_v > volume_tolerance
                results.append(
                    ValidationResult(
                        ticker=ticker,
                        field=volume_col,
                        yf_value=yf_vol,
                        reference_value=ref_vol,
                        relative_diff=rel_diff_v,
                        flagged=flagged_v,
                        reason="volume_mismatch" if flagged_v else "",
                    )
                )

        # مکث بین درخواست‌ها - همان الگوی SCANNER_BATCH_DELAY موجود در پروژه
        if i < len(sample) - 1:
            time.sleep(batch_delay)

    return results


def validation_summary(results: list[ValidationResult]) -> dict:
    """
    خلاصه آماری برای لاگ روزانه. اگر flag_rate بالا باشد، این معمولا نشانه
    یک مشکل سیستمی (مثلا نرخ ارز اشتباه یا تاخیر کل بازار) است، نه چند
    سهم خراب مجزا — و باید کل خروجی آن روز با احتیاط بیشتری بررسی شود.
    """
    total = len(results)
    flagged = sum(r.flagged for r in results)
    unavailable = sum(r.reason == "reference_unavailable" for r in results)
    checked = total - unavailable
    flag_rate = (flagged / checked) if checked else 0.0
    return {
        "total_checked": checked,
        "unavailable": unavailable,
        "flagged": flagged,
        "flag_rate": round(flag_rate, 4),
        "systemic_alert": flag_rate > 0.20 and checked >= 5,
    }


# ===========================================================================
# بخش ۲: پاک‌سازی آماری فیلدهای بنیادی قبل از رتبه‌بندی صدکی
# ===========================================================================

# مقادیر خارج از این بازه تقریبا همیشه نشانه داده خراب‌اند، نه یک شرکت واقعی.
# این‌ها را با احتیاط و بر اساس یونیورس خودتان تنظیم کنید.
FIELD_SANITY_BOUNDS: dict[str, tuple[float, float]] = {
    "pe_ratio": (0, 500),
    "ev_ebitda": (-100, 500),
    "peg_ratio": (-10, 50),
    "debt_to_equity": (0, 2000),  # بانک‌ها/بیمه را جدا مدیریت کنید (طبق فیلتر موجود در README)
    "current_ratio": (0, 50),
    "roe": (-500, 500),  # درصد
    "fcf_yield": (-100, 100),  # درصد
    "rsi": (0, 100),
}


def sanity_check_field(field_name: str, value: float) -> tuple[bool, str]:
    """
    چک اول و ارزان: آیا مقدار اصلا در بازه فیزیکی/منطقی ممکن است؟
    این جایگزین MAD نیست، پیش از آن اجرا می‌شود — چون یک مقدار به‌شدت خراب
    می‌تواند خود میانه/MAD را هم منحرف کند وقتی تعداد سهام کم است.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False, "missing"
    bounds = FIELD_SANITY_BOUNDS.get(field_name)
    if bounds is None:
        return True, ""
    low, high = bounds
    if value < low or value > high:
        return False, f"out_of_bounds[{low},{high}]"
    return True, ""


def mad_outlier_mask(series: pd.Series, z_thresh: float = 5.0) -> pd.Series:
    """
    تشخیص outlier با Median Absolute Deviation. z_thresh=5 عمدا محافظه‌کارانه
    است تا سهام واقعا رشدی/ارزان حذف نشوند و فقط خطاهای فاحش داده
    (مثلا P/E=99999 به‌خاطر EPS نزدیک صفر) گرفته شوند.
    """
    clean = series.dropna()
    if len(clean) < 5:
        return pd.Series(False, index=series.index)
    median = clean.median()
    mad = (clean - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median).abs() / mad
    return modified_z > z_thresh


def validate_fundamentals(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """
    برای هر فیلد بنیادی، هم sanity bound و هم MAD-outlier را چک می‌کند.
    سیاست پیش‌فرض: مقدار مشکوک به NaN تبدیل می‌شود (نه حذف کل سهم)، چون
    رتبه‌بندی صدکی می‌تواند با NaN در یک معیار خاص کنار بیاید (وزن‌دهی مجدد
    روی معیارهای موجود) بدون این‌که کل سهم از اسکن حذف شود.

    ستون‌های `<field>_bound_flag` و `<field>_mad_flag` برای شفافیت/دیباگ
    آینده نگه داشته می‌شوند (شبیه بخش «باگ‌های مهم» در README).
    """
    out = df.copy()
    for field_name in fields:
        if field_name not in out.columns:
            continue

        bound_flags = [sanity_check_field(field_name, v)[1] for v in out[field_name]]
        out[f"{field_name}_bound_flag"] = bound_flags
        out[f"{field_name}_mad_flag"] = mad_outlier_mask(out[field_name])

        needs_clean = (out[f"{field_name}_bound_flag"] != "") | out[f"{field_name}_mad_flag"]
        n_cleaned = int(needs_clean.sum())
        if n_cleaned:
            logger.info("%s: %d مقدار مشکوک به NaN تبدیل شد.", field_name, n_cleaned)
        out.loc[needs_clean, field_name] = np.nan

    return out


# ===========================================================================
# نمونه استفاده در هر اسکنر (mainهای موجود را دست نمی‌زند، فقط قبل از
# رتبه‌بندی صدا زده می‌شود)
# ===========================================================================
"""
from data_validation import (
    validate_fundamentals, cross_validate_prices, validation_summary,
)

# --- قبل از رتبه‌بندی صدکی، فیلدهای بنیادی را پاک‌سازی کنید ---
df = validate_fundamentals(df, fields=["pe_ratio", "ev_ebitda", "debt_to_equity", "roe"])

# --- بعد از جمع‌آوری قیمت/حجم، یک نمونه را با Stooq چک کنید ---
results = cross_validate_prices(df, sample_frac=0.15, batch_delay=1.0)
summary = validation_summary(results)
logger.info("Cross-validation summary: %s", summary)

if summary["systemic_alert"]:
    # پیشنهاد: اجرای امروز را با یک برچسب هشدار در داشبورد منتشر کنید،
    # نه این‌که کل اجرا را متوقف کنید (که با ماهیت اسکن روزانه ناسازگار است)
    logger.warning("نرخ پرچم‌خوردن غیرعادی بالاست - داده امروز را با احتیاط منتشر کنید.")
"""
