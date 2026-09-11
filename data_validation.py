"""
data_validation.py
ماژول اعتبارسنجی و راستی‌آزمایی متقاطع داده برای اسکنرهای بازار سهام.

هدف: کاهش خطای ناشی از داده نادرست/ناقص yfinance، از طریق دو مکانیزم مستقل:

  1) راستی‌آزمایی متقاطع نمونه‌ای قیمت/حجم با Twelve Data (منبع دوم رایگان،
     نیاز به یک API key رایگان دارد) — چون نمی‌توان همه سهام را چک کرد
     بدون تشدید مشکل rate-limiting یاهو که در README مستند شده.

     نکته مهم درباره انتخاب منبع: در ابتدا Stooq امتحان شد، ولی مشخص شد
     Stooq با یک مکانیزم کپچا/کلید IP-محور کار می‌کند که فقط برای IPهای
     مسدودشده (مثل دیتاسنترها/GitHub Actions) کپچا نشان می‌دهد - یعنی
     گرفتن کلید از یک مرورگر خانگی (IP غیرمسدود) اصلا امکان‌پذیر نیست،
     و even اگر کلیدی به‌دست بیاید معلوم نیست از IP دیتاسنتر کار کند.
     Twelve Data برخلاف آن، دقیقا برای دسترسی برنامه‌ای/ابری طراحی شده:
     ثبت‌نام با ایمیل، بدون کپچا، کلید فوری کار می‌کند - از جمله از
     GitHub Actions. پلن رایگان: ۸۰۰ درخواست در روز، ۸ درخواست در دقیقه.

  2) پاک‌سازی آماری فیلدهای بنیادی قبل از ورود به فرمول امتیازدهی:
     - بازه منطقی فیزیکی (sanity bounds) برای هر فیلد
     - تشخیص outlier با MAD (Median Absolute Deviation) که در برابر
       چند مقدار پرت شدید مقاوم‌تر از z-score معمولی است

طراحی شده برای الحاق به هر سه اسکنر (market_scanner.py /
short_term_scanner.py / reversal_scanner.py) بدون تغییر منطق امتیازدهی
موجود — فقط قبل از مرحله رتبه‌بندی صدکی فراخوانی می‌شود.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("data_validation")


# ===========================================================================
# بخش ۱: راستی‌آزمایی متقاطع قیمت/حجم با Twelve Data
# ===========================================================================

TWELVEDATA_QUOTE_URL = "https://api.twelvedata.com/quote"

# پلن رایگان: ۸ درخواست در دقیقه - یعنی حداقل ۷.۵ ثانیه فاصله بین درخواست‌ها.
# کمی محافظه‌کارانه‌تر (۸ ثانیه) در نظر گرفته شده تا تاخیر شبکه هم پوشش داده شود.
TWELVEDATA_MIN_DELAY_SECONDS = 8.0


def _get_twelvedata_api_key() -> str:
    """
    کلید رایگان را از https://twelvedata.com/pricing (پلن Basic/رایگان)
    با ثبت‌نام ساده ایمیلی بگیرید - بدون کپچا، بلافاصله فعال است و از
    هر IP (از جمله GitHub Actions) کار می‌کند. در متغیر محیطی
    TWELVEDATA_API_KEY قرار دهید (مثلا به‌عنوان GitHub Secret).
    """
    return os.environ.get("TWELVEDATA_API_KEY", "")


def _looks_like_us_ticker(ticker: str) -> bool:
    """
    فقط نمادهای بدون پسوند صرافی (heuristic برای سهام آمریکایی مثل
    yfinance) را برای راستی‌آزمایی در نظر می‌گیریم - نمادهای اروپایی
    معمولا پسوندی مثل .DE/.PA/.AS دارند که نگاشت آن‌ها به Twelve Data
    نیاز به پارامتر exchange/mic_code جدا دارد و برای سادگی این‌جا رد
    می‌شوند (فقط باعث کاهش اندازه نمونه می‌شود، نه خطا).
    """
    return "." not in ticker


def fetch_twelvedata_last_close(
    ticker: str, session: Optional[requests.Session] = None, timeout: int = 8
) -> Optional[dict]:
    """آخرین قیمت پایانی/حجم را از Twelve Data می‌گیرد؛ در صورت شکست None برمی‌گرداند."""
    api_key = _get_twelvedata_api_key()
    if not api_key:
        logger.warning(
            "TWELVEDATA_API_KEY تنظیم نشده - راستی‌آزمایی قیمت انجام نمی‌شود. "
            "کلید رایگان را از https://twelvedata.com/pricing بگیرید."
        )
        return None

    if not _looks_like_us_ticker(ticker):
        return None

    sess = session or requests
    try:
        resp = sess.get(
            TWELVEDATA_QUOTE_URL,
            params={"symbol": ticker, "apikey": api_key},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("status") == "error":
            code = data.get("code")
            message = data.get("message", "")
            if code == 429 or "limit" in message.lower():
                logger.warning("سقف مجاز Twelve Data رد شده - راستی‌آزمایی امروز محدود می‌شود.")
            else:
                logger.warning("Twelve Data error برای %s: %s", ticker, message)
            return None

        close = data.get("close")
        volume = data.get("volume")
        if close is None:
            return None

        return {
            "date": data.get("datetime"),
            "close": float(close),
            "volume": float(volume) if volume is not None else np.nan,
        }
    except Exception as e:
        logger.warning("Twelve Data fetch failed for %s: %s", ticker, e)
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
    sample_frac: float = 0.08,
    min_sample: int = 10,
    max_sample: int = 40,
    price_tolerance: float = 0.03,
    volume_tolerance: float = 0.5,
    batch_delay: float = TWELVEDATA_MIN_DELAY_SECONDS,
    random_seed: Optional[int] = None,
) -> list[ValidationResult]:
    """
    یک نمونه تصادفی از سهام (نه همه) را با Twelve Data راستی‌آزمایی می‌کند.

    max_sample=40 عمدا اضافه شده: با محدودیت ۸ درخواست/دقیقه پلن رایگان،
    نمونه بزرگ‌تر باعث طولانی شدن غیرمنطقی اجرا می‌شود (۴۰ نمونه ≈ ۵.۵
    دقیقه با تاخیر ۸ ثانیه‌ای). اگر نیاز به نمونه بزرگ‌تر دارید، یا پلن
    پولی Twelve Data را در نظر بگیرید یا batch_delay را کاهش دهید (با
    ریسک رد شدن سقف دقیقه‌ای).

    price_tolerance=0.03 عمدا سخت‌گیر نیست، چون دو منبع مختلف گاهی چند
    دقیقه اختلاف تاخیر گزارش دارند و این طبیعی است، نه خطا.

    فقط نمادهای بدون پسوند صرافی (heuristic برای سهام آمریکایی) واقعا
    چک می‌شوند؛ بقیه به‌عنوان 'reference_unavailable' علامت می‌خورند.
    """
    n = max(min_sample, int(len(df) * sample_frac))
    n = min(n, max_sample, len(df))
    sample = df.sample(n=n, random_state=random_seed)

    results: list[ValidationResult] = []
    session = requests.Session()

    for i, (_, row) in enumerate(sample.iterrows()):
        ticker = row[ticker_col]
        ref = fetch_twelvedata_last_close(ticker, session=session)

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

    اگر همه نمونه‌ها 'unavailable' باشند (مثلا چون TWELVEDATA_API_KEY تنظیم
    نشده یا سقف روزانه/دقیقه‌ای رد شده)، این به‌عنوان یک هشدار جدا مشخص
    می‌شود - چون flag_rate=0.0 در آن حالت گمراه‌کننده است (به این معنی
    نیست که «همه چیز تایید شد»، بلکه اصلا چیزی چک نشده است).
    """
    total = len(results)
    flagged = sum(r.flagged for r in results)
    unavailable = sum(r.reason == "reference_unavailable" for r in results)
    checked = total - unavailable
    flag_rate = (flagged / checked) if checked else 0.0
    fully_unavailable = total > 0 and checked == 0
    if fully_unavailable:
        logger.warning(
            "راستی‌آزمایی Twelve Data برای هیچ نمونه‌ای انجام نشد (%d/%d نامعتبر) - "
            "احتمالا TWELVEDATA_API_KEY تنظیم نشده یا سقف رد شده است.",
            unavailable, total,
        )
    return {
        "total_checked": checked,
        "unavailable": unavailable,
        "flagged": flagged,
        "flag_rate": round(flag_rate, 4),
        "systemic_alert": flag_rate > 0.20 and checked >= 5,
        "fully_unavailable": fully_unavailable,
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

# --- بعد از جمع‌آوری قیمت/حجم، یک نمونه را با Twelve Data چک کنید ---
results = cross_validate_prices(df, sample_frac=0.15, batch_delay=1.0)
summary = validation_summary(results)
logger.info("Cross-validation summary: %s", summary)

if summary["systemic_alert"]:
    # پیشنهاد: اجرای امروز را با یک برچسب هشدار در داشبورد منتشر کنید،
    # نه این‌که کل اجرا را متوقف کنید (که با ماهیت اسکن روزانه ناسازگار است)
    logger.warning("نرخ پرچم‌خوردن غیرعادی بالاست - داده امروز را با احتیاط منتشر کنید.")
"""
