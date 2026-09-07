"""
analyst_signal.py
ماژول اصلاح معیار «نظر تحلیل‌گران» در اسکنر بلندمدت (market_scanner.py).

مشکل: قیمت هدف تحلیل‌گران معمولا *بعد از* حرکت قیمت به‌روزرسانی می‌شود، نه
قبل از آن (تحلیل‌گران به‌طور میانگین روند را دنبال می‌کنند، پیش‌بینی نمی‌کنند -
این یک یافته تکرارشده در ادبیات مالی است). این یعنی این معیار می‌تواند
به‌جای «سیگنال آینده‌نگر»، صرفا «تاییدیه حرکت گذشته» باشد و وزن‌دهی به آن
باید محتاطانه باشد.

README از قبل وزن این معیار را به ۱۵٪ محدود کرده که تصمیم درستی است.
این ماژول دو بهبود مکمل اضافه می‌کند، بدون نیاز به تغییر وزن‌ها:

  1. پراکندگی قیمت هدف (اختلاف‌نظر تحلیل‌گران) را به‌عنوان سیگنال
     «عدم قطعیت» اندازه می‌گیرد - اگر تحلیل‌گران خیلی با هم اختلاف دارند،
     اعتماد کمتری به میانگین/میانه قیمت هدف باید داد.

  2. تازگی توصیه‌ها را چک می‌کند - یک قیمت هدف سه‌ماه‌پیش که از آن زمان
     قیمت سهم ۴۰٪ حرکت کرده، دیگر معنای اولیه‌اش را ندارد.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("analyst_signal")


# ===========================================================================
# بخش ۱: پراکندگی قیمت هدف به‌عنوان سیگنال عدم قطعیت
# ===========================================================================

@dataclass
class AnalystConfidence:
    ticker: str
    mean_target: float
    dispersion_pct: float      # (high - low) / mean، به درصد
    confidence_multiplier: float  # بین ۰ تا ۱ - در امتیاز نهایی این معیار ضرب شود
    n_analysts: int
    reason: str = ""


# آستانه‌ها بر اساس این منطق: پراکندگی زیر ۱۵٪ یعنی اجماع نسبی، بالای
# ۵۰٪ یعنی عملا تحلیل‌گران هیچ توافقی ندارند و میانگین بی‌معنی می‌شود.
DISPERSION_LOW = 0.15
DISPERSION_HIGH = 0.50
MIN_ANALYSTS_FOR_CONFIDENCE = 3  # با کمتر از این تعداد، میانگین آماری قابل اتکا نیست


def compute_analyst_confidence(
    ticker: str,
    target_mean: float,
    target_low: float,
    target_high: float,
    n_analysts: int,
) -> AnalystConfidence:
    """
    یک ضریب اطمینان بین ۰ و ۱ برمی‌گرداند که باید در امتیاز خام معیار
    «نظر تحلیل‌گران» ضرب شود (نه این‌که جایگزین وزن ۱۵٪ شود - مکمل آن است).

    منطق: هرچه پراکندگی قیمت هدف بیشتر و تعداد تحلیل‌گران کمتر، ضریب
    اطمینان به سمت صفر می‌رود (یعنی این معیار عملا خنثی/کم‌اثر می‌شود
    برای آن سهم خاص، بدون این‌که کل فرمول بهم بریزد).
    """
    if n_analysts <= 0 or target_mean is None or np.isnan(target_mean) or target_mean == 0:
        return AnalystConfidence(
            ticker=ticker, mean_target=target_mean or float("nan"),
            dispersion_pct=float("nan"), confidence_multiplier=0.0,
            n_analysts=n_analysts, reason="insufficient_data",
        )

    dispersion_pct = abs(target_high - target_low) / abs(target_mean)

    # ضریب پراکندگی: ۱.۰ زیر آستانه پایین، ۰.۰ بالای آستانه بالا، خطی در میانه
    if dispersion_pct <= DISPERSION_LOW:
        dispersion_factor = 1.0
    elif dispersion_pct >= DISPERSION_HIGH:
        dispersion_factor = 0.0
    else:
        dispersion_factor = 1.0 - (dispersion_pct - DISPERSION_LOW) / (
            DISPERSION_HIGH - DISPERSION_LOW
        )

    # ضریب تعداد تحلیل‌گران: زیر حداقل، به‌طور خطی جریمه می‌شود
    coverage_factor = min(1.0, n_analysts / MIN_ANALYSTS_FOR_CONFIDENCE)

    confidence_multiplier = round(dispersion_factor * coverage_factor, 4)

    reason = ""
    if n_analysts < MIN_ANALYSTS_FOR_CONFIDENCE:
        reason = f"پوشش کم ({n_analysts} تحلیل‌گر)"
    elif dispersion_pct >= DISPERSION_HIGH:
        reason = f"پراکندگی بسیار بالا ({dispersion_pct:.0%})"

    return AnalystConfidence(
        ticker=ticker, mean_target=target_mean, dispersion_pct=round(dispersion_pct, 4),
        confidence_multiplier=confidence_multiplier, n_analysts=n_analysts, reason=reason,
    )


# ===========================================================================
# بخش ۲: تازگی توصیه نسبت به حرکت قیمت
# ===========================================================================

@dataclass
class RecommendationFreshness:
    ticker: str
    price_move_since_target_pct: float
    is_stale: bool
    reason: str = ""


# اگر قیمت سهم از زمان آخرین به‌روزرسانی قابل‌مشاهده بیش از این درصد حرکت
# کرده باشد، فرض می‌کنیم قیمت هدف دیگر بازتاب‌دهنده واقعیت فعلی نیست -
# حتی اگر تاریخ دقیق آخرین توصیه در دسترس نباشد (yfinance این را همیشه
# نمی‌دهد)، این یک پروکسی عملی است.
STALE_MOVE_THRESHOLD = 0.25


def check_recommendation_freshness(
    ticker: str, current_price: float, target_mean: float
) -> RecommendationFreshness:
    """
    یک پروکسی ساده برای تازگی: اگر قیمت فعلی خیلی از میانگین قیمت هدف
    فاصله گرفته (در هر جهت)، این نشانه غیرمستقیم این است که یا قیمت هدف
    قدیمی است، یا یک اتفاق قیمتی بزرگ (شوک خبری) رخ داده که تحلیل‌گران
    هنوز به آن واکنش نشان نداده‌اند - در هر دو حالت، به این معیار در حال
    حاضر نباید اعتماد کامل کرد.
    """
    if not current_price or not target_mean or np.isnan(current_price) or np.isnan(target_mean):
        return RecommendationFreshness(
            ticker=ticker, price_move_since_target_pct=float("nan"),
            is_stale=True, reason="missing_data",
        )

    move_pct = abs(current_price - target_mean) / target_mean
    is_stale = move_pct > STALE_MOVE_THRESHOLD

    return RecommendationFreshness(
        ticker=ticker,
        price_move_since_target_pct=round(move_pct, 4),
        is_stale=is_stale,
        reason=f"فاصله قیمت/هدف {move_pct:.0%} - احتمال شوک اخیر یا هدف قدیمی" if is_stale else "",
    )


# ===========================================================================
# ترکیب هر دو سیگنال در یک ضریب نهایی
# ===========================================================================

def adjusted_analyst_score(
    raw_analyst_score: float,
    ticker: str,
    current_price: float,
    target_mean: float,
    target_low: float,
    target_high: float,
    n_analysts: int,
) -> tuple[float, dict]:
    """
    امتیاز خام معیار «نظر تحلیل‌گران» (که در فرمول اصلی با وزن ۱۵٪ ضرب
    می‌شود) را بر اساس اطمینان و تازگی تعدیل می‌کند.

    خروجی: (امتیاز تعدیل‌شده, دیکشنری جزئیات برای لاگ/دیباگ)
    این طراحی عمدا non-invasive است - فقط raw_analyst_score را در یک عدد
    بین ۰ و ۱ ضرب می‌کند، بدون تغییر در باقی فرمول امتیازدهی.
    """
    confidence = compute_analyst_confidence(ticker, target_mean, target_low, target_high, n_analysts)
    freshness = check_recommendation_freshness(ticker, current_price, target_mean)

    staleness_penalty = 0.5 if freshness.is_stale else 1.0  # جریمه نصف‌کننده، نه صفرکننده
    final_multiplier = confidence.confidence_multiplier * staleness_penalty

    adjusted = raw_analyst_score * final_multiplier

    details = {
        "dispersion_pct": confidence.dispersion_pct,
        "confidence_multiplier": confidence.confidence_multiplier,
        "n_analysts": n_analysts,
        "price_move_since_target_pct": freshness.price_move_since_target_pct,
        "is_stale": freshness.is_stale,
        "final_multiplier": round(final_multiplier, 4),
    }

    if final_multiplier < 0.3:
        logger.info(
            "%s: معیار تحلیل‌گران عملا خنثی شد (ضریب=%.2f) - %s %s",
            ticker, final_multiplier, confidence.reason, freshness.reason,
        )

    return adjusted, details


# ===========================================================================
# نمونه استفاده در market_scanner.py
# ===========================================================================
"""
from analyst_signal import adjusted_analyst_score

# به‌جای استفاده مستقیم از raw_analyst_score در فرمول وزن‌دهی نهایی:
adjusted_score, details = adjusted_analyst_score(
    raw_analyst_score=row["analyst_score"],   # امتیاز خام موجود شما (۰ تا ۱۰۰ یا هرچه)
    ticker=row["ticker"],
    current_price=row["price"],
    target_mean=row["target_mean_price"],
    target_low=row["target_low_price"],
    target_high=row["target_high_price"],
    n_analysts=row["number_of_analyst_opinions"],
)
df.at[idx, "analyst_score"] = adjusted_score
df.at[idx, "analyst_confidence_detail"] = str(details)  # برای دیباگ/شفافیت در داشبورد

# سپس وزن ۱۵٪ موجود در README دقیقا مثل قبل روی analyst_score اعمال می‌شود -
# فقط حالا این امتیاز خودش منعکس‌کننده میزان اطمینان به آن است.
"""
