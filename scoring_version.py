"""
scoring_version.py
محاسبه یک شناسه کوتاه از وزن‌های امتیازدهی فعلی هر اسکنر.

هدف: در picks_history.jsonl ثبت شود که هر پیشنهاد با کدام «نسخه» از
فرمول امتیازدهی صادر شده. اگر بعداً وزن‌ها را تغییر دهید (کاملاً طبیعی و
مورد انتظار در طول زمان)، performance_scorecard.py می‌تواند پیشنهادهای
نسخه‌های مختلف را جدا نگه دارد - وگرنه همبستگی امتیاز-بازده یک عدد
بی‌معنی می‌شود، چون دارید دو فرمول متفاوت را به‌عنوان یکی تحلیل می‌کنید.

⚠️ این یک هش امنیتی نیست، فقط برای *تشخیص تغییر* در مقادیر وزن است.
"""

from __future__ import annotations

import hashlib


def compute_version(weights: dict) -> str:
    """
    weights: دیکشنری نام وزن -> مقدار، مثلا
    {"growth_value": 0.35, "quality": 0.30, "technical": 0.20, "analyst": 0.15}

    یک شناسه ۸ کاراکتری برمی‌گرداند که هر تغییر در مقادیر (حتی جزئی، مثلاً
    ۰.۳۵ به ۰.۳۰) آن را عوض می‌کند. کلیدها قبل از هش مرتب می‌شوند تا ترتیب
    تعریف در کد اثری روی نتیجه نداشته باشد.
    """
    canonical = ",".join(f"{k}={v}" for k, v in sorted(weights.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


# ===========================================================================
# نمونه استفاده (نزدیک تعریف وزن‌ها در هر اسکنر)
# ===========================================================================
"""
from scoring_version import compute_version

WEIGHT_GROWTH_VALUE = 0.35
WEIGHT_QUALITY = 0.30
WEIGHT_TECHNICAL = 0.20
WEIGHT_ANALYST = 0.15

SCORING_VERSION = compute_version({
    "growth_value": WEIGHT_GROWTH_VALUE,
    "quality": WEIGHT_QUALITY,
    "technical": WEIGHT_TECHNICAL,
    "analyst": WEIGHT_ANALYST,
})

# سپس هنگام ثبت پیشنهاد:
log_daily_picks(picks, scanner_name="market_scanner", scoring_version=SCORING_VERSION, ...)
"""
