"""
sector_overrides.py
ماژول اصلاح طبقه‌بندی صنعتی برای اسکنرهای بازار سهام.

مشکل: معیارهایی مثل معافیت بانک‌ها/بیمه/یوتیلیتی از فیلتر بدهی (که در
README مستند شده) به برچسب sector/industry گرفته‌شده از yfinance وابسته‌اند.
این برچسب‌ها گاهی:
  - خیلی کلی‌اند (یک هلدینگ صنعتی-مالی زیر یک sector نامربوط قرار می‌گیرد)
  - اشتباه‌اند (به‌خصوص برای شرکت‌های دوگانه‌کاره یا تازه‌طبقه‌بندی‌شده)
  - بین GICS و طبقه‌بندی داخلی یاهو ناسازگارند

راه‌حل مقیاس‌پذیر: یک نگاشت override دستی برای مواردی که شناسایی شده‌اند،
دقیقا با همان الگویی که README برای نگه‌داری دستی لیست یورواستوکس ۵۰
استفاده کرده - چون یک دیتابیس طبقه‌بندی کامل و رایگان وجود ندارد که
بشود صرفا به آن اعتماد کرد.

این ماژول جایگزین برچسب yfinance نمی‌شود؛ فقط برای موارد شناخته‌شده‌ی
اشتباه، override اعمال می‌کند و برای بقیه از برچسب اصلی استفاده می‌کند.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("sector_overrides")


# ===========================================================================
# دسته‌های صنعتی که در فرمول امتیازدهی رفتار خاص دارند
# (مطابق README: "بانک‌ها/بیمه/یوتیلیتی از فیلتر بدهی معاف‌اند")
# ===========================================================================

HIGH_LEVERAGE_SECTORS = {"Financial Services", "Utilities", "Financials", "Banking", "Insurance"}


# ===========================================================================
# نگاشت override دستی - فقط مواردی که واقعا بررسی و تایید شده‌اند اضافه شوند.
# هر رکورد باید یک دلیل مستند داشته باشد تا در آینده قابل ممیزی باشد.
# ===========================================================================

@dataclass
class SectorOverride:
    ticker: str
    correct_sector: str
    yfinance_reported_sector: str
    reason: str
    verified_date: str  # YYYY-MM-DD - تاریخی که این override بررسی/تایید شد


# نمونه اولیه - این لیست باید به‌مرور که موارد اشتباه پیدا می‌شوند تکمیل شود.
# ساختار عمدا شفاف نگه داشته شده تا افزودن مورد جدید ساده باشد.
SECTOR_OVERRIDES: dict[str, SectorOverride] = {
    "BRK-B": SectorOverride(
        ticker="BRK-B",
        correct_sector="Financial Services",
        yfinance_reported_sector="Financial Services",  # نمونه: در حال حاضر درست گزارش می‌شود
        reason=(
            "هلدینگ برکشایر هاتاوی مالکیت کسب‌وکارهای غیرمالی زیادی هم دارد "
            "(بیمه، راه‌آهن، انرژی، تولید) - برچسب واحد sector می‌تواند گمراه‌کننده باشد. "
            "برای فیلتر بدهی باید به‌عنوان مالی/بیمه در نظر گرفته شود."
        ),
        verified_date="2026-01-15",
    ),
    # مثال ساختگی برای نشان دادن الگو - این را با موارد واقعی که در طول
    # اجرای روزانه پیدا می‌کنید جایگزین/تکمیل کنید:
    # "XYZ": SectorOverride(
    #     ticker="XYZ",
    #     correct_sector="Utilities",
    #     yfinance_reported_sector="Industrials",
    #     reason="یک شرکت آب/برق منطقه‌ای که یاهو به‌اشتباه Industrials برچسب زده",
    #     verified_date="2026-03-10",
    # ),
}


def get_effective_sector(ticker: str, yfinance_sector: str | None) -> str:
    """
    برچسب صنعتی «مؤثر» را برمی‌گرداند: اگر override تایید‌شده وجود دارد
    از آن استفاده کن، وگرنه برچسب yfinance را برگردان.

    اگر override موجود باشد ولی sector گزارش‌شده توسط yfinance از زمان
    ثبت override تغییر کرده باشد (یعنی یاهو خودش را اصلاح کرده)، یک هشدار
    می‌دهد تا override منسوخ حذف شود.
    """
    override = SECTOR_OVERRIDES.get(ticker)
    if override is None:
        return yfinance_sector or "Unknown"

    if yfinance_sector and yfinance_sector != override.yfinance_reported_sector:
        logger.warning(
            "Override برای %s ممکن است منسوخ باشد: yfinance اکنون '%s' گزارش می‌دهد "
            "ولی override بر اساس '%s' ثبت شده بود (تاریخ ثبت: %s). لطفا override را "
            "بازبینی کنید.",
            ticker, yfinance_sector, override.yfinance_reported_sector, override.verified_date,
        )

    return override.correct_sector


def is_high_leverage_sector(ticker: str, yfinance_sector: str | None) -> bool:
    """
    آیا این سهم باید از فیلتر بدهی معاف باشد؟ از برچسب مؤثر (override-aware)
    استفاده می‌کند، نه مستقیم برچسب خام yfinance.
    """
    effective = get_effective_sector(ticker, yfinance_sector)
    return effective in HIGH_LEVERAGE_SECTORS


def apply_sector_overrides(df, ticker_col: str = "ticker", sector_col: str = "sector"):
    """
    یک ستون جدید 'effective_sector' به دیتافریم اضافه می‌کند که برای فیلتر
    بدهی و سایر منطق‌های وابسته به صنعت باید به‌جای ستون خام استفاده شود.
    ستون اصلی sector دست‌نخورده باقی می‌ماند (برای نمایش/دیباگ).
    """
    df = df.copy()
    df["effective_sector"] = [
        get_effective_sector(t, s) for t, s in zip(df[ticker_col], df[sector_col])
    ]
    df["is_high_leverage_sector"] = [
        is_high_leverage_sector(t, s) for t, s in zip(df[ticker_col], df[sector_col])
    ]
    return df


# ===========================================================================
# ابزار کمکی برای پیدا کردن کاندیدهای override جدید:
# شرکت‌هایی که نسبت بدهی/سرمایه‌شان به‌طور مشکوکی بالاست ولی در دسته
# پرمعافیت قرار نگرفته‌اند - این‌ها کاندید بررسی دستی‌اند، نه لزوما خطا.
# ===========================================================================

def flag_high_debt_non_exempt(df, sector_col: str = "effective_sector",
                               debt_col: str = "debt_to_equity",
                               threshold: float = 300.0):
    """
    سهامی که D/E بالای آستانه دارند ولی در HIGH_LEVERAGE_SECTORS نیستند را
    برمی‌گرداند. این‌ها یا واقعا شرکت‌های پراهرم غیرمالی‌اند (باید در فیلتر
    اصلی جریمه شوند - رفتار درست است)، یا برچسب صنعتی‌شان اشتباه است
    (کاندید افزودن به SECTOR_OVERRIDES). تشخیص این دو حالت نیاز به بررسی
    دستی دارد؛ این تابع فقط لیست کاندیدها را برای بررسی می‌دهد.
    """
    if sector_col not in df.columns or debt_col not in df.columns:
        return df.iloc[0:0]
    mask = (df[debt_col] > threshold) & (~df["is_high_leverage_sector"])
    return df.loc[mask, ["ticker", sector_col, debt_col]].sort_values(debt_col, ascending=False)


# ===========================================================================
# نمونه استفاده (جایگزین منطق فعلی معافیت بدهی در هر سه اسکنر)
# ===========================================================================
"""
from sector_overrides import apply_sector_overrides, flag_high_debt_non_exempt

df = apply_sector_overrides(df, ticker_col="ticker", sector_col="sector")

# در فرمول امتیازدهی، به‌جای:
#   if row["sector"] in ("Financial Services", "Utilities"): skip_debt_filter
# از این استفاده کنید:
#   if row["is_high_leverage_sector"]: skip_debt_filter

# هر چند وقت یک‌بار (مثلا ماهانه) این را اجرا کنید تا کاندیدهای override
# جدید را پیدا کنید:
candidates = flag_high_debt_non_exempt(df, threshold=300.0)
if not candidates.empty:
    logger.info("کاندیدهای بررسی دستی برای override صنعتی:\\n%s", candidates.to_string())
"""
