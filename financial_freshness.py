"""
financial_freshness.py
ماژول چک تازگی داده صورت‌های مالی (TTM).

مشکل: بسیاری از نسبت‌های ارزندگی که yfinance می‌دهد (P/E، EV/EBITDA و...)
بر پایه TTM (دوازده ماه گذشته/trailing twelve months) هستند، نه لزوما
آخرین گزارش فصلی. برای اکثر شرکت‌ها این مشکلی ایجاد نمی‌کند، ولی در دو
حالت می‌تواند گمراه‌کننده باشد:

  1. شرکت به‌تازگی یک تغییر ساختاری بزرگ داشته (فروش یک بخش، ادغام بزرگ،
     spin-off) - در این حالت TTM ترکیبی از «قبل» و «بعد» تغییر است و
     هیچ‌کدام از دو واقعیت را درست نشان نمی‌دهد.

  2. گزارش فصلی جدید هنوز توسط بازار/دیتافید منعکس نشده - یعنی عملا
     دارید با داده‌ای که چند ماه قدیمی است تصمیم می‌گیرید بدون اینکه
     متوجه باشید.

راه‌حل این ماژول: به‌جای تلاش برای «تشخیص» تغییر ساختاری (که نیاز به
داده اضافی دارد که معمولا در دسترس رایگان نیست)، صرفا **سن داده مالی**
را اندازه می‌گیرد و به کاربر/داشبورد نشان می‌دهد - شفافیت به‌جای حدس.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("financial_freshness")


# ===========================================================================
# آستانه‌ها
# ===========================================================================

# اکثر شرکت‌ها هر ۹۰-۱۰۰ روز گزارش می‌دهند؛ اگر بیش از این از آخرین
# گزارش گذشته، یعنی یا گزارش بعدی دیر منتشر شده یا دیتافید به‌روز نیست
FRESH_DAYS = 100
STALE_DAYS = 200          # بین ۱۰۰ تا ۲۰۰ روز: هشدار ملایم
VERY_STALE_DAYS = 200     # بیش از ۲۰۰ روز: هشدار جدی - گزارش فصلی جا افتاده

# اگر قیمت سهم از زمان آخرین گزارش مالی بیش از این درصد حرکت کرده،
# نسبت‌های ارزندگی (که از EPS/EBITDA گزارش‌شده استفاده می‌کنند) دیگر
# لزوما تصویر واقعی امروز نیستند، چون صورت مخرج (قیمت) تغییر کرده ولی
# صورت (سود/جریان نقدی) هنوز مال گزارش قدیمی است
PRICE_DRIFT_WARNING = 0.30


@dataclass
class FinancialFreshness:
    ticker: str
    days_since_report: Optional[int]
    freshness_label: str   # fresh / aging / stale / unknown
    price_drift_pct: Optional[float]
    price_drift_warning: bool
    reason: str = ""


def classify_freshness(days_since_report: Optional[int]) -> str:
    """طبقه‌بندی ساده بر اساس تعداد روز از آخرین گزارش مالی."""
    if days_since_report is None or (isinstance(days_since_report, float) and np.isnan(days_since_report)):
        return "unknown"
    if days_since_report <= FRESH_DAYS:
        return "fresh"
    if days_since_report <= STALE_DAYS:
        return "aging"
    return "stale"


def days_since_last_report(last_fiscal_date: Optional[str], today: Optional[date] = None) -> Optional[int]:
    """
    last_fiscal_date: تاریخ آخرین گزارش مالی به فرمت YYYY-MM-DD (از فیلدهایی
    مثل mostRecentQuarter یا lastFiscalYearEnd در yfinance گرفته می‌شود).
    اگر این فیلد در داده yfinance موجود نباشد (که گاهی برای برخی نمادها
    اتفاق می‌افتد)، None برگردانید تا freshness_label برابر 'unknown' شود -
    نه این‌که به‌اشتباه 'fresh' فرض شود.
    """
    if not last_fiscal_date:
        return None
    try:
        report_date = datetime.strptime(last_fiscal_date[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    today = today or date.today()
    return (today - report_date).days


def check_financial_freshness(
    ticker: str,
    last_fiscal_date: Optional[str],
    current_price: Optional[float] = None,
    price_at_last_report: Optional[float] = None,
    today: Optional[date] = None,
) -> FinancialFreshness:
    """
    بررسی کامل تازگی داده مالی یک سهم. price_at_last_report اختیاری است -
    اگر ندارید (که معمولا ندارید مگر تاریخچه قیمت را جدا نگه دارید)، فقط
    سن گزارش چک می‌شود و price_drift محاسبه نمی‌شود.
    """
    days = days_since_last_report(last_fiscal_date, today)
    label = classify_freshness(days)

    price_drift_pct = None
    price_drift_warning = False
    if current_price and price_at_last_report and price_at_last_report > 0:
        price_drift_pct = round(abs(current_price - price_at_last_report) / price_at_last_report, 4)
        price_drift_warning = price_drift_pct > PRICE_DRIFT_WARNING

    reason_parts = []
    if label == "stale":
        reason_parts.append(f"{days} روز از آخرین گزارش مالی گذشته")
    elif label == "unknown":
        reason_parts.append("تاریخ آخرین گزارش مالی در دسترس نیست")
    if price_drift_warning:
        reason_parts.append(f"قیمت {price_drift_pct:.0%} از زمان آخرین گزارش حرکت کرده")

    return FinancialFreshness(
        ticker=ticker,
        days_since_report=days,
        freshness_label=label,
        price_drift_pct=price_drift_pct,
        price_drift_warning=price_drift_warning,
        reason="؛ ".join(reason_parts),
    )


def apply_freshness_check(df, ticker_col: str = "ticker", fiscal_date_col: str = "last_fiscal_date",
                           price_col: str = "price"):
    """
    برای کل دیتافریم چک تازگی را اجرا می‌کند و ستون‌های
    'financial_freshness' و 'financial_freshness_reason' اضافه می‌کند.
    این فیلتر نیست - فقط شفافیت اضافه می‌کند؛ تصمیم درباره این‌که آیا به
    یک سهم با داده stale اعتماد کنید یا نه، دست کاربر می‌ماند (می‌توانید
    این ستون را در داشبورد HTML به‌صورت یک برچسب رنگی نمایش دهید).
    """
    df = df.copy()
    labels, reasons = [], []
    for _, row in df.iterrows():
        result = check_financial_freshness(
            ticker=row[ticker_col],
            last_fiscal_date=row.get(fiscal_date_col),
            current_price=row.get(price_col),
        )
        labels.append(result.freshness_label)
        reasons.append(result.reason)
    df["financial_freshness"] = labels
    df["financial_freshness_reason"] = reasons

    n_stale = sum(l == "stale" for l in labels)
    n_unknown = sum(l == "unknown" for l in labels)
    if n_stale:
        logger.info("%d سهم دارای داده مالی قدیمی (stale) شناسایی شدند.", n_stale)
    if n_unknown:
        logger.info("%d سهم بدون تاریخ گزارش مالی مشخص (unknown) هستند.", n_unknown)

    return df


# ===========================================================================
# نمونه استفاده در market_scanner.py (جایی که نسبت‌های ارزندگی محاسبه می‌شوند)
# ===========================================================================
"""
from financial_freshness import apply_freshness_check

# yfinance فیلد 'mostRecentQuarter' را به‌صورت timestamp یونیکس می‌دهد؛
# قبل از استفاده به رشته تاریخ تبدیل کنید:
# df["last_fiscal_date"] = pd.to_datetime(df["most_recent_quarter"], unit="s").dt.strftime("%Y-%m-%d")

df = apply_freshness_check(df, ticker_col="ticker", fiscal_date_col="last_fiscal_date", price_col="price")

# در داشبورد HTML، برای سهامی که financial_freshness == 'stale' یا 'unknown'
# است، یک آیکون هشدار کنار امتیاز ارزندگی نمایش دهید - نه این‌که آن‌ها را
# از لیست top-N حذف کنید (چون ممکن است هنوز بهترین گزینه موجود باشند،
# فقط باید با آگاهی بیشتری بررسی شوند).
stale_tickers = df.loc[df["financial_freshness"].isin(["stale", "unknown"]), "ticker"].tolist()
"""
