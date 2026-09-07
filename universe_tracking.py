"""
universe_tracking.py
ماژول ردیابی یونیورس برای کاهش اثر سوگیری بقا (Survivorship Bias).

نکته مهم درباره این مشکل در این پروژه به‌طور خاص:
چون این یک اسکنر *روزانه* است (نه بک‌تست تاریخی)، نمی‌توان و لازم نیست
اعضای گذشته شاخص را در محاسبات "امروز" وارد کرد — رتبه‌بندی امروز باید
روی اعضای واقعی امروز باشد. مشکل واقعی جای دیگری است:

  1. اگر اسکریپ لیست S&P 500 از ویکی‌پدیا به‌خاطر تغییر ساختار صفحه خراب
     شود (مثلا فقط ۵۰ نماد به‌جای ۵۰۰ برگرداند)، این می‌تواند بی‌سروصدا رخ
     دهد و کسی متوجه نشود — هیچ خطایی پرتاب نمی‌شود، فقط یونیورس کوچک‌تر
     می‌شود و همه به نظر می‌رسد کار می‌کند.

  2. وقتی یک شرکت از شاخص خارج می‌شود (ادغام، ورشکستگی، حذف)، اگر این
     تغییر لاگ نشود، در آینده که کسی بخواهد تاریخچه اسکن‌ها را برای
     ارزیابی عملکرد واقعی استراتژی تحلیل کند (نه فقط استفاده روزانه)،
     یونیورس‌های گذشته قابل بازسازی نخواهند بود و آن تحلیل به‌طور ذاتی
     سوگیری بقا خواهد داشت. راه‌حل: از امروز به بعد یک رکورد point-in-time
     نگه دارید، حتی اگر گذشته قابل بازسازی نیست.

  3. تغییرات غیرعادی زیاد در یک روز (churn بالا) معمولا نشانه یک باگ در
     اسکریپینگ است، نه یک روز واقعی با اتفاقات زیاد شاخصی.

این ماژول یک تاریخچه append-only از عضویت یونیورس نگه می‌دارد و هر روز
تغییرات را با سه دسته چک اعتبار مقایسه می‌کند.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger("universe_tracking")


# ===========================================================================
# محدوده‌های منطقی مورد انتظار برای اندازه یونیورس -
# اگر اسکرِیپینگ خراب شود، این اولین علامت است
# ===========================================================================

EXPECTED_UNIVERSE_SIZE = {
    "sp500": (480, 520),          # S&P 500 گاهی به‌خاطر dual-class کمی بیشتر از ۵۰۰ ردیف دارد
    "nasdaq100": (95, 105),
    "eurostoxx50": (48, 52),
    "sp500_eurostoxx50": (528, 570),  # ترکیب دو یونیورس بالا
}

MAX_REASONABLE_DAILY_CHURN = 5     # بیش از این تعداد تغییر عضویت در یک روز، مشکوک است
EUROSTOXX_STALENESS_DAYS = 400     # طبق توصیه خود README: باید سالی یک‌بار به‌روز شود


@dataclass
class UniverseSnapshot:
    scan_date: str          # YYYY-MM-DD
    universe_name: str      # "sp500" یا "eurostoxx50"
    tickers: list[str]


@dataclass
class UniverseDiff:
    universe_name: str
    scan_date: str
    added: list[str]
    removed: list[str]
    size_before: int
    size_after: int
    size_alert: bool
    size_alert_reason: str
    churn_alert: bool


# ===========================================================================
# ذخیره و بازیابی تاریخچه (append-only JSONL - ساده، بدون نیاز به دیتابیس)
# ===========================================================================

def save_snapshot(
    tickers: list[str],
    universe_name: str,
    history_path: str = "universe_history.jsonl",
    scan_date: str | None = None,
) -> None:
    """یک رکورد جدید به تاریخچه اضافه می‌کند (هرگز رکوردهای قبلی را تغییر نمی‌دهد)."""
    snapshot = UniverseSnapshot(
        scan_date=scan_date or date.today().isoformat(),
        universe_name=universe_name,
        tickers=sorted(tickers),
    )
    path = Path(history_path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(snapshot), ensure_ascii=False) + "\n")


def load_last_snapshot(
    universe_name: str, history_path: str = "universe_history.jsonl"
) -> UniverseSnapshot | None:
    """آخرین رکورد ذخیره‌شده برای یک یونیورس مشخص را برمی‌گرداند (اگر وجود داشته باشد)."""
    path = Path(history_path)
    if not path.exists():
        return None

    last: UniverseSnapshot | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("universe_name") == universe_name:
                last = UniverseSnapshot(**rec)
    return last


# ===========================================================================
# مقایسه روز به روز + چک‌های اعتبار
# ===========================================================================

def check_universe_size(universe_name: str, size: int) -> tuple[bool, str]:
    """آیا اندازه یونیورس در بازه منطقی مورد انتظار است؟"""
    bounds = EXPECTED_UNIVERSE_SIZE.get(universe_name)
    if bounds is None:
        return False, ""
    low, high = bounds
    if size < low or size > high:
        return True, f"اندازه یونیورس ({size}) خارج از بازه مورد انتظار [{low},{high}] است"
    return False, ""


def diff_and_log_universe(
    tickers: list[str],
    universe_name: str,
    history_path: str = "universe_history.jsonl",
    scan_date: str | None = None,
    max_reasonable_churn: int = MAX_REASONABLE_DAILY_CHURN,
) -> UniverseDiff:
    """
    یونیورس امروز را با آخرین رکورد ذخیره‌شده مقایسه می‌کند، تغییرات را
    لاگ می‌کند (برای ساختن یک تاریخچه point-in-time به‌مرور زمان)، سپس
    خود رکورد امروز را ذخیره می‌کند.

    این تابع باید *قبل از* اجرای اسکن اصلی صدا زده شود، تا اگر churn_alert
    یا size_alert فعال شد، بتوان تصمیم گرفت که آیا اسکن با این یونیورس
    ادامه یابد یا با آخرین یونیورس معتبر جایگزین شود.
    """
    scan_date = scan_date or date.today().isoformat()
    current = set(tickers)

    prev_snapshot = load_last_snapshot(universe_name, history_path)
    prev = set(prev_snapshot.tickers) if prev_snapshot else set()

    added = sorted(current - prev)
    removed = sorted(prev - current)

    size_alert, size_reason = check_universe_size(universe_name, len(current))
    churn = len(added) + len(removed)
    churn_alert = bool(prev) and churn > max_reasonable_churn

    if removed:
        logger.info(
            "%s: %d نماد از یونیورس خارج شدند (تاریخ %s): %s",
            universe_name, len(removed), scan_date, ", ".join(removed),
        )
    if added:
        logger.info(
            "%s: %d نماد جدید به یونیورس اضافه شدند (تاریخ %s): %s",
            universe_name, len(added), scan_date, ", ".join(added),
        )
    if size_alert:
        logger.warning("هشدار اندازه یونیورس [%s]: %s", universe_name, size_reason)
    if churn_alert:
        logger.warning(
            "هشدار churn غیرعادی [%s]: %d تغییر در یک روز (آستانه: %d) - "
            "احتمالا مشکل اسکریپینگ لیست است، نه یک روز واقعی با تغییرات شاخصی.",
            universe_name, churn, max_reasonable_churn,
        )

    save_snapshot(tickers, universe_name, history_path, scan_date)

    return UniverseDiff(
        universe_name=universe_name,
        scan_date=scan_date,
        added=added,
        removed=removed,
        size_before=len(prev),
        size_after=len(current),
        size_alert=size_alert,
        size_alert_reason=size_reason,
        churn_alert=churn_alert,
    )


# ===========================================================================
# چک تازگی لیست دستی یورواستوکس ۵۰
# ===========================================================================

def check_eurostoxx_staleness(
    last_manual_update: str, today: date | None = None
) -> tuple[bool, int]:
    """
    طبق توصیه خود README، لیست یورواستوکس ۵۰ باید سالی یک‌بار به‌روز شود.
    این تابع فقط یک محاسبه ساده تاریخ است - تاریخ آخرین به‌روزرسانی دستی
    را باید خودتان در یک فایل کوچک (مثلا eurostoxx_meta.json) نگه دارید
    و هر بار که لیست را دستی به‌روز کردید، آپدیت کنید.

    last_manual_update: تاریخ به فرمت YYYY-MM-DD
    """
    today = today or date.today()
    last_update = datetime.strptime(last_manual_update, "%Y-%m-%d").date()
    days_since = (today - last_update).days
    is_stale = days_since > EUROSTOXX_STALENESS_DAYS
    return is_stale, days_since


# ===========================================================================
# نمونه استفاده (در ابتدای هر اسکنر، بلافاصله بعد از گرفتن لیست یونیورس)
# ===========================================================================
"""
from universe_tracking import diff_and_log_universe, check_eurostoxx_staleness

sp500_tickers = fetch_sp500_from_wikipedia()   # تابع موجود در کد فعلی شما
diff = diff_and_log_universe(sp500_tickers, "sp500", history_path="data/universe_history.jsonl")

if diff.size_alert or diff.churn_alert:
    # پیشنهاد: به‌جای متوقف کردن کامل اجرا، از آخرین یونیورس معتبر ذخیره‌شده
    # استفاده کنید و در داشبورد یک هشدار نمایش دهید که «امروز از یونیورس
    # دیروز استفاده شد چون لیست جدید مشکوک به نظر می‌رسید»
    logger.warning("استفاده از یونیورس روز قبل به‌جای لیست امروز که مشکوک است.")
    sp500_tickers = load_last_snapshot("sp500", "data/universe_history.jsonl").tickers

is_stale, days = check_eurostoxx_staleness("2025-09-01")
if is_stale:
    logger.warning("لیست یورواستوکس ۵۰ از %d روز پیش به‌روزرسانی نشده - بررسی دستی لازم است.", days)
"""
