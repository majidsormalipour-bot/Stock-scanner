"""
picks_tracker.py
ماژول ثبت تاریخچه پیشنهادها - پایه برای اندازه‌گیری عملکرد واقعی اسکنرها.

مشکل: سیستم فعلی هر روز رتبه‌بندی می‌کند ولی هیچ‌جا ثبت نمی‌شود که «سهمی
که ۳۰ روز پیش امتیاز ۰.۸۵ گرفت، واقعاً چقدر بازده داشت؟». بدون این تاریخچه،
نمی‌توان فهمید فرمول امتیازدهی واقعاً پیش‌بینی‌کننده است یا نه - فقط حدس
می‌زنیم.

این ماژول فقط بخش «ثبت» را انجام می‌دهد (سبک، بدون شبکه اضافه، هر روز در
انتهای هر اسکن اجرا می‌شود). تحلیل و محاسبه بازده واقعی در یک اسکریپت
جدا (performance_scorecard.py) انجام می‌شود که هفتگی/ماهانه اجرا می‌شود -
چون آن بخش نیاز به گرفتن قیمت فعلی صدها سهم دارد و نباید هر روز تکرار شود.

طراحی مثل universe_tracking.py: append-only JSONL، ساده، بدون دیتابیس.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("picks_tracker")


@dataclass
class PickRecord:
    date: str          # YYYY-MM-DD - تاریخی که این پیشنهاد صادر شد
    scanner: str        # "market_scanner" / "short_term_scanner" / "reversal_scanner"
    ticker: str
    score: float
    price: float         # قیمت به ارز اصلی سهم (native)، نه ارز نمایشی تبدیل‌شده -
                          # تا در تحلیل بعدی نیازی به نگرانی درباره نرخ ارز نباشد
    currency: str


def log_daily_picks(
    picks_df: pd.DataFrame,
    scanner_name: str,
    price_col: str = "current_price",
    score_col: str = "total_score",
    currency_col: str = "currency",
    history_path: str = "data/picks_history.jsonl",
    pick_date: Optional[str] = None,
) -> int:
    """
    هر ردیف از picks_df (که ایندکسش ticker است) را به‌عنوان یک رکورد جدید
    اضافه می‌کند. باید با قیمت *native* (قبل از تبدیل به ارز نمایشی) صدا
    زده شود - این مسئولیت فراخوان (خود اسکنر) است، نه این تابع.

    برمی‌گرداند: تعداد رکوردهای نوشته‌شده.
    """
    pick_date = pick_date or date.today().isoformat()
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with path.open("a", encoding="utf-8") as f:
        for ticker, row in picks_df.iterrows():
            price = row.get(price_col)
            score = row.get(score_col)
            if price is None or score is None or pd.isna(price) or pd.isna(score):
                continue
            record = PickRecord(
                date=pick_date,
                scanner=scanner_name,
                ticker=str(ticker),
                score=round(float(score), 6),
                price=round(float(price), 4),
                currency=str(row.get(currency_col, "USD")),
            )
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n_written += 1

    logger.info("%d پیشنهاد از %s برای تاریخ %s ثبت شد.", n_written, scanner_name, pick_date)
    return n_written


def load_picks_history(history_path: str = "data/picks_history.jsonl") -> pd.DataFrame:
    """کل تاریخچه را به‌صورت DataFrame برمی‌گرداند (ستون‌ها: date, scanner, ticker, score, price, currency)."""
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "scanner", "ticker", "score", "price", "currency"])

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame(columns=["date", "scanner", "ticker", "score", "price", "currency"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ===========================================================================
# نمونه استفاده (در انتهای run() هر سه اسکنر، درست قبل از تبدیل ارز نمایشی)
# ===========================================================================
"""
from picks_tracker import log_daily_picks

picks = diversify_top_picks(df, top_n)
picks = revalidate_prices(picks)

# قیمت native را همین‌جا (قبل از تبدیل ارز) ثبت می‌کنیم:
log_daily_picks(picks, scanner_name="market_scanner", history_path="data/picks_history.jsonl")

# بعد از این خط، تبدیل ارز نمایشی مثل قبل ادامه پیدا می‌کند (picks تغییر
# می‌کند ولی رکورد ثبت‌شده دست‌نخورده می‌ماند، چون قبل از تغییر کپی گرفته شد)
"""
