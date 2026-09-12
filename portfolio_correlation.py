"""
portfolio_correlation.py
ماژول تشخیص ریسک تمرکز پنهان در پیشنهادهای نهایی.

مشکل: diversify_top_picks فقط محدودیت «حداکثر N سهم در هر صنعت» را
رعایت می‌کند. ولی دو سهم از دو صنعت متفاوت (مثلاً یک بانک و یک شرکت
انرژی) می‌توانند در عمل رفتار قیمتی بسیار مشابهی داشته باشند (مثلاً هر
دو به نرخ بهره یا قیمت نفت حساس باشند). برچسب صنعت یک proxy ناقص برای
ریسک هم‌حرکتی واقعی است.

این ماژول همبستگی واقعی بازده روزانه بین پیشنهادهای نهایی (نه کل
یونیورس) را محاسبه می‌کند - چون فقط ۱۰-۱۵ نماد است، هزینه شبکه‌اش
ناچیز است و نیازی به احتیاط rate-limit مثل بقیه پروژه ندارد.

⚠️ محدودیت روش‌شناسی: همبستگی بر پایه بازده به ارز محلی هر سهم محاسبه
می‌شود (بدون تعدیل نرخ ارز)، چون اثر نرخ ارز روی همبستگی نسبی بین دو
سهم اروپایی/آمریکایی معمولاً کوچک است و برای این تحلیل کیفی کافی است.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("portfolio_correlation")

# آستانه‌ها heuristic‌اند، نه بهینه‌سازی‌شده روی داده خاص:
# همبستگی زوجی بالای ۰.۷ در بازه ۶ ماهه برای دو سهم «متفاوت» قابل توجه است
HIGH_PAIR_CORRELATION = 0.70
# یک جفت با همبستگی بسیار شدید به‌تنهایی هم قابل هشدار است، حتی اگر
# بقیه سبد کاملاً مستقل باشند - چون یعنی عملاً دو "نسخه" از یک سهم دارید
VERY_HIGH_SINGLE_PAIR = 0.85
# میانگین همبستگی زوجی بالای ۰.۵ برای یک سبد ۱۰-۱۵ تایی «متنوع فرضی»
# نشانه این است که تنوع صنعتی، تنوع واقعی ریسک ایجاد نکرده
HIGH_AVG_CORRELATION = 0.50
# دو یا بیشتر جفت پرهمبستگی، حتی اگر هرکدام به‌تنهایی خیلی شدید نباشند،
# نشانه یک الگوی تکرارشونده است نه یک اتفاق مجزا
MIN_HIGH_PAIRS_FOR_ALERT = 2
MIN_TICKERS_FOR_ANALYSIS = 4


@dataclass
class CorrelationPair:
    ticker_a: str
    ticker_b: str
    correlation: float


def fetch_price_history(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """
    قیمت پایانی چند نماد را هم‌زمان می‌گیرد. چون تعداد کم است (۱۰-۱۵ تا،
    نه کل یونیورس)، یک درخواست دسته‌ای ساده کافی است و نیازی به
    batching/delay مثل fetch_universe_data ندارد.
    """
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame()
    # وقتی بیش از یک نماد باشد، yfinance ستون‌های multi-index برمی‌گرداند
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", axis=1, level=0)
    else:
        close = data[["Close"]] if "Close" in data.columns else data
    return close


def compute_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    """ماتریس همبستگی بازده روزانه (نه قیمت خام - قیمت خام همبستگی کاذب بالایی از روند کلی بازار می‌گیرد)."""
    returns = price_df.pct_change().dropna(how="all")
    return returns.corr()


def find_high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = HIGH_PAIR_CORRELATION
) -> list[CorrelationPair]:
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr_matrix.iloc[i, j]
            if pd.notna(c) and c >= threshold:
                pairs.append(CorrelationPair(cols[i], cols[j], round(float(c), 3)))
    return sorted(pairs, key=lambda p: -p.correlation)


def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    n = len(corr_matrix)
    if n < 2:
        return float("nan")
    vals = [
        corr_matrix.iloc[i, j]
        for i in range(n) for j in range(i + 1, n)
        if pd.notna(corr_matrix.iloc[i, j])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def analyze_concentration(tickers: list[str], period: str = "6mo") -> dict:
    """
    تابع اصلی: قیمت‌ها را می‌گیرد، همبستگی را حساب می‌کند و یک خلاصه
    ساختاریافته برمی‌گرداند که می‌تواند مستقیم در HTML نمایش داده شود.
    """
    tickers = list(dict.fromkeys(tickers))  # حذف تکراری با حفظ ترتیب
    if len(tickers) < MIN_TICKERS_FOR_ANALYSIS:
        return {"available": False, "reason": "تعداد پیشنهادها برای تحلیل همبستگی کافی نیست"}

    try:
        price_df = fetch_price_history(tickers, period=period)
        if price_df.empty or price_df.shape[1] < MIN_TICKERS_FOR_ANALYSIS:
            return {"available": False, "reason": "داده قیمت کافی برای تحلیل همبستگی گرفته نشد"}

        corr = compute_correlation_matrix(price_df)
        avg_corr = average_pairwise_correlation(corr)
        high_pairs = find_high_correlation_pairs(corr)
        max_pair_corr = high_pairs[0].correlation if high_pairs else 0.0
        concentration_alert = (
            (pd.notna(avg_corr) and avg_corr >= HIGH_AVG_CORRELATION)
            or len(high_pairs) >= MIN_HIGH_PAIRS_FOR_ALERT
            or max_pair_corr >= VERY_HIGH_SINGLE_PAIR
        )

        return {
            "available": True,
            "n_tickers": corr.shape[0],
            "avg_correlation": round(avg_corr, 3) if pd.notna(avg_corr) else None,
            "high_correlation_pairs": high_pairs,
            "concentration_alert": concentration_alert,
        }
    except Exception as e:
        logger.warning("تحلیل همبستگی شکست خورد: %s", e)
        return {"available": False, "reason": str(e)}


def build_html_note(result: dict) -> str:
    """یک بنر HTML آماده برای درج در گزارش می‌سازد؛ اگر چیزی برای هشدار نباشد، رشته خالی برمی‌گرداند."""
    if not result.get("available"):
        return ""

    if not result.get("concentration_alert"):
        return ""

    avg_corr = result.get("avg_correlation")
    pairs = result.get("high_correlation_pairs", [])
    pairs_text = "، ".join(f"{p.ticker_a}-{p.ticker_b} ({p.correlation:.2f})" for p in pairs[:5])

    detail = f"میانگین همبستگی زوجی: {avg_corr:.2f}" if avg_corr is not None else ""
    if pairs_text:
        detail += f"{' · ' if detail else ''}جفت‌های پرهمبستگی: {pairs_text}"

    return f"""
  <div class="warning" style="border-color:#8b6914;">
    ⚠️ <b>ریسک تمرکز پنهان</b>: با وجود تنوع صنعتی ظاهری، بازده قیمتی این
    پیشنهادها در ۶ ماه اخیر همبستگی بالایی داشته ({detail}). یعنی در یک
    ریزش بازار، احتمالاً همزمان افت می‌کنند - تنوع صنعتی به‌تنهایی تضمین
    کاهش ریسک سبد نیست.
  </div>"""


# ===========================================================================
# نمونه استفاده (بعد از diversify_top_picks، قبل از write_html_report)
# ===========================================================================
"""
from portfolio_correlation import analyze_concentration, build_html_note

picks = diversify_top_picks(df, top_n)
concentration = analyze_concentration(picks.index.tolist())
concentration_note = build_html_note(concentration)

write_html_report(picks, html_file, currency=DISPLAY_CURRENCY, fx_rate=fx_rate or 1.0,
                   concentration_note=concentration_note)
"""
