# -*- coding: utf-8 -*-
"""
performance_scorecard.py
اسکریپت مستقل برای اندازه‌گیری عملکرد واقعی پیشنهادهای گذشته.

برخلاف سه اسکنر اصلی، این اسکریپت هر روز اجرا نمی‌شود - چون باید قیمت
فعلی صدها/هزاران رکورد تاریخی را بگیرد که هزینه شبکه/زمان زیادی دارد.
پیشنهاد: هفتگی (مثلاً یک‌شنبه‌ها) از طریق یک workflow جدا اجرا شود.

منطق کار:
  ۱. تاریخچه پیشنهادها را از data/picks_history.jsonl می‌خواند
  ۲. برای هر نماد یکتا، قیمت فعلی را از yfinance می‌گیرد
  ۳. بازده هر پیشنهاد را نسبت به قیمت روز پیشنهاد محاسبه می‌کند
  ۴. با شاخص مرجع (S&P 500 برای دلاری، یورواستوکس ۵۰ برای یورویی)
     مقایسه می‌کند تا آلفا (بازده اضافه بر بازار) به‌دست بیاید
  ۵. همبستگی بین امتیاز و بازده واقعی را حساب می‌کند - این مهم‌ترین
     عدد است: اگر همبستگی نزدیک صفر یا منفی باشد، یعنی فرمول امتیازدهی
     در عمل کار نمی‌کند و باید بازبینی شود.

⚠️ محدودیت روش‌شناسی مهم: «بازده» در این گزارش یعنی «بازده تا امروز»،
نه بازده دقیق در یک بازه ثابت (مثلاً دقیقاً ۲۸ روز بعد). یک پیشنهاد
۴۰ روزه در گروه «۲۸+ روز» با بازده ۴۰ روزه‌اش حساب می‌شود، نه بازده
دقیق روز ۲۸ام. این برای شروع کافی است؛ اگر دقت بیشتری لازم شد، باید
قیمت تاریخی دقیق در pick_date+N را هم ذخیره کرد (نه فقط قیمت فعلی).
"""

import argparse
import logging
import os
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from picks_tracker import load_picks_history
except ImportError as e:
    raise SystemExit(f"picks_tracker.py در دسترس نیست: {e}")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("performance_scorecard")

MATURITY_BUCKETS = [7, 28, 84]  # روز - تقریبا ۱ هفته، ۴ هفته، ۱۲ هفته
BENCHMARK_US = "^GSPC"
BENCHMARK_EU = "^STOXX50E"

BATCH_DELAY = float(os.environ.get("SCANNER_BATCH_DELAY", "1.0"))


def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """قیمت فعلی هر نماد یکتا را می‌گیرد - یک درخواست در هر تیک، با تاخیر بین درخواست‌ها
    برای هماهنگی با همان محدودیت‌های rate-limit یاهو که در بقیه پروژه رعایت می‌شود."""
    prices = {}
    for i, ticker in enumerate(tickers):
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.warning("قیمت فعلی %s گرفته نشد: %s", ticker, e)
        if (i + 1) % 20 == 0:
            logger.info("Progress: %d/%d قیمت فعلی گرفته شد...", i + 1, len(tickers))
        time.sleep(BATCH_DELAY)
    return prices


def fetch_benchmark_series(ticker: str, start_date: str) -> pd.Series | None:
    """کل تاریخچه قیمت پایانی شاخص مرجع را از start_date تا امروز می‌گیرد."""
    try:
        hist = yf.Ticker(ticker).history(start=start_date)
        if hist.empty:
            return None
        return hist["Close"]
    except Exception as e:
        logger.warning("داده شاخص مرجع %s گرفته نشد: %s", ticker, e)
        return None


def _nearest_price(series: pd.Series, target_date) -> float | None:
    """نزدیک‌ترین قیمت موجود به یک تاریخ مشخص را برمی‌گرداند (برای تعطیلات/آخر هفته)."""
    if series is None or series.empty:
        return None
    idx = series.index
    target = pd.Timestamp(target_date)
    if target.tzinfo is None and idx.tz is not None:
        target = target.tz_localize(idx.tz)
    pos = idx.searchsorted(target)
    pos = min(max(pos, 0), len(idx) - 1)
    return float(series.iloc[pos])


def build_performance_table(history: pd.DataFrame) -> pd.DataFrame:
    """
    تاریخچه خام پیشنهادها را با قیمت فعلی و بازده شاخص مرجع ترکیب می‌کند
    و یک جدول کامل با ستون‌های stock_return / benchmark_return / alpha
    برمی‌گرداند.
    """
    today = pd.Timestamp(date.today())
    history = history.copy()
    history["age_days"] = (today - history["date"]).dt.days

    unique_tickers = sorted(history["ticker"].unique())
    logger.info("گرفتن قیمت فعلی برای %d نماد یکتا...", len(unique_tickers))
    current_prices = fetch_current_prices(unique_tickers)
    history["current_price"] = history["ticker"].map(current_prices)
    history["stock_return"] = (history["current_price"] / history["price"]) - 1

    earliest = history["date"].min().strftime("%Y-%m-%d")
    logger.info("گرفتن شاخص‌های مرجع از %s تا امروز...", earliest)
    bench_us = fetch_benchmark_series(BENCHMARK_US, earliest)
    bench_eu = fetch_benchmark_series(BENCHMARK_EU, earliest)

    benchmark_returns = []
    for _, row in history.iterrows():
        series = bench_eu if row["currency"] == "EUR" else bench_us
        if series is None:
            benchmark_returns.append(np.nan)
            continue
        p0 = _nearest_price(series, row["date"])
        p1 = series.iloc[-1]
        if p0 and p0 > 0:
            benchmark_returns.append((p1 / p0) - 1)
        else:
            benchmark_returns.append(np.nan)

    history["benchmark_return"] = benchmark_returns
    history["alpha"] = history["stock_return"] - history["benchmark_return"]
    return history


def summarize(history: pd.DataFrame) -> pd.DataFrame:
    """
    برای هر اسکنر و هر بازه بلوغ (۷/۲۸/۸۴ روز)، آمار خلاصه می‌سازد:
    تعداد، نرخ موفقیت (٪ پیشنهادهای با بازده مثبت)، میانگین بازده،
    میانگین آلفا (بازده اضافه بر شاخص مرجع)، و همبستگی امتیاز-بازده.

    همبستگی مهم‌ترین ستون این جدول است: نزدیک صفر یا منفی یعنی فرمول
    امتیازدهی فعلی در عمل قدرت پیش‌بینی ندارد و باید بازبینی شود.
    """
    rows = []
    for scanner, sdf in history.groupby("scanner"):
        for horizon in MATURITY_BUCKETS:
            matured = sdf[sdf["age_days"] >= horizon].dropna(subset=["stock_return"])
            if len(matured) < 5:
                rows.append({
                    "scanner": scanner, "horizon_days": horizon, "n": len(matured),
                    "hit_rate": np.nan, "avg_return": np.nan, "avg_alpha": np.nan,
                    "score_return_corr": np.nan, "note": "داده کافی نیست (حداقل ۵ نمونه لازم است)",
                })
                continue

            hit_rate = (matured["stock_return"] > 0).mean()
            avg_return = matured["stock_return"].mean()
            avg_alpha = matured["alpha"].mean()
            corr = matured["score"].corr(matured["stock_return"])

            rows.append({
                "scanner": scanner, "horizon_days": horizon, "n": len(matured),
                "hit_rate": round(float(hit_rate), 4),
                "avg_return": round(float(avg_return), 4),
                "avg_alpha": round(float(avg_alpha), 4) if pd.notna(avg_alpha) else np.nan,
                "score_return_corr": round(float(corr), 4) if pd.notna(corr) else np.nan,
                "note": "",
            })
    return pd.DataFrame(rows)


def top_vs_bottom_quartile(history: pd.DataFrame, horizon: int = 28) -> pd.DataFrame:
    """
    برای هر اسکنر: آیا پیشنهادهایی که امتیاز بالاتری گرفتند، واقعا بازده
    بهتری هم داشتند؟ چارک بالا (۲۵٪ امتیاز برتر) را با چارک پایین مقایسه
    می‌کند - این عدد از ضریب همبستگی برای کاربر غیرآماری قابل‌فهم‌تر است.
    """
    rows = []
    for scanner, sdf in history.groupby("scanner"):
        matured = sdf[sdf["age_days"] >= horizon].dropna(subset=["stock_return"])
        if len(matured) < 8:
            continue
        q_high = matured["score"].quantile(0.75)
        q_low = matured["score"].quantile(0.25)
        top = matured[matured["score"] >= q_high]
        bottom = matured[matured["score"] <= q_low]
        rows.append({
            "scanner": scanner,
            "horizon_days": horizon,
            "top_quartile_avg_return": round(float(top["stock_return"].mean()), 4),
            "bottom_quartile_avg_return": round(float(bottom["stock_return"].mean()), 4),
            "n_top": len(top), "n_bottom": len(bottom),
        })
    return pd.DataFrame(rows)


def write_html_report(summary: pd.DataFrame, quartile: pd.DataFrame, history: pd.DataFrame,
                       out_path: str = "site/performance_scorecard.html") -> None:
    scanner_fa = {
        "market_scanner": "بلندمدت", "short_term_scanner": "کوتاه‌مدت", "reversal_scanner": "بازگشتی",
    }
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def fmt_pct(x):
        return f"{x:+.1%}" if pd.notna(x) else "—"

    rows_html = []
    for _, r in summary.iterrows():
        name = scanner_fa.get(r["scanner"], r["scanner"])
        if r["note"]:
            rows_html.append(
                f"<tr><td>{name}</td><td>{r['horizon_days']} روز</td>"
                f"<td colspan='4' style='color:#888'>{r['note']} (n={r['n']})</td></tr>"
            )
            continue
        corr = r["score_return_corr"]
        corr_color = "#4caf50" if pd.notna(corr) and corr > 0.15 else (
            "#ef5350" if pd.notna(corr) and corr < -0.05 else "#ffb74d"
        )
        rows_html.append(f"""
        <tr>
          <td>{name}</td><td>{r['horizon_days']} روز (n={r['n']})</td>
          <td>{r['hit_rate']:.0%}</td>
          <td>{fmt_pct(r['avg_return'])}</td>
          <td>{fmt_pct(r['avg_alpha'])}</td>
          <td style="color:{corr_color};font-weight:bold">{corr:.2f}</td>
        </tr>""")

    quartile_html = "".join(f"""
        <tr>
          <td>{scanner_fa.get(r['scanner'], r['scanner'])}</td>
          <td>{r['horizon_days']} روز</td>
          <td style="color:#4caf50">{fmt_pct(r['top_quartile_avg_return'])} (n={r['n_top']})</td>
          <td style="color:#ef5350">{fmt_pct(r['bottom_quartile_avg_return'])} (n={r['n_bottom']})</td>
        </tr>""" for _, r in quartile.iterrows()) or "<tr><td colspan='4' style='color:#888'>داده کافی نیست</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>کارنامه عملکرد واقعی</title>
<style>
  body {{ font-family: Tahoma, Arial, sans-serif; background:#0f1115; color:#e6e6e6;
         padding:20px; max-width:700px; margin:auto; }}
  h1 {{ font-size:19px; }}
  h2 {{ font-size:15px; color:#ccc; margin-top:28px; }}
  .updated {{ color:#888; font-size:12px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-bottom:10px; }}
  th, td {{ padding:8px 6px; text-align:center; border-bottom:1px solid #2c303a; }}
  th {{ color:#999; font-weight:normal; font-size:11px; }}
  .note {{ background:#151a24; border:1px solid #2c303a; border-radius:10px; padding:14px 16px;
           font-size:12px; color:#aaa; margin-top:20px; line-height:1.8; }}
  .warn {{ color:#ffb74d; }}
</style>
</head>
<body>
  <h1>📈 کارنامه عملکرد واقعی پیشنهادها</h1>
  <p class="updated">به‌روزرسانی: {now} · بر پایه {len(history)} پیشنهاد ثبت‌شده</p>

  <h2>آیا امتیاز بالاتر واقعاً بازده بهتری می‌دهد؟</h2>
  <table>
    <tr><th>اسکنر</th><th>بازه</th><th>نرخ موفقیت</th><th>میانگین بازده</th>
        <th>میانگین آلفا (نسبت به شاخص)</th><th>همبستگی امتیاز-بازده</th></tr>
    {"".join(rows_html)}
  </table>

  <h2>چارک برتر امتیاز در برابر چارک پایین (بازه ۲۸ روزه)</h2>
  <table>
    <tr><th>اسکنر</th><th>بازه</th><th>۲۵٪ امتیاز برتر</th><th>۲۵٪ امتیاز پایین</th></tr>
    {quartile_html}
  </table>

  <div class="note">
    <b>راهنمای خواندن این جدول:</b><br>
    • <b>همبستگی امتیاز-بازده</b> مهم‌ترین عدد است: هرچه به ۱ نزدیک‌تر باشد یعنی
    امتیاز بالاتر واقعاً با بازده بهتر همراه بوده. نزدیک صفر یا منفی
    (<span class="warn">نارنجی/قرمز</span>) یعنی فرمول در این بازه زمانی
    قدرت پیش‌بینی نداشته - این لزوماً به‌معنای خراب بودن فرمول نیست، ممکن
    است به تعداد نمونه کم یا شرایط خاص بازار مربوط باشد.<br>
    • <b>آلفا</b> یعنی بازده اضافه نسبت به شاخص مرجع (S&P 500 یا یورواستوکس ۵۰).
    آلفای مثبت یعنی این اسکنر بهتر از صرفاً نگه‌داشتن شاخص عمل کرده.<br>
    • بازده‌ها «بازده تا امروز» هستند، نه بازده دقیق در همان روز بازه
    (مثلاً یک پیشنهاد ۴۰ روزه در بازه «۲۸+ روز» با بازده ۴۰ روزه‌اش حساب
    می‌شود). این گزارش صرفاً تحلیلی است، نه توصیه مالی.
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def run(history_path: str = "data/picks_history.jsonl",
        out_html: str = "performance_scorecard.html",
        out_csv: str = "performance_detail.csv") -> None:
    history = load_picks_history(history_path)
    if history.empty:
        logger.warning("هیچ تاریخچه‌ای در %s پیدا نشد - هنوز داده کافی برای تحلیل جمع نشده.", history_path)
        return

    history = build_performance_table(history)
    history.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("جزئیات کامل در %s ذخیره شد.", out_csv)

    summary = summarize(history)
    quartile = top_vs_bottom_quartile(history, horizon=28)
    write_html_report(summary, quartile, history, out_path=out_html)
    logger.info("کارنامه عملکرد در %s ذخیره شد.", out_html)

    print("\n=== خلاصه ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="کارنامه عملکرد واقعی پیشنهادهای اسکنرها")
    parser.add_argument("--history", default="data/picks_history.jsonl")
    parser.add_argument("--out-html", default="performance_scorecard.html")
    parser.add_argument("--out-csv", default="performance_detail.csv")
    args = parser.parse_args()
    run(history_path=args.history, out_html=args.out_html, out_csv=args.out_csv)
