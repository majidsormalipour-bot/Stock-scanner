# -*- coding: utf-8 -*-
"""
یک صفحه فرود ساده می‌سازد که به هر سه گزارش لینک می‌دهد، همراه با
زمان دقیق آخرین اجرا - تا وقتی از گوشی باز می‌کنید بدانید داده چقدر تازه است.

همچنین (در صورت موفقیت‌آمیز بودن) یک بنر «رژیم بازار» نمایش می‌دهد -
صرفا برای زمینه تفسیر، نه فیلتر کردن سهام (جزئیات در market_regime.py).
"""
import datetime

try:
    from market_regime import fetch_market_regime_via_yfinance, regime_interpretation
except ImportError:
    fetch_market_regime_via_yfinance = regime_interpretation = None

now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

regime_banner_html = ""
if fetch_market_regime_via_yfinance is not None:
    try:
        regime = fetch_market_regime_via_yfinance()
        if regime is not None:
            trend_fa = {
                "strong_uptrend": "صعودی قوی 🟢",
                "uptrend": "صعودی",
                "sideways": "رنج‌زده/خنثی 🟡",
                "downtrend": "نزولی",
                "strong_downtrend": "نزولی قوی 🔴",
            }.get(regime.trend_label, regime.trend_label)
            vol_fa = {
                "low": "کم", "normal": "عادی", "elevated": "بالا",
                "high": "شدید 🔴", "unknown": "نامشخص",
            }.get(regime.volatility_label, regime.volatility_label)

            notes = regime_interpretation(regime)
            notes_html = "".join(f"<li>{n}</li>" for n in notes.values())

            regime_banner_html = f"""
  <div class="regime">
    <div class="regime-title">🌡️ وضعیت کلی بازار ({regime.as_of_date})</div>
    <div class="regime-metrics">روند: <b>{trend_fa}</b> · نوسان: <b>{vol_fa}</b>
      · افت از سقف ۵۲هفته: <b>{regime.drawdown_from_high_pct:.1%}</b></div>
    <ul class="regime-notes">{notes_html}</ul>
    <div class="regime-disclaimer">این صرفا زمینه تفسیر است، نه فیلتری روی سهام - تصمیم نهایی با شماست.</div>
  </div>"""
    except Exception:
        regime_banner_html = ""

html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>داشبورد اسکن سهام</title>
<style>
  body {{ font-family: Tahoma, Arial, sans-serif; background:#0f1115; color:#e6e6e6;
         padding:24px; max-width:480px; margin:auto; }}
  h1 {{ font-size:20px; }}
  .updated {{ color:#888; font-size:12px; margin-bottom:24px; }}
  a.card {{ display:block; background:#1b1e26; border:1px solid #2c303a; border-radius:12px;
            padding:20px; margin-bottom:16px; text-decoration:none; color:#e6e6e6; }}
  a.card:active {{ background:#242832; }}
  .title {{ font-size:16px; font-weight:bold; margin-bottom:6px; }}
  .desc {{ font-size:13px; color:#aaa; }}
  .emoji {{ font-size:22px; margin-left:8px; }}
  .regime {{ background:#151a24; border:1px solid #2c303a; border-radius:10px; padding:14px 16px; margin-bottom:20px; }}
  .regime-title {{ font-weight:bold; font-size:14px; margin-bottom:6px; }}
  .regime-metrics {{ font-size:13px; color:#bbb; margin-bottom:8px; }}
  .regime-notes {{ font-size:12px; color:#a8c8e0; margin:0 0 8px 0; padding-right:18px; }}
  .regime-disclaimer {{ font-size:11px; color:#777; }}
</style>
</head>
<body>
  <h1>📊 داشبورد اسکن بازار سهام</h1>
  <p class="updated">آخرین اجرا: {now}</p>
  {regime_banner_html}

  <a class="card" href="market_scan_report.html">
    <span class="emoji">🏆</span>
    <div class="title">پیشنهادهای بلندمدت</div>
    <div class="desc">رتبه‌بندی بر اساس بنیاد، سلامت مالی، روند و نظر تحلیل‌گران</div>
  </a>

  <a class="card" href="short_term_scan_report.html">
    <span class="emoji">⚡</span>
    <div class="title">فرصت‌های کوتاه‌مدت</div>
    <div class="desc">مومنتوم، حجم معاملات، MACD و حد ضرر پیشنهادی</div>
  </a>

  <a class="card" href="reversal_scan_report.html">
    <span class="emoji">🔄</span>
    <div class="title">بازگشت از حمایت</div>
    <div class="desc">سهام افتاده ولی ارزنده، نزدیک سطح حمایت فنی</div>
  </a>

  <p style="color:#666;font-size:11px;margin-top:24px;">
    این گزارش‌ها هر روز خودکار به‌روزرسانی می‌شوند و صرفاً بر پایه فرمول‌های
    عددی هستند، نه توصیه مالی.
  </p>
</body>
</html>"""

with open("site/index.html", "w", encoding="utf-8") as f:
    f.write(html)
