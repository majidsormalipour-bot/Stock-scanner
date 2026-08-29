# -*- coding: utf-8 -*-
"""
یک صفحه فرود ساده می‌سازد که به هر دو گزارش لینک می‌دهد، همراه با
زمان دقیق آخرین اجرا - تا وقتی از گوشی باز می‌کنید بدانید داده چقدر تازه است.
"""
import datetime

now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

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
</style>
</head>
<body>
  <h1>📊 داشبورد اسکن بازار سهام</h1>
  <p class="updated">آخرین اجرا: {now}</p>

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

  <p style="color:#666;font-size:11px;margin-top:24px;">
    این گزارش‌ها هر روز خودکار به‌روزرسانی می‌شوند و صرفاً بر پایه فرمول‌های
    عددی هستند، نه توصیه مالی.
  </p>
</body>
</html>"""

with open("site/index.html", "w", encoding="utf-8") as f:
    f.write(html)
