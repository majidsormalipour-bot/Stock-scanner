"""
cross_signal.py
ماژول تشخیص «توافق چند استراتژی» - وقتی یک سهم هم‌زمان در چند اسکنر
با فلسفه‌های متفاوت (بلندمدت/کوتاه‌مدت/بازگشتی) در رتبه‌های برتر ظاهر
می‌شود.

چرا این سیگنال ارزش دارد: سه اسکنر عمداً مستقل و بر پایه باورهای متفاوت
طراحی شده‌اند (طبق README پروژه) و اغلب سیگنال‌های متضاد می‌دهند. وقتی
با وجود این استقلال، یک سهم در دو یا سه اسکنر هم‌زمان بالا می‌آید، این
قوی‌تر از ظاهر شدن در فقط یکی است - چون یعنی از زوایای مختلف (ارزندگی+
سلامت مالی، مومنتوم قیمتی، نزدیکی به حمایت) synced شده.

⚠️ این ماژول در generate_index.py اجرا می‌شود، نه در خود سه اسکنر - چون
باید بعد از این‌که هر سه کامل شدند اجرا شود (در workflow روزانه، این سه
به‌ترتیب اجرا می‌شوند و فایل‌های CSV هرکدام را در همان پوشه کاری باقی
می‌گذارند).

نکته روش‌شناسی: به‌جای پیشنهادهای نهایی «متنوع‌شده» (که فیلتر سقف صنعتی
رویشان اعمال شده)، از N رتبه برتر فایل کامل CSV هر اسکنر استفاده می‌شود
- تقریب معقولی از چیزی است که کاربر می‌بیند، هرچند ممکن است کمی با
پیشنهادهای نهایی (بعد از تنوع‌بخشی) فرق داشته باشد.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger("cross_signal")

SCANNER_LABELS_FA = {
    "market_scanner": "بلندمدت",
    "short_term_scanner": "کوتاه‌مدت",
    "reversal_scanner": "بازگشتی",
}

DEFAULT_CSV_PATHS = {
    "market_scanner": "market_scan_results.csv",
    "short_term_scanner": "short_term_scan_results.csv",
    "reversal_scanner": "reversal_scan_results.csv",
}


def _load_top_n(csv_path: str, top_n: int, score_col: str = "total_score") -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    if score_col not in df.columns:
        return pd.DataFrame()
    return df.sort_values(score_col, ascending=False).head(top_n)[[score_col]]


def find_agreement(
    csv_paths: dict[str, str] = None, top_n: int = 15, score_col: str = "total_score"
) -> pd.DataFrame:
    """
    برای هر نمادی که در top_n حداقل ۲ اسکنر ظاهر شده، یک ردیف با امتیاز
    آن در هرکدام از اسکنرها برمی‌گرداند. اگر کمتر از ۲ فایل CSV در دسترس
    باشد (مثلاً یکی از اسکنرها خطا داده)، DataFrame خالی برمی‌گرداند.
    """
    csv_paths = csv_paths or DEFAULT_CSV_PATHS
    frames = {}
    for name, path in csv_paths.items():
        if not os.path.exists(path):
            continue
        try:
            frames[name] = _load_top_n(path, top_n=top_n, score_col=score_col)
        except Exception as e:
            logger.warning("خواندن %s شکست خورد: %s", path, e)

    if len(frames) < 2:
        return pd.DataFrame()

    all_tickers = set()
    for df in frames.values():
        all_tickers |= set(df.index)

    rows = []
    for ticker in all_tickers:
        present_in = [name for name, df in frames.items() if ticker in df.index]
        if len(present_in) < 2:
            continue
        row = {
            "ticker": ticker,
            "n_scanners": len(present_in),
            "scanners": present_in,
        }
        for name, df in frames.items():
            row[name] = round(float(df.loc[ticker, score_col]), 3) if ticker in df.index else None
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["n_scanners"] + list(frames.keys()), ascending=False)
    return result


def build_html_section(agreement_df: pd.DataFrame) -> str:
    """یک بخش HTML آماده برای درج در صفحه فرود (generate_index.py) می‌سازد."""
    if agreement_df is None or agreement_df.empty:
        return ""

    scanner_cols = [c for c in DEFAULT_CSV_PATHS if c in agreement_df.columns]
    rows_html = []
    for _, r in agreement_df.iterrows():
        badges = " ".join(
            f'<span class="badge">{SCANNER_LABELS_FA.get(s, s)}: {r[s]:.2f}</span>'
            for s in scanner_cols if pd.notna(r.get(s))
        )
        rows_html.append(f"""
        <div class="agree-card">
          <span class="agree-ticker">{r['ticker']}</span>
          <span class="agree-count">{r['n_scanners']} اسکنر موافق</span>
          <div class="agree-badges">{badges}</div>
        </div>""")

    return f"""
  <div class="agree-section">
    <div class="agree-title">🤝 سهام مورد توافق چند استراتژی</div>
    <p class="agree-desc">این سهام هم‌زمان در رتبه‌های برتر بیش از یک اسکنر مستقل ظاهر شده‌اند -
    سیگنالی قوی‌تر از ظاهر شدن در فقط یک استراتژی.</p>
    {''.join(rows_html)}
  </div>"""


CSS_SNIPPET = """
  .agree-section { background:#151a24; border:1px solid #2c303a; border-radius:10px;
                    padding:14px 16px; margin-bottom:20px; }
  .agree-title { font-weight:bold; font-size:14px; margin-bottom:6px; }
  .agree-desc { font-size:12px; color:#999; margin-bottom:10px; }
  .agree-card { background:#1b1e26; border-radius:8px; padding:10px 12px; margin-bottom:8px; }
  .agree-ticker { font-weight:bold; color:#4fc3f7; direction:ltr; font-size:15px; }
  .agree-count { color:#a8d8a8; font-size:12px; margin-right:10px; }
  .agree-badges { margin-top:6px; }
  .badge { display:inline-block; background:#2c303a; border-radius:6px; padding:3px 8px;
           font-size:11px; color:#ccc; margin-left:6px; }
"""


# ===========================================================================
# نمونه استفاده (در generate_index.py، بعد از اجرای هر سه اسکنر)
# ===========================================================================
"""
from cross_signal import find_agreement, build_html_section, CSS_SNIPPET

agreement = find_agreement(top_n=15)
agreement_html = build_html_section(agreement)
# CSS_SNIPPET را داخل تگ <style> صفحه اضافه کنید، و agreement_html را
# در بدنه صفحه (مثلاً بعد از بنر رژیم بازار) درج کنید.
"""
