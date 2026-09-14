# -*- coding: utf-8 -*-
"""
tests/test_market_scanner.py
تست‌های رگرسیون برای فرمول‌های امتیازدهی market_scanner.py.

هدف این فایل جلوگیری از تکرار دسته‌ای از باگ‌هاست که خود README پروژه
مستند کرده - یعنی تغییری در کد که بی‌سروصدا منطق امتیازدهی/فیلتر را
خراب می‌کند و کسی متوجه نمی‌شود تا وقتی به‌صورت دستی کشف شود.

این‌ها تست واحد (unit test) هستند، نه تست یکپارچگی؛ هیچ‌کدام به شبکه
واقعی (yfinance/Twelve Data) نیاز ندارند - همه با داده ساختگی کار می‌کنند.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_scanner as ms


# ===========================================================================
# normalize() - رتبه‌بندی صدکی
# ===========================================================================

def test_normalize_bounded_between_zero_and_one():
    s = pd.Series([10, 20, 30, 40, 50])
    result = ms.normalize(s, higher_is_better=True)
    assert (result >= 0).all() and (result <= 1).all()


def test_normalize_higher_is_better_direction():
    s = pd.Series([10, 20, 30])
    result = ms.normalize(s, higher_is_better=True)
    # بزرگ‌ترین مقدار باید بالاترین رتبه را بگیرد
    assert result.iloc[2] > result.iloc[0]


def test_normalize_lower_is_better_direction():
    s = pd.Series([10, 20, 30])
    result = ms.normalize(s, higher_is_better=False)
    # کوچک‌ترین مقدار باید بالاترین رتبه را بگیرد (مثلاً P/E پایین بهتر است)
    assert result.iloc[0] > result.iloc[2]


def test_normalize_robust_to_outlier():
    """
    رگرسیون مفهومی: رتبه‌بندی صدکی نباید مثل min-max توسط یک مقدار پرت
    شدید مچاله شود. یک مقدار پرت غول‌آسا نباید باعث شود بقیه مقادیر
    عملاً همه شبیه هم (نزدیک صفر) به نظر برسند.
    """
    s = pd.Series([10, 12, 11, 13, 9, 10000])  # آخری یک outlier فاحش است
    result = ms.normalize(s, higher_is_better=True)
    normal_values = result.iloc[:5]
    # مقادیر عادی باید همچنان در بازه‌ای پخش‌شده باشند، نه همه چسبیده به صفر
    assert normal_values.max() - normal_values.min() > 0.3


def test_normalize_all_nan_returns_neutral():
    s = pd.Series([np.nan, np.nan, np.nan])
    result = ms.normalize(s)
    assert (result == 0.5).all()


# ===========================================================================
# sector_relative_normalize() - مقایسه درون‌صنعتی، نه کل بازار
# ===========================================================================

def test_sector_relative_normalize_independent_groups():
    """
    رگرسیون مهم: یک سهم با P/E بالاتر از میانگین کل بازار، اگر در صنعتی با
    P/E ذاتاً بالا باشد (مثلاً فناوری) و نسبت به هم‌صنعتی‌هایش پایین باشد،
    باید امتیاز خوبی بگیرد - نه اینکه با بانک‌ها مقایسه شود.
    """
    df = pd.DataFrame({
        "pe_ratio": [10, 12, 40, 45],
        "sector": ["Financial Services", "Financial Services", "Technology", "Technology"],
    })
    result = ms.sector_relative_normalize(df, "pe_ratio", higher_is_better=False)
    # سهم فناوری با P/E=40 نسبت به هم‌صنعتی‌اش (45) بهتر است، پس باید رتبه بالاتر بگیرد
    assert result.iloc[2] > result.iloc[3]
    # سهم مالی با P/E=10 نسبت به هم‌صنعتی‌اش (12) بهتر است
    assert result.iloc[0] > result.iloc[1]


def test_sector_relative_normalize_uses_effective_sector_if_present():
    """اگر ستون effective_sector (خروجی sector_overrides.py) موجود باشد،
    باید به‌جای sector خام از آن استفاده شود - یعنی این دو سهم دیگر با هم
    مقایسه نمی‌شوند، هرچند sector خام هر دو یکسان است."""
    df = pd.DataFrame({
        "pe_ratio": [10, 50],
        "sector": ["Industrials", "Industrials"],           # برچسب خام (فرضاً اشتباه)
        "effective_sector": ["Financial Services", "Technology"],  # برچسب اصلاح‌شده
    })
    result_effective = ms.sector_relative_normalize(df, "pe_ratio", higher_is_better=False)

    df_raw_only = df.drop(columns=["effective_sector"])
    result_raw = ms.sector_relative_normalize(df_raw_only, "pe_ratio", higher_is_better=False)

    # با sector خام (هر دو در یک گروه)، این دو باید نسبت به هم رتبه‌بندی شوند
    # (نتایج متفاوت). با effective_sector (هرکدام گروه جدا)، هر دو مستقل
    # محاسبه می‌شوند - نتیجه باید متفاوت از حالت raw باشد.
    assert not result_effective.equals(result_raw), (
        "استفاده از effective_sector باید نتیجه را نسبت به گروه‌بندی بر پایه "
        "sector خام تغییر دهد؛ در غیر این صورت یعنی اصلاح صنعتی اعمال نمی‌شود."
    )


# ===========================================================================
# score_*() - بازه خروجی و رفتار منطقی
# ===========================================================================

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "pe_ratio": [15, 25, 100, np.nan],
        "ev_to_ebitda": [10, 15, 40, 12],
        "peg_ratio": [1.0, 1.5, 3.0, 0.8],
        "fcf_yield": [0.06, 0.03, -0.02, 0.05],
        "revenue_growth": [0.10, 0.05, -0.10, 0.20],
        "earnings_growth": [0.08, 0.02, -0.20, 0.15],
        "profit_margin": [0.20, 0.10, 0.02, 0.15],
        "roe": [0.25, 0.15, 0.05, 0.18],
        "debt_to_equity": [50, 150, 400, 60],
        "current_ratio": [1.8, 1.2, 0.8, 1.5],
        "free_cashflow": [1e9, 5e8, -1e8, 8e8],
        "momentum_12_1": [0.20, 0.05, -0.15, 0.10],
        "above_sma50": [True, True, False, True],
        "above_sma200": [True, False, False, True],
        "golden_cross": [True, False, False, True],
        "volatility_annualized": [0.25, 0.30, 0.50, 0.20],
        "analyst_upside": [0.10, 0.05, -0.05, 0.08],
        "recommendation": ["strong_buy", "hold", "sell", "buy"],
        "sector": ["Technology", "Technology", "Energy", "Technology"],
    })


def test_score_growth_value_bounded(sample_df):
    result = ms.score_growth_value(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_quality_bounded(sample_df):
    result = ms.score_quality(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_technical_bounded(sample_df):
    result = ms.score_technical(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_analyst_bounded(sample_df):
    result = ms.score_analyst(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_analyst_recommendation_ordering():
    """strong_buy باید امتیاز بیشتری از sell بگیرد - یک چک عقل سلیم ساده."""
    df = pd.DataFrame({
        "recommendation": ["strong_buy", "sell"],
        "analyst_upside": [0.0, 0.0],  # یکسان نگه داشته تا فقط اثر recommendation سنجیده شود
    })
    result = ms.score_analyst(df)
    assert result.iloc[0] > result.iloc[1]


def test_score_quality_debt_neutral_for_exempt_sector():
    """
    مکمل رگرسیون باگ #۱ در README در سطح امتیازدهی (نه فقط فیلتر):
    وقتی is_high_leverage_sector=True باشد، وضعیت بدهی نباید در
    فرمول‌های safety/quality (که این پرچم را چک می‌کنند) جریمه شود.
    این چک مستقیم روی reversal_scanner.score_safety انجام می‌شود چون
    market_scanner.score_quality فعلاً چنین معافیتی در سطح امتیازدهی ندارد
    (فقط در apply_quality_filters اعمال می‌شود) - این خودش یک یافته
    مستندسازی‌شده است، نه یک باگ.
    """
    pytest.skip(
        "score_quality در market_scanner.py معافیت بدهی را در سطح امتیاز اعمال "
        "نمی‌کند (فقط apply_quality_filters این کار را می‌کند) - پوشش این مورد "
        "در tests/test_reversal_scanner.py برای score_safety انجام شده."
    )


# ===========================================================================
# apply_quality_filters() - رگرسیون مستقیم روی باگ‌های مستند‌شده در README
# ===========================================================================

def test_debt_filter_exempts_high_leverage_sector():
    """رگرسیون باگ #۱ در README: بانک با بدهی بالا نباید حذف شود."""
    df = pd.DataFrame({
        "ticker": ["BANK", "INDUSTRIAL"],
        "market_cap": [5e10, 5e10],
        "currency": ["USD", "USD"],
        "sector": ["Financial Services", "Industrials"],
        "debt_to_equity": [500, 500],  # هر دو بدهی بسیار بالا دارند
        "pe_ratio": [10, 10], "ev_to_ebitda": [8, 8], "revenue_growth": [0.05, 0.05],
        "profit_margin": [0.1, 0.1], "momentum_3m": [0.02, 0.02], "roe": [0.1, 0.1],
        "target_mean_price": [100, 100],
    }).set_index("ticker")

    result = ms.apply_quality_filters(df, min_market_cap=1e9, min_data_fields=3)
    assert "BANK" in result.index, "بانک با بدهی بالا نباید به‌خاطر بدهی حذف شود"
    assert "INDUSTRIAL" not in result.index, "شرکت غیرمالی با بدهی/سرمایه ۵۰۰٪ باید حذف شود"


def test_min_data_fields_not_impossible():
    """
    رگرسیون مستقیم روی باگ #۲ در README: بعد از هر تغییر در لیست
    key_fields، min_data_fields نباید از تعداد فیلدهای ممکن بیشتر شود -
    وگرنه هیچ سهمی هرگز از فیلتر رد نمی‌شود.

    این تست با یک سهم که دقیقاً تعداد فیلد پیش‌فرض key_fields را کامل
    دارد چک می‌کند که رد نشود.
    """
    all_fields_df = pd.DataFrame({
        "ticker": ["FULL"],
        "market_cap": [5e10], "currency": ["USD"], "sector": ["Technology"],
        "pe_ratio": [20], "ev_to_ebitda": [15], "revenue_growth": [0.1],
        "profit_margin": [0.15], "momentum_3m": [0.05], "roe": [0.2],
        "target_mean_price": [100],
    }).set_index("ticker")

    result = ms.apply_quality_filters(all_fields_df, min_market_cap=1e9)
    assert "FULL" in result.index, (
        "سهمی با همه فیلدهای کلیدی پر باید از فیلتر تکمیل‌بودن داده رد شود؛ "
        "اگر این fail شد یعنی min_data_fields از تعداد فیلدهای موجود بیشتر شده."
    )


def test_market_cap_filter_converts_eur_to_usd_for_comparison():
    """
    رگرسیون مفهومی: در یونیورس ترکیبی، ارزش بازار شرکت یورویی باید قبل از
    مقایسه با آستانه (که به دلار است) به دلار تبدیل شود - وگرنه آستانه
    برای یک گروه سخت‌گیرانه‌تر از گروه دیگر می‌شود.
    """
    df = pd.DataFrame({
        "ticker": ["EUR_CO"],
        "market_cap": [1.5e9],  # به یورو - اگر بدون تبدیل مقایسه شود، رد کاذب می‌شود
        "currency": ["EUR"],
        "sector": ["Technology"],
        "pe_ratio": [20], "ev_to_ebitda": [15], "revenue_growth": [0.1],
        "profit_margin": [0.15], "momentum_3m": [0.05], "roe": [0.2],
        "target_mean_price": [100],
    }).set_index("ticker")

    # نرخ ۰.۸۶ یعنی ۱.۵ میلیارد یورو ≈ ۱.۷۴ میلیارد دلار - باید از آستانه ۲ میلیارد رد شود
    result_below = ms.apply_quality_filters(df, min_market_cap=2e9, fx_rate_usd_eur=0.86)
    assert "EUR_CO" not in result_below.index

    # ولی با آستانه ۱ میلیارد دلار باید قبول شود
    result_above = ms.apply_quality_filters(df, min_market_cap=1e9, fx_rate_usd_eur=0.86)
    assert "EUR_CO" in result_above.index


def test_error_rows_are_excluded():
    df = pd.DataFrame({
        "ticker": ["GOOD", "BAD"],
        "market_cap": [5e10, 5e10], "currency": ["USD", "USD"], "sector": ["Technology", "Technology"],
        "pe_ratio": [20, None], "ev_to_ebitda": [15, None], "revenue_growth": [0.1, None],
        "profit_margin": [0.15, None], "momentum_3m": [0.05, None], "roe": [0.2, None],
        "target_mean_price": [100, None],
        "error": [None, "insufficient_history"],
    }).set_index("ticker")
    result = ms.apply_quality_filters(df, min_market_cap=1e9, min_data_fields=3)
    assert "GOOD" in result.index
    assert "BAD" not in result.index


# ===========================================================================
# get_usd_to_eur_rate() - رگرسیون مستقیم روی باگ #۳ در README
# ===========================================================================

def test_fx_rate_falls_back_on_nan(monkeypatch):
    """رگرسیون باگ #۳: اگر یاهو یک نرخ NaN برگرداند، باید fallback فعال شود، نه NaN منتشر شود."""
    class FakeHist:
        empty = False
        def __getitem__(self, key):
            return pd.Series([np.nan])

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period): return FakeHist()

    monkeypatch.setattr(ms.yf, "Ticker", FakeTicker)
    rate = ms.get_usd_to_eur_rate()
    assert rate == 0.86


def test_fx_rate_falls_back_on_out_of_range(monkeypatch):
    """اگر نرخ برگشتی خارج از بازه منطقی (۰.۵ تا ۱.۵) باشد، باید fallback فعال شود."""
    class FakeHist:
        empty = False
        def __getitem__(self, key):
            return pd.Series([5.0])  # نرخ غیرممکن برای USD/EUR

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period): return FakeHist()

    monkeypatch.setattr(ms.yf, "Ticker", FakeTicker)
    rate = ms.get_usd_to_eur_rate()
    assert rate == 0.86


def test_fx_rate_accepts_valid_value(monkeypatch):
    class FakeHist:
        empty = False
        def __getitem__(self, key):
            return pd.Series([0.92])

    class FakeTicker:
        def __init__(self, symbol): pass
        def history(self, period): return FakeHist()

    monkeypatch.setattr(ms.yf, "Ticker", FakeTicker)
    rate = ms.get_usd_to_eur_rate()
    assert rate == 0.92


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
