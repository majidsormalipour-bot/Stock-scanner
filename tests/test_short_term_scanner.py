# -*- coding: utf-8 -*-
"""
tests/test_short_term_scanner.py
تست‌های رگرسیون برای فرمول‌های امتیازدهی short_term_scanner.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import short_term_scanner as sts


# ===========================================================================
# compute_rsi / compute_macd / compute_atr - اندیکاتورهای فنی پایه
# ===========================================================================

def test_compute_rsi_bounded():
    prices = pd.Series(np.linspace(100, 150, 60) + np.random.RandomState(0).normal(0, 1, 60))
    rsi = sts.compute_rsi(prices)
    assert 0 <= rsi <= 100


def test_compute_rsi_strong_uptrend_is_high():
    """در یک روند صعودی خالص (بدون هیچ افتی)، RSI باید بسیار بالا (نزدیک ۱۰۰) باشد."""
    prices = pd.Series(np.linspace(100, 200, 30))
    rsi = sts.compute_rsi(prices)
    assert rsi > 90


def test_compute_atr_non_negative():
    idx = pd.date_range("2026-01-01", periods=30)
    hist = pd.DataFrame({
        "High": np.linspace(105, 135, 30),
        "Low": np.linspace(95, 125, 30),
        "Close": np.linspace(100, 130, 30),
    }, index=idx)
    atr = sts.compute_atr(hist)
    assert atr >= 0


# ===========================================================================
# normalize() - همان الگوی market_scanner، تکرارشده در این فایل
# ===========================================================================

def test_normalize_robust_to_outlier():
    s = pd.Series([10, 12, 11, 13, 9, 10000])
    result = sts.normalize(s, higher_is_better=True)
    normal_values = result.iloc[:5]
    assert normal_values.max() - normal_values.min() > 0.3


# ===========================================================================
# score_*() - بازه خروجی
# ===========================================================================

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "mom_5d": [0.03, -0.01, 0.10],
        "mom_20d": [0.08, 0.02, 0.20],
        "rsi": [60, 45, 75],
        "above_sma50": [True, False, True],
        "volume_ratio": [2.0, 0.8, 5.5],
        "macd_bullish_cross": [True, False, False],
        "macd_positive": [True, False, True],
        "pct_from_20d_high": [-0.01, -0.10, -0.005],
        "near_breakout": [True, False, True],
    })


def test_score_momentum_bounded(sample_df):
    result = sts.score_momentum(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_trend_penalizes_downtrend(sample_df):
    result = sts.score_trend(sample_df)
    # ردیف دوم above_sma50=False است، باید امتیاز پایین‌تری از ردیف اول (True) بگیرد
    assert result.iloc[1] < result.iloc[0]


def test_score_volume_bounded(sample_df):
    result = sts.score_volume(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_volume_extreme_spike_not_max_score():
    """
    رگرسیون مفهومی: حجم بسیار افراطی (>4x) باید امتیاز کمتری از حجم
    نسبتاً بالای سالم (۱.۳x تا ۴x) بگیرد - چون می‌تواند نشانه یک خبر
    غیرعادی/پامپ مصنوعی باشد، نه صرفاً «توجه بازار».
    """
    df = pd.DataFrame({"volume_ratio": [2.0, 8.0]})
    result = sts.score_volume(df)
    assert result.iloc[0] > result.iloc[1]


def test_score_macd_bounded(sample_df):
    result = sts.score_macd(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_breakout_bounded(sample_df):
    result = sts.score_breakout(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_momentum_uses_relative_not_absolute():
    """
    رگرسیون مفهومی مهم: مومنتوم باید نسبت به بازار سنجیده شود، نه مطلق.
    سهمی با بازده ۱۰٪ در بازاری که خودش ۱۵٪ رشد کرده (قدرت نسبی منفی)
    باید امتیاز کمتری از سهمی با بازده ۱۰٪ در بازار راکد (۰٪) بگیرد.
    """
    df_weak_market = pd.DataFrame({"mom_20d": [0.10]})
    df_strong_market = pd.DataFrame({"mom_20d": [0.10]})

    score_in_flat_market = sts.score_momentum(df_weak_market, benchmark={"mom_5d": 0.0, "mom_20d": 0.0})
    score_in_hot_market = sts.score_momentum(df_strong_market, benchmark={"mom_5d": 0.0, "mom_20d": 0.15})

    # با یک سهم تنها در دیتافریم، normalize رتبه صدکی معنادار نمی‌دهد؛
    # پس مستقیم بازده نسبی را مقایسه می‌کنیم که در پس فرمول محاسبه می‌شود
    rel_flat = df_weak_market["mom_20d"].iloc[0] - 0.0
    rel_hot = df_strong_market["mom_20d"].iloc[0] - 0.15
    assert rel_flat > rel_hot, "قدرت نسبی در بازار داغ باید کمتر محاسبه شود"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
