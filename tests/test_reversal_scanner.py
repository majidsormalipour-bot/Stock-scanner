# -*- coding: utf-8 -*-
"""
tests/test_reversal_scanner.py
تست‌های رگرسیون برای فرمول‌های امتیازدهی reversal_scanner.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reversal_scanner as rs


def test_compute_rsi_strong_uptrend_is_high():
    """رگرسیون همان اصلاح RSI که در short_term_scanner کشف شد - اینجا هم تکرار شده بود."""
    prices = pd.Series(np.linspace(100, 200, 30))
    rsi = rs.compute_rsi(prices)
    assert rsi > 90


def test_compute_rsi_strong_downtrend_is_low():
    prices = pd.Series(np.linspace(200, 100, 30))
    rsi = rs.compute_rsi(prices)
    assert rsi < 10


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "pe_ratio": [8, 12, 30],
        "ev_to_ebitda": [6, 9, 20],
        "peg_ratio": [0.8, 1.2, 2.5],
        "pct_above_52w_low": [0.05, 0.20, 0.40],
        "pct_vs_sma200": [-0.02, -0.15, 0.10],
        "rsi": [25, 45, 65],
        "made_new_low_last_3d": [False, True, False],
        "mom_5d": [0.02, -0.10, 0.01],
        "profit_margin": [0.10, 0.05, 0.15],
        "roe": [0.15, 0.08, 0.20],
        "debt_to_equity": [80, 250, 60],
        "earnings_growth": [-0.05, -0.40, 0.10],
        "sector": ["Technology", "Technology", "Technology"],
    })


def test_score_value_bounded(sample_df):
    result = rs.score_value(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_support_bounded(sample_df):
    result = rs.score_support(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_support_near_low_scores_higher():
    df = pd.DataFrame({
        "pct_above_52w_low": [0.03, 0.35],  # اولی خیلی نزدیک کف، دومی دور
        "pct_vs_sma200": [0.0, 0.0],
    })
    result = rs.score_support(df)
    assert result.iloc[0] > result.iloc[1]


def test_score_stabilizing_bounded(sample_df):
    result = rs.score_stabilizing(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_stabilizing_penalizes_ongoing_freefall():
    """رگرسیون مفهومی مهم این اسکنر: سهمی که هنوز به‌شدت در حال سقوط است
    (mom_5d بسیار منفی) نباید امتیاز بالا بگیرد، حتی اگر RSI پایین باشد -
    این دقیقاً فیلتری است که از «گرفتن چاقوی در حال سقوط» جلوگیری می‌کند."""
    still_falling = pd.DataFrame({"rsi": [25], "made_new_low_last_3d": [True], "mom_5d": [-0.15]})
    stabilized = pd.DataFrame({"rsi": [25], "made_new_low_last_3d": [False], "mom_5d": [0.01]})
    score_falling = rs.score_stabilizing(still_falling).iloc[0]
    score_stable = rs.score_stabilizing(stabilized).iloc[0]
    assert score_stable > score_falling


def test_score_safety_bounded(sample_df):
    result = rs.score_safety(sample_df)
    assert (result >= 0).all() and (result <= 1).all()


def test_score_safety_debt_exemption_for_high_leverage_sector():
    """
    رگرسیون مستقیم روی باگ #۱ در README، این‌بار در سطح امتیازدهی (نه
    فقط فیلتر): وقتی is_high_leverage_sector=True است، امتیاز بدهی باید
    دقیقاً ۰.۵ (خنثی) باشد، صرف‌نظر از میزان بدهی واقعی.
    """
    df = pd.DataFrame({
        "profit_margin": [0.1, 0.1],
        "roe": [0.1, 0.1],
        "debt_to_equity": [500, 500],
        "sector": ["Financial Services", "Industrials"],
        "is_high_leverage_sector": [True, False],
        "earnings_growth": [0.05, 0.05],
    })
    result = rs.score_safety(df)
    # سهم معاف (بانک) باید امتیاز بالاتری از سهم غیرمعاف با همان بدهی بگیرد
    # چون بخش بدهی آن خنثی (۰.۵) می‌ماند، نه جریمه‌شده
    assert result.iloc[0] > result.iloc[1]


def test_score_safety_severe_earnings_decline_penalized():
    df = pd.DataFrame({"earnings_growth": [-0.50, -0.10, 0.05]})
    result = rs.score_safety(df)
    assert result.iloc[0] < result.iloc[1] < result.iloc[2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
