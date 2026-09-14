# -*- coding: utf-8 -*-
"""tests/test_rank_bands.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bands import compute_rank_labels


def test_distinct_scores_get_distinct_ranks():
    labels = compute_rank_labels([0.90, 0.60, 0.30])
    assert labels == ["رتبه 1", "رتبه 2", "رتبه 3"]


def test_close_scores_grouped_into_band():
    labels = compute_rank_labels([0.90, 0.74, 0.72, 0.71, 0.55, 0.30], epsilon=0.03)
    assert labels[1] == labels[2] == labels[3] == "رتبه 2–4"
    assert labels[0] == "رتبه 1"
    assert labels[4] == "رتبه 5"


def test_empty_list():
    assert compute_rank_labels([]) == []


def test_single_score():
    assert compute_rank_labels([0.5]) == ["رتبه 1"]


def test_all_scores_within_epsilon_form_one_band():
    labels = compute_rank_labels([0.51, 0.50, 0.49], epsilon=0.03)
    assert len(set(labels)) == 1
    assert labels[0] == "رتبه 1–3"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
