"""
rank_bands.py
ماژول کوچک: نمایش «رتبه» به‌جای «امتیاز قطعی» وقتی تفاوت امتیاز بین
سهام مجاور ناچیز است.

مشکل: نمایش «امتیاز: ۰.۷۴» القا می‌کند دقت بالایی دارد، در حالی که
تفاوت ۰.۷۴ و ۰.۷۱ (که هر دو از جمع وزنی چند رتبه‌بندی صدکی به‌دست
آمده‌اند) می‌تواند کاملاً در نویز داده باشد. نمایش رتبه به‌صورت باند
(مثلاً «رتبه ۳ تا ۵») صادقانه‌تر است: یعنی «این سهم‌ها عملاً هم‌سطح‌اند»
به‌جای یک ترتیب قطعی کاذب.

منطق ساده است: امتیازها باید نزولی مرتب باشند (همان ترتیب نمایش).
هر دو سهم مجاور که فاصله امتیازشان کمتر از epsilon باشد در یک باند
مشترک قرار می‌گیرند.
"""

from __future__ import annotations

from collections import defaultdict

DEFAULT_EPSILON = 0.03


def compute_rank_labels(scores: list[float], epsilon: float = DEFAULT_EPSILON) -> list[str]:
    """
    scores: لیست امتیازها، از قبل نزولی مرتب‌شده (همان ترتیبی که نمایش
    داده می‌شود). خروجی: یک برچسب رتبه به همان طول ورودی، مثل
    "رتبه ۱" یا "رتبه ۳–۵" برای سهامی که در یک باند مشترک قرار گرفته‌اند.
    """
    n = len(scores)
    if n == 0:
        return []

    band_id = [0] * n
    for i in range(1, n):
        gap = scores[i - 1] - scores[i]
        band_id[i] = band_id[i - 1] if gap <= epsilon else band_id[i - 1] + 1

    band_positions: dict[int, list[int]] = defaultdict(list)
    for idx, b in enumerate(band_id):
        band_positions[b].append(idx + 1)  # رتبه ۱-پایه

    labels = []
    for b in band_id:
        positions = band_positions[b]
        if len(positions) == 1:
            labels.append(f"رتبه {positions[0]}")
        else:
            labels.append(f"رتبه {positions[0]}–{positions[-1]}")
    return labels


# ===========================================================================
# نمونه استفاده (در write_html_report هر سه اسکنر)
# ===========================================================================
"""
from rank_bands import compute_rank_labels

scores = df["total_score"].tolist()
rank_labels = compute_rank_labels(scores)

for (ticker, row), rank_label in zip(df.iterrows(), rank_labels):
    # به‌جای فقط "امتیاز کل: 0.74/1.00"، نمایش بده:
    # f"{rank_label} · امتیاز: {row['total_score']:.2f}/1.00"
    ...
"""
