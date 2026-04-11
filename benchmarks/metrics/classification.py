"""Classification metrics: Precision, Recall, F1."""

from __future__ import annotations

from typing import Any


def precision_recall_f1(
    detected: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    match_fn: Any = None,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for detection benchmarks.

    Args:
        detected: List of detected findings.
        expected: List of expected findings (ground truth).
        match_fn: Callable(detected_item, expected_item) -> bool.
                  Defaults to matching on 'object_id' key.
    """
    if match_fn is None:
        def match_fn(d: dict, e: dict) -> bool:
            return d.get("object_id") == e.get("object_id")

    matched_expected: set[int] = set()
    true_positives = 0

    for det in detected:
        for i, exp in enumerate(expected):
            if i not in matched_expected and match_fn(det, exp):
                true_positives += 1
                matched_expected.add(i)
                break

    precision = true_positives / len(detected) if detected else 1.0
    recall = true_positives / len(expected) if expected else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
