from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SetMetrics:
    precision: float
    recall: float
    f1: float
    intersection_size: int


def compute_set_metrics(
    expected: AbstractSet[T], predicted: AbstractSet[T]
) -> SetMetrics:
    intersection_size = len(expected & predicted)
    precision = intersection_size / len(predicted) if predicted else 0.0
    recall = intersection_size / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return SetMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        intersection_size=intersection_size,
    )
