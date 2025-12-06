from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

class LogisticRegression:
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]

    def __init__(
        self,
        *,
        C: float = ...,
        class_weight: Mapping[int, float] | str | None = ...,
        max_iter: int = ...,
        solver: str = ...,
    ) -> None: ...
    def fit(
        self,
        X: NDArray[np.float64] | NDArray[np.float32] | Sequence[Sequence[float]],
        y: NDArray[np.float64] | NDArray[np.int64] | Sequence[int | float],
    ) -> LogisticRegression: ...
    def predict_proba(
        self,
        X: NDArray[np.float64] | NDArray[np.float32] | Sequence[Sequence[float]],
    ) -> NDArray[np.float64]: ...
