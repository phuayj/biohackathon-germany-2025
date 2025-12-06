from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

class IsotonicRegression:
    X_thresholds_: NDArray[np.float64]
    y_thresholds_: NDArray[np.float64]

    def __init__(self, *, out_of_bounds: str | None = ...) -> None: ...
    def fit(
        self,
        X: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
        y: NDArray[np.float64] | NDArray[np.int64] | Sequence[float | int],
    ) -> IsotonicRegression: ...
    def predict(
        self, T: NDArray[np.float64] | NDArray[np.float32] | Sequence[float]
    ) -> NDArray[np.float64]: ...
