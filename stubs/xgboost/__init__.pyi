from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

class XGBClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = ...,
        max_depth: int = ...,
        learning_rate: float = ...,
        scale_pos_weight: float | None = ...,
        use_label_encoder: bool = ...,
        eval_metric: str | None = ...,
    ) -> None: ...
    def fit(
        self,
        X: NDArray[np.float64] | NDArray[np.float32] | Sequence[Sequence[float]],
        y: NDArray[np.float64] | NDArray[np.int64] | Sequence[int | float],
    ) -> XGBClassifier: ...
    def predict_proba(
        self,
        X: NDArray[np.float64] | NDArray[np.float32] | Sequence[Sequence[float]],
    ) -> NDArray[np.float64]: ...
