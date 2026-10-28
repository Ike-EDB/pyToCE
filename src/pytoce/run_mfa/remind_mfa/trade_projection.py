"""Reuse native historical trade fits in the prospective trade calculation."""

from typing import Protocol

import flodym as fd  # type: ignore[import-untyped]
import numpy as np


class ProportionalRegression(Protocol):
    """Public interface of the REMIND-MFA 0.2.0 proportional regression model."""

    def __init__(
        self,
        *,
        data_to_extrapolate: np.ndarray,
        predictor_values: np.ndarray,
        weights: np.ndarray,
        independent_dims: tuple[int, ...],
    ) -> None:
        """Admit historical responses, full predictors and independent coordinates."""

    @property
    def fit_prms(self) -> np.ndarray:
        """Expose the native fitted parameters after a successful regression."""
        ...

    def extrapolate(self) -> np.ndarray:
        """Fit the native model and evaluate it with its historical restoration."""
        ...

    @staticmethod
    def func(predictor: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """Evaluate the native proportional law using already fitted parameters."""
        ...


class HistoricalTradeProjection:
    """Retain one exact historical fit per market and trade side in an MFA session.

    Scenario admission fixes history through 2019. Demand and lifetime changes
    alter the future predictors. Intermediate and manufactured trade retain
    their historical good shares before fitting. Native SciPy fitting supplies
    the regression coefficients and tolerance.
    """

    def __init__(self, regression: type[ProportionalRegression]) -> None:
        """Bind the admitted public native model and initially absent fitted batches."""
        self._regression = regression
        self._coefficients: dict[tuple[str, str], np.ndarray] = {}

    def project(
        self, key: tuple[str, str], historic: fd.FlodymArray, scaler: fd.FlodymArray
    ) -> fd.FlodymArray:
        """Project regional trade in Mt from the unchanged fitted historical relation.

        Each non-time coordinate has its own native proportional coefficient.
        The last observed nonnegative trade and fitted nonnegative prediction
        form the source's equally weighted geometric blend. Restore historical
        quantities afterward, preserving its operation and coordinate order.
        """
        dimensions = historic.dims.union_with(scaler.dims).get_subset(
            ("t",) + historic.dims.letters[1:]
        )
        predictor = scaler.cast_to(dimensions)
        historical_values = np.asarray(historic.values)
        predictor_values = np.asarray(predictor.values)
        historical_count = historic.shape[0]
        coefficients = self._coefficients.get(key)
        if coefficients is None:
            # Only the five latest historical observations have positive weight.
            weights = np.maximum(0.0, np.arange(-historical_count, 0) + 6)
            weights /= weights.sum()
            model = self._regression(
                data_to_extrapolate=historical_values,
                predictor_values=predictor_values,
                weights=np.broadcast_to(
                    weights.reshape((historical_count,) + (1,) * (historical_values.ndim - 1)),
                    historic.shape,
                ),
                independent_dims=tuple(range(1, dimensions.ndim)),
            )
            model.extrapolate()
            coefficients = model.fit_prms.copy()
            coefficients.setflags(write=False)
            self._coefficients[key] = coefficients
        prediction = self._regression.func(predictor_values, np.moveaxis(coefficients, -1, 0))
        prediction[:historical_count] = historical_values
        nonnegative = fd.FlodymArray(dims=dimensions, values=prediction).maximum(0.0)
        last = historic[historic.dims["h"].items[-1]].maximum(0.0).cast_to(dimensions)
        projected = last**0.5 * nonnegative**0.5
        projected[{"t": historic.dims["h"]}] = historic
        return projected
