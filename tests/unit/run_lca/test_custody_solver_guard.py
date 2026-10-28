"""Simulated external solver failure on an admitted rounded waste-trade cycle."""

from fractions import Fraction

import numpy as np
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from scipy.sparse import csc_matrix

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.run_lca.calculation import custody
from pytoce.run_lca.calculation.providers import MaterialCarbon


def test_custody_rejects_simulated_negative_solver_return_after_rounded_trade_cycle(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a simulated negative return where rounded transfers lose the positivity premise.

    This tests the dependency-return guard. The fixed ten-region trade construction checks the
    rounded
    positive-input feedback cycle. The native solve must return finite values
    before one negative result is substituted. The simulated return is finite.
    """
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    family = state.materials.index("HDPE")
    good = state.goods.index("Packaging")
    state.annual_inflow[1, :10, good] = 1.0
    state.stock_by_cohort[1, 1, :10, good] = 1.0
    state.outflow_by_cohort[2, 1, :10, good] = 1.0
    # For Q=2**54 Mt, the rounding interval is 0 < collected Mt < 2. One Mt supplies the
    # deterministic positive cohort origin.
    trade = np.asarray((0.25, 0.75))[:, None] * 2**54
    inputs.waste_imports[2, :, :10, family, good] = trade
    inputs.waste_exports[2, :, :10, family, good] = trade
    mfa = evaluate_mfa(state, inputs, recipe)
    real_splu = custody.splu
    solved_with_substitution: list[bool] = []

    class SimulatedNegativeSolve:
        """Factor and solve the real matrix before substituting one external return value."""

        def __init__(self, matrix: csc_matrix) -> None:
            """Keep the actual admitted system and its unmodified SuperLU factorization."""
            self.matrix = matrix
            self.factorization = real_splu(matrix)

        def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
            """Substitute only after observing the ten-region rounded feedback block."""
            actual = self.factorization.solve(right_hand_side)
            entries = self.matrix.tocoo()
            feedback = entries.data == -0.1
            if not np.any(feedback):
                solved_with_substitution.append(False)
                return actual
            rows, columns = entries.row[feedback], entries.col[feedback]
            assert len(rows) == 10 and len(set(columns)) == 1
            market = int(columns[0])
            assert all(self.matrix[market, int(region)] == -1.0 for region in rows)
            # Ten stored fl(1/10) coefficients sum exactly to 1 + 2**-54.
            # The positive-input cycle therefore lacks a nonnegative inverse.
            assert sum(Fraction(-float(value)) for value in entries.data[feedback]) > 1
            assert np.isfinite(right_hand_side).all() and np.any(right_hand_side[:, 0] > 0)
            assert np.isfinite(actual).all()
            simulated = actual.copy()
            simulated[market, 0] = -1.0
            solved_with_substitution.append(True)
            return simulated

    # Patch the imported external factory only; graph construction and both
    # native factorization and solve still execute through their real owners.
    monkeypatch.setattr(custody, "splu", SimulatedNegativeSolve)
    trace = None
    with pytest.raises(
        ArithmeticError,
        match=r"^Constituent mixing returned invalid origin amounts in 2020\.$",
    ):
        trace = custody.trace_carbon_custody(mfa, carbon, recipe)
    assert trace is None
    assert solved_with_substitution == [False, True]
