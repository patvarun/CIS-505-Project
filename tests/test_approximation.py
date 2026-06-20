"""
Focused tests for approximation.py.

Same harness as tests/test_monte_carlo.py: the project root goes on ``sys.path``
for imports, and the ``graph`` fixture loads the graph with cwd pinned to that
root (``load_graph`` reads ``data/*.csv`` relative to the working directory).

The benchmark's accuracy curve is KNOWN to be non-monotonic -- L=16 (ratio 1.087)
beats L=20 (ratio 1.130) on the identical route, and coarse grids (L <= 8) reroute
structurally -- so these tests encode the true two-regime properties rather than
naive per-step monotonicity.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from load_graph import load_graph
from approximation import empirical_epsilon, run_approximation

SOURCE, TARGET = 0, 4
LEVELS_LIST = [4, 6, 8, 10, 12, 16, 20, 30, 40]
BASELINE_LEVELS = 40


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def graph():
    # load_graph() resolves data/*.csv relative to cwd; pin it to the project root.
    prev = os.getcwd()
    os.chdir(_PROJECT_ROOT)
    try:
        return load_graph()
    finally:
        os.chdir(prev)


@pytest.fixture(scope="session")
def result(graph):
    return run_approximation(
        graph, source=SOURCE, target=TARGET,
        levels_list=LEVELS_LIST, baseline_levels=BASELINE_LEVELS,
    )


@pytest.fixture(scope="session")
def by_level(result):
    return {r.levels: r for r in result.rows}


# --------------------------------------------------------------------------- #
# 1. Baseline exactness                                                         #
# --------------------------------------------------------------------------- #
def test_baseline_ratio_is_exactly_one(result, by_level):
    assert by_level[result.baseline_levels].approx_ratio == 1.0


# --------------------------------------------------------------------------- #
# 2. Optimum is a lower bound: no coarser grid beats the fine baseline on cost  #
# --------------------------------------------------------------------------- #
def test_baseline_is_lower_bound(result):
    for r in result.rows:
        if r.reached:
            assert r.approx_ratio >= 1.0 - 1e-9, (
                f"L={r.levels} ratio {r.approx_ratio} < 1.0: a coarser grid beat "
                f"the fine baseline on dollar cost -- real finding, investigate."
            )


# --------------------------------------------------------------------------- #
# 3. State monotonicity: states_settled is non-decreasing as soc_levels rises   #
# --------------------------------------------------------------------------- #
def test_states_settled_monotonic_in_levels(result):
    rows = sorted(result.rows, key=lambda r: r.levels)
    for lo, hi in zip(rows, rows[1:]):
        assert lo.states_settled <= hi.states_settled, (
            f"states_settled dropped going finer: L={lo.levels} "
            f"({lo.states_settled}) -> L={hi.levels} ({hi.states_settled})"
        )


# --------------------------------------------------------------------------- #
# 4. Trend, not strict order: coarse half is worse ON AVERAGE than fine half    #
# --------------------------------------------------------------------------- #
def test_accuracy_trend_coarse_worse_on_average(result):
    reached = sorted((r for r in result.rows if r.reached), key=lambda r: r.levels)
    half = len(reached) // 2
    coarse = reached[:half]                 # lower soc_levels
    fine = reached[len(reached) - half:]    # higher soc_levels

    avg_coarse = sum(r.approx_ratio for r in coarse) / len(coarse)
    avg_fine = sum(r.approx_ratio for r in fine) / len(fine)
    assert avg_coarse >= avg_fine, (
        f"coarse-half avg ratio {avg_coarse:.4f} < fine-half avg {avg_fine:.4f}"
    )


# --------------------------------------------------------------------------- #
# 5. Reroute detection: coarse grids reroute, every L>=10 matches the baseline  #
# --------------------------------------------------------------------------- #
def test_two_regime_reroute(result, by_level):
    baseline_route = by_level[result.baseline_levels].route_names

    coarse_rerouted = [
        L for L in (6, 8)
        if L in by_level and by_level[L].route_names != baseline_route
    ]
    assert coarse_rerouted, (
        "expected at least one of L in {6, 8} to take a structurally different "
        "route than the baseline"
    )

    for r in result.rows:
        if r.levels >= 10:
            assert r.route_names == baseline_route, (
                f"L={r.levels} diverged from the baseline route: {r.route_names}"
            )


# --------------------------------------------------------------------------- #
# 6. Reachability regimes: L=4 too coarse (unreached), every L>=6 reached        #
# --------------------------------------------------------------------------- #
def test_reachability_regimes(by_level):
    assert not by_level[4].reached, "L=4 expected unreached (too-coarse regime)"
    for L in LEVELS_LIST:
        if L >= 6:
            assert by_level[L].reached, f"L={L} expected to reach the target"


# --------------------------------------------------------------------------- #
# Bonus: empirical_epsilon maps each level to (approx_ratio - 1.0)               #
# --------------------------------------------------------------------------- #
def test_empirical_epsilon_matches_ratio(result):
    eps = empirical_epsilon(result)
    assert set(eps) == {r.levels for r in result.rows}
    for r in result.rows:
        if r.reached:
            assert eps[r.levels] == pytest.approx(r.approx_ratio - 1.0)
    assert eps[result.baseline_levels] == 0.0
