"""
Approximation-scheme benchmark for the EV route search (tickets T11/T12).

``VehicleParams.soc_levels`` is the discretization knob.  Finer grids approach the
true continuous optimum but settle more states; coarser grids are cheaper to
search, but the ``ceil`` rounding in ``drive_levels`` wastes more battery per leg,
which inflates the realized cost.  This module sweeps ``soc_levels`` against a
fine-grained baseline and reports the runtime-vs-accuracy tradeoff.

Runtime is measured purely by the deterministic counters the search already
returns -- ``states_settled`` and ``heap_pushes`` -- never wall-clock time, so the
benchmark is reproducible.  Accuracy is the empirical (1 + epsilon) factor,
``cost_dollars / baseline_cost``; cost is in real dollars and therefore already
comparable across grids (``level_kwh = capacity / soc_levels`` scales correctly).
"""
import sys
from dataclasses import dataclass, replace
from typing import Optional

from ev_model import VehicleParams
from dijkstra_ev import SearchResult, dijkstra_ev
from load_graph import load_graph


@dataclass
class ApproxRow:
    """One ``soc_levels`` setting and the cost/runtime it produced."""
    levels: int
    reached: bool
    cost_dollars: float
    time_h: float
    states_settled: int
    heap_pushes: int
    approx_ratio: float       # cost_dollars / baseline_cost; inf when unreached
    route_names: list[str]    # chosen path, to spot when coarsening reroutes


@dataclass
class ApproxResult:
    """A full ``soc_levels`` sweep against one fine-grained baseline."""
    baseline_levels: int
    baseline_cost: float
    rows: list[ApproxRow]     # sorted by levels ascending


def _route_names(G, route: list[int]) -> list[str]:
    return [G.nodes[n]["name"] for n in route]


def _row_for(
    G,
    source: int,
    target: int,
    params: VehicleParams,
    levels: int,
    baseline_cost: float,
    cached: Optional[SearchResult] = None,
) -> ApproxRow:
    # Reuse the baseline search for its own row so the ratio is exactly 1.0.
    res = cached if cached is not None else dijkstra_ev(
        G, source, target, replace(params, soc_levels=levels)
    )
    ratio = (
        res.total_cost_dollars / baseline_cost
        if res.reached and baseline_cost > 0
        else float("inf")
    )
    return ApproxRow(
        levels=levels,
        reached=res.reached,
        cost_dollars=res.total_cost_dollars,
        time_h=res.total_time_h,
        states_settled=res.states_settled,
        heap_pushes=res.heap_pushes,
        approx_ratio=ratio,
        route_names=_route_names(G, res.route),
    )


def run_approximation(
    G,
    source: int = 0,
    target: int = 4,
    levels_list: Optional[list[int]] = None,
    baseline_levels: int = 40,
    params: Optional[VehicleParams] = None,
) -> ApproxResult:
    """Sweep ``soc_levels`` over ``levels_list`` and rate each against the baseline.

    The baseline is a single fine-grained search at ``baseline_levels`` treated as
    ground truth.  For every L in ``levels_list`` a fresh search runs with
    ``soc_levels=L`` (all other params preserved via ``dataclasses.replace``); the
    L == baseline_levels row reuses the baseline run, so its ratio is exactly 1.0.
    Rows are returned sorted by levels ascending.
    """
    if params is None:
        params = VehicleParams()
    if levels_list is None:
        levels_list = [4, 6, 8, 10, 12, 16, 20, 30, 40]

    baseline = dijkstra_ev(
        G, source, target, replace(params, soc_levels=baseline_levels)
    )
    baseline_cost = baseline.total_cost_dollars

    rows = [
        _row_for(
            G, source, target, params, L, baseline_cost,
            cached=baseline if L == baseline_levels else None,
        )
        for L in levels_list
    ]
    rows.sort(key=lambda r: r.levels)

    return ApproxResult(
        baseline_levels=baseline_levels,
        baseline_cost=baseline_cost,
        rows=rows,
    )


def _summary_line(result: ApproxResult) -> str:
    within = [r for r in result.rows if r.reached and r.approx_ratio <= 1.05]
    if not within:
        return "  No soc_levels setting came within 5% of the baseline optimum."

    best = min(within, key=lambda r: r.levels)
    baseline_row = next(
        (r for r in result.rows if r.levels == result.baseline_levels), None
    )
    if baseline_row is None:
        return (
            f"  Within 5% of optimum: soc_levels={best.levels} "
            f"(ratio {best.approx_ratio:.4f}); no baseline row for a state delta."
        )

    fewer = baseline_row.states_settled - best.states_settled
    return (
        f"  Within 5% of optimum: soc_levels={best.levels} "
        f"(ratio {best.approx_ratio:.4f}) settles {fewer} fewer states than "
        f"baseline L={result.baseline_levels} "
        f"({best.states_settled} vs {baseline_row.states_settled})."
    )


def print_approximation_report(result: ApproxResult, route_label: str) -> None:
    """Print the sweep as a table, styled like ``print_itinerary``."""
    sep  = "=" * 66
    sep2 = "-" * 66

    print(f"\n{sep}")
    print(f"  Approximation benchmark: {route_label}")
    print(
        f"  baseline soc_levels = {result.baseline_levels}  "
        f"(cost ${result.baseline_cost:.2f} = ground-truth optimum)"
    )
    print("  runtime proxies: states_settled / heap_pushes (no wall-clock)")
    print(sep)

    fmt = "  {lv:>6}  {rc:>7}  {cost:>9}  {ratio:>13}  {settled:>9}  {pushes:>8}"
    print(fmt.format(
        lv="levels", rc="reached", cost="cost $",
        ratio="ratio (1+eps)", settled="settled", pushes="pushes",
    ))
    print(fmt.format(
        lv="-" * 6, rc="-" * 7, cost="-" * 9,
        ratio="-" * 13, settled="-" * 9, pushes="-" * 8,
    ))

    for r in result.rows:
        print(fmt.format(
            lv=r.levels,
            rc="yes" if r.reached else "no",
            cost=f"{r.cost_dollars:.2f}" if r.reached else "--",
            ratio=f"{r.approx_ratio:.4f}" if r.reached else "--",
            settled=r.states_settled,
            pushes=r.heap_pushes,
        ))

    print()
    print("  Chosen path per soc_levels (route_names recorded per row):")
    for r in result.rows:
        if r.reached:
            path = " -> ".join(r.route_names)
            print(
                f"    L={r.levels:>2}  ${r.cost_dollars:>6.2f}  "
                f"ratio {r.approx_ratio:>7.4f}   {path}"
            )
        else:
            print(f"    L={r.levels:>2}  unreached (no path)")

    print(f"\n{sep2}")
    print(_summary_line(result))
    print(f"{sep2}\n")


def empirical_epsilon(aggregate: ApproxResult) -> dict[int, float]:
    """Map each ``soc_levels`` to its measured epsilon = ``approx_ratio - 1.0``.

    This is the empirical (1 + epsilon) inflation at that coarseness.  Unreached
    settings keep their ``inf`` ratio and so map to ``inf``.
    """
    return {r.levels: r.approx_ratio - 1.0 for r in aggregate.rows}


def theoretical_bound_note() -> str:
    """Plain-language intuition for why coarsening inflates cost by (1 + epsilon).

    Returned as a string (not printed) so callers can place it where they like.
    This is the intuition behind the empirical curve, not a formal proof.
    """
    return (
        "  Why coarsening inflates cost (intuition, not a proof):\n"
        "    Each drive leg rounds its energy UP to a whole battery level via\n"
        "    ceil() in drive_levels(), so a single leg wastes at most one level\n"
        "    of charge: level_kwh = battery_capacity_kwh / soc_levels.  Coarser\n"
        "    grids (smaller soc_levels) make level_kwh bigger, so the worst-case\n"
        "    wasted-then-repaid charge per leg grows -- and that waste is exactly\n"
        "    the (1 + epsilon) cost inflation.  A path with k drive legs carries\n"
        "    at most ~k * level_kwh of slack, which shrinks as soc_levels rises,\n"
        "    matching the empirical epsilon trending toward 0 as the grid refines."
    )


def _knee_line(aggregate: ApproxResult, baseline_states: Optional[int]) -> str:
    # accuracy-per-state = (fraction of states saved) / (epsilon + C); the small
    # constant C keeps the baseline (epsilon = 0) from dividing by zero and lets
    # a near-exact-but-barely-cheaper grid lose to a much cheaper, slightly-worse one.
    C = 0.01
    if not baseline_states:
        return "  Knee: baseline state count unavailable."

    best, best_score = None, -1.0
    for r in aggregate.rows:
        if not r.reached:
            continue
        saved_frac = (baseline_states - r.states_settled) / baseline_states
        score = saved_frac / ((r.approx_ratio - 1.0) + C)
        if score > best_score:
            best, best_score = r, score

    if best is None:
        return "  Knee: no reached setting to recommend."

    saved_pct = (baseline_states - best.states_settled) / baseline_states * 100.0
    eps_pct   = (best.approx_ratio - 1.0) * 100.0
    return (
        f"  Knee (best accuracy-per-state): soc_levels={best.levels}  ->  "
        f"+{eps_pct:.1f}% cost for -{saved_pct:.0f}% states  (score {best_score:.1f})."
    )


def print_tradeoff_report(aggregate: ApproxResult) -> None:
    """Print epsilon vs state-savings per setting, styled like ``print_itinerary``."""
    sep  = "=" * 66
    sep2 = "-" * 66

    eps = empirical_epsilon(aggregate)
    baseline_row = next(
        (r for r in aggregate.rows if r.levels == aggregate.baseline_levels), None
    )
    baseline_states = baseline_row.states_settled if baseline_row else None

    print(f"\n{sep}")
    print("  Runtime vs accuracy tradeoff")
    if baseline_states is not None:
        print(
            f"  baseline soc_levels = {aggregate.baseline_levels} "
            f"({baseline_states} states settled, cost ${aggregate.baseline_cost:.2f})"
        )
    print(sep)

    fmt = "  {lv:>6}  {ratio:>12}  {eps:>10}  {states:>14}"
    print(fmt.format(
        lv="levels", ratio="approx_ratio", eps="cost eps", states="states vs base",
    ))
    print(fmt.format(
        lv="-" * 6, ratio="-" * 12, eps="-" * 10, states="-" * 14,
    ))

    for r in aggregate.rows:
        if r.reached:
            ratio_s = f"{r.approx_ratio:.4f}"
            eps_s   = f"{eps[r.levels] * 100:+.1f}%"
            if baseline_states:
                saved = (baseline_states - r.states_settled) / baseline_states * 100.0
                states_s = "0%" if saved < 0.05 else f"-{saved:.0f}%"
            else:
                states_s = "n/a"
            note = ""
        else:
            ratio_s, eps_s, states_s = "--", "--", "n/a"
            note = "  (unreached, no valid route)"
        print(fmt.format(lv=r.levels, ratio=ratio_s, eps=eps_s, states=states_s) + note)

    print(f"\n{sep2}")
    print(_knee_line(aggregate, baseline_states))
    print(f"{sep2}")
    print()
    print(theoretical_bound_note())
    print()


if __name__ == "__main__":
    G = load_graph()

    SOURCE, TARGET = 0, 4
    LEVELS_LIST = [4, 6, 8, 10, 12, 16, 20, 30, 40]
    BASELINE_LEVELS = 40

    result = run_approximation(
        G, source=SOURCE, target=TARGET,
        levels_list=LEVELS_LIST, baseline_levels=BASELINE_LEVELS,
    )

    if result.baseline_cost <= 0:
        print(
            f"Baseline route {SOURCE} -> {TARGET} unreachable at "
            f"L={BASELINE_LEVELS}; cannot benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)

    src = G.nodes[SOURCE]["name"]
    dst = G.nodes[TARGET]["name"]
    print_approximation_report(result, route_label=f"{src} -> {dst}")
    print_tradeoff_report(result)
