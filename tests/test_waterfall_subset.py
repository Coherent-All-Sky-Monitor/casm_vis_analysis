"""
run_waterfall reads only the sub-triangle of the plotted inputs.

The reduced array plus the rank remapping must select exactly the same
baselines, with the same conjugation, as the full upper-triangular array did.
Baseline ordering follows casm_io/docs/CONVENTIONS.md.
"""

import numpy as np

from casm_io.correlator.baselines import (
    build_input_subset_plan,
    triu_flat_index,
)


def test_rank_remap_selects_the_same_cells():
    nsig = 12
    n_bl = nsig * (nsig + 1) // 2
    rng = np.random.default_rng(0)
    full = (rng.normal(size=(2, 3, n_bl)) +
            1j * rng.normal(size=(2, 3, n_bl))).astype(np.complex64)

    packet_indices = [7, 2, 9, 2]  # unsorted, with a duplicate
    sel = sorted(set(packet_indices))
    bl_idx, bl_conj, sel_plan = build_input_subset_plan(sel, nsig)
    assert sel_plan == sel
    assert not bl_conj.any()

    sub = full[:, :, bl_idx]
    rank = {p: r for r, p in enumerate(sel)}
    ranks = [rank[p] for p in packet_indices]
    m = len(sel)

    for i, pi in enumerate(packet_indices):
        for j, pj in enumerate(packet_indices):
            # plot_waterfall's cell lookup, full array vs sub-triangle
            lo, hi = min(pi, pj), max(pi, pj)
            expected = full[:, :, triu_flat_index(nsig, lo, hi)]
            conj_full = pi > pj

            ri, rj = ranks[i], ranks[j]
            rlo, rhi = min(ri, rj), max(ri, rj)
            got = sub[:, :, triu_flat_index(m, rlo, rhi)]
            conj_sub = ri > rj

            assert conj_sub == conj_full
            assert np.array_equal(got, expected)


def test_subset_is_smaller():
    nsig = 128
    sel = list(range(0, 19))
    bl_idx, _, _ = build_input_subset_plan(sel, nsig)
    assert len(bl_idx) == 19 * 20 // 2
    assert len(bl_idx) < nsig * (nsig + 1) // 2 // 40
