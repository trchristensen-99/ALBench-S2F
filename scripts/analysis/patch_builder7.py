# ruff: noqa
"""Seventh patch for build_justification_html.py: the Oct 27 cards, card patches, spec patches and the robustness table."""

import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (s.count(a), a[:90])
    s = s.replace(a, b)


rep(
    "    from ledger_oct26 import oct26_cards\n    return oct22_cards(d) + oct23_cards(d) + oct25_cards(d) + oct25b_cards(d) + oct26_cards(d) + cards\n",
    "    from ledger_oct26 import oct26_cards\n    from ledger_oct27 import oct27_cards, patch_cards\n\n    all_cards = oct22_cards(d) + oct23_cards(d) + oct25_cards(d) + oct25b_cards(d) + oct26_cards(d) + cards\n    all_cards += oct27_cards(d)\n    patch_cards(all_cards, d)\n    return all_cards\n",
)
rep(
    "    spec = spec_rows(d)\n",
    "    spec = spec_rows(d)\n    from ledger_oct27 import robustness_html, spec_patch\n\n    spec_patch(spec, d)\n",
)
rep('coverage_html(d) + "</section>")', 'coverage_html(d) + robustness_html(d) + "</section>")')
rep(
    '"Results from the Oct 25-26 program",',
    '"Results from the Oct 25-26 program", "Results from the Oct 27 program",',
)
rep(
    "combined by a non-negative ElasticNet ensemble fitted on validation predictions.",
    "combined by a non-negative ElasticNet ensemble fitted on validation predictions (the oracle-relabelled validation set, V0c).",
)
rep(
    "The coordinator stage (+0.005 on genomic and random 30k) pools to about +0.002 across the grid, below the 0.005 a change must clear.",
    "The coordinator stage pools to about +0.003 across the grid with no detectable heterogeneity, still below the 0.005 a change must clear.",
)
rep(
    "CHANGELOG = [\n",
    'CHANGELOG = [\n    ("2026-10-27 20:00", "Added the Oct 27 results: the core reservoir set per dataset size, which loop elements earn their place (separate coder, coordinator against a random pick, reflector, guidance), consistency of the coordinator effect across sizes and reservoirs, the extrapolation under error-based metrics, training-recipe sensitivity (patience, epochs), the oracle-relabelled validation set adopted as standard and tested at 10k and 100k, HepG2 and the differential at 10k and 100k, reduced menus and ensemble methods across reservoirs, a 600k recipe retune, and a table of how far each assumption has been tested across sizes and reservoirs."),\n',
)
open(p, "w").write(s)
print("patched", p)
