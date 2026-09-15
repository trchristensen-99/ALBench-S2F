"""Domain-blind machinery: the AL loop, pools, the registry, asset paths, run config.

Nothing in this package may import from a specific study (MPRA, yeast, a particular
oracle or student). That rule is what makes a new setting a matter of adding a domain
rather than editing the harness. See docs/STRUCTURE.md.
"""
