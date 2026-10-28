{{core}}
{{coverage_view}}
{{do_not_repeat_view}}
{{failure_view}}
{{taxonomy}}
ASSIGNMENT: propose 10 candidate architectures as one-line sketches. Do not write model code.
- Cover at least 6 different families. At least 4 sketches must be in families marked empty or
holding fewer than 2 candidates.
- At most 2 sketches may be variations of a family that already holds 5 or more candidates.
- For each sketch give the mechanism and why it should help on this data, not just a name.
- Do not repeat anything in the do-not-repeat list.
- You may draw on recent machine learning work outside genomics where the mechanism suits short,
fixed-length sequences.
{{forced_families}}
{{extra_constraints}}
OUTPUT: write sketches.json, a list of 10 objects with keys:
family, name, mechanism, why_for_mpra, nearest_existing, difference, recipe_needs
