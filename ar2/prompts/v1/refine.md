{{core}}
{{family_view}}
{{lessons_view_family}}
{{failure_view}}
ASSIGNMENT: make ONE structural change to parent_model.py that you expect to raise validation r.
Angle for this round: {{angle}}.
- One change only, so the result can be attributed. Two are allowed only if one cannot work
without the other; say so in the card.
- Do not change training settings; the tuner handles those. Do not simply rescale width or depth
unless the angle is capacity.
- Use the diagnostics. A large gap between train r and validation r calls for less capacity or
more regularisation in the architecture. A small gap with low train r calls for more capacity
or a better inductive bias.
- Do not repeat a change listed under EARLIER CHANGES.
- State your hypothesis as a prediction that the result could contradict.
{{extra_constraints}}
{{speed_note}}
OUTPUT: write the full model.py and a card.json with what_changed and hypothesis filled in.
