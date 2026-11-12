import re

p = "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad/dispatcher.py"
s = open(p).read()

anchor = '    "REFMALINOISRECIPE_30K.json": ["malinois_e800_bs2048_adamw"],\n}'
assert s.count(anchor) == 1, "MIG_BAD anchor moved; re-read before patching"
assert "ARCHREP_10k.json" not in s, "already patched"

add = (
    '    "REFMALINOISRECIPE_30K.json": ["malinois_e800_bs2048_adamw"],\n'
    "    # The ARCHREP_ replicate menus re-use entries measured elsewhere, so they inherit those\n"
    "    # entries' slice behaviour under a NEW file name -- which is what MIG_BAD keys on. Each\n"
    "    # holds one DREAM-Attn recipe already known to OOM under its original menu. DREAM-RNN here\n"
    "    # is bs512 and Malinois bs1024, and only bs2048 has ever lost a slice for those two.\n"
    '    "ARCHREP_10k.json": ["dream_attn_e800_wd04_lr0003"],\n'
    '    "ARCHREP_30k.json": ["dream_attn_e500_bs512_adamw"],\n'
    '    "ARCHREP_100k.json": ["dream_attn_e800_wd02_lr0003"],\n'
    "}"
)
open(p, "w").write(s.replace(anchor, add))
print("MIG_BAD extended with the three ARCHREP menus")
