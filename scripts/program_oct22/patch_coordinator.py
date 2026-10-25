"""Add a COORDINATOR stage and a proposer_style option to albench/arch_search/agent_loop.py.

Coordinator: the proposer over-generates (ideas_per_round, default 3 x per_round); a coordinator agent then chooses which per_round ideas are coded and trained.
Everything else (output layout, resume, accounting) is unchanged; unselected ideas get status 'skipped' and never reach the coder.
"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (s.count(a), a[:100])
    s = s.replace(a, b)


# ------------------------------------------------------------------ config
rep('''    reflector_noise_note: bool = True  # tell the reflector single-run val noise is ~0.008
''', '''    reflector_noise_note: bool = True  # tell the reflector single-run val noise is ~0.008
    select_per_round: int | None = None  # coordinator: how many ideas to keep per round; None -> --per_round
    proposer_style: str = "none"  # "explore" adds the bold/diverse paragraph of the single loop's --style explore to the proposer
''')
rep('''        for need, before in (("code", "propose"), ("evaluate", "code")):
            if need in st and (before not in st or st.index(before) > st.index(need)):
                raise AgentConfigError(f"stage {need!r} needs {before!r} earlier in {st}")''',
    '''        for need, before in (("code", "propose"), ("evaluate", "code"), ("coordinate", "propose")):
            if need in st and (before not in st or st.index(before) > st.index(need)):
                raise AgentConfigError(f"stage {need!r} needs {before!r} earlier in {st}")
        if "coordinate" in st and st.index("coordinate") > st.index("code"):
            raise AgentConfigError(f"stage 'coordinate' must come before 'code' in {st}")
        if self.proposer_style not in ("none", "explore"):
            raise AgentConfigError(f"proposer_style must be 'none' or 'explore', got {self.proposer_style!r}")''')
rep('''bad_roles = set(self.role_models) - {"proposer", "coder", "reflector"}''', '''bad_roles = set(self.role_models) - {"proposer", "coder", "reflector", "coordinator"}''')

# ------------------------------------------------------------------ number of ideas
rep('''    def n_ideas(self) -> int:
        n = self.cfg.ideas_per_round or self.args.per_round
        return max(1, min(n, self.target - self.midx))
''', '''    def n_select(self) -> int:
        """Models to train this round (what the coordinator keeps)."""
        return max(1, min(self.cfg.select_per_round or self.args.per_round, self.target - self.midx))

    def n_ideas(self) -> int:
        if "coordinate" in self.cfg.stages:  # over-generate; the coordinator cuts back to n_select()
            return max(self.n_select() + 1, self.cfg.ideas_per_round or 3 * self.args.per_round)
        n = self.cfg.ideas_per_round or self.args.per_round
        return max(1, min(n, self.target - self.midx))
''')

# ------------------------------------------------------------------ prompts + stage
rep('''PROPOSER_SETUP = (''', '''PROPOSER_EXPLORE = (
    "EXPLORATION STYLE: be bold. Spend most of your proposals on structurally different ideas that the history does not yet contain "
    "(different block types, receptive-field layouts, pooling, attention placement, heads, widths and depths), not small variations of the current best. "
    "Each proposal should differ from the others and from earlier ones in more than one design choice, while still being a model you expect to train well."
)

COORDINATOR_SYSTEM = """\\
You are the COORDINATOR in a team of agents that searches for neural-network architectures predicting \\
MPRA activity from 200-bp DNA. A proposer has suggested {n} candidate ideas this round and only {k} can \\
be trained. You choose which {k}. You never see test data; you see the validation results of everything \\
trained so far, a lessons note and the candidates.

The final product is an ENSEMBLE of all trained models, so a good pick is (a) likely to reach a high \\
validation Pearson r, (b) different from the models already trained and from your other picks (a \\
different mechanism, receptive field, pooling or head, not a width tweak), and (c) unlikely to fail \\
(shape errors, too slow for the time cap, unstable training, parameter limit). Drop duplicates of \\
earlier failures. Exploit what the history shows works, but keep at least one bet that tests something \\
untried when the history is thin.

Reply with exactly:
### SELECT: <{k} candidate numbers separated by commas>
### RATIONALE: <one short sentence per chosen candidate>"""


def parse_selection(text: str, n: int, k: int) -> list[int]:
    """0-based indices of the chosen candidates: 1-based numbers after '### SELECT:', duplicates and out-of-range numbers dropped, padded in proposal order."""
    m = re.search(r"###\\s*SELECT:\\s*([0-9][0-9,\\s;and]*)", text or "")
    chosen: list[int] = []
    if m:
        for tok in re.findall(r"\\d+", m.group(1)):
            i = int(tok) - 1
            if 0 <= i < n and i not in chosen:
                chosen.append(i)
    for i in range(n):  # fall back to proposal order for anything missing
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:k]


PROPOSER_SETUP = (''')
# proposer system additions
rep('''    return PROPOSER_SYSTEM.replace("{n}", str(n)), "\\n".join(parts)''', '''    system = PROPOSER_SYSTEM.replace("{n}", str(n))
    if ctx.cfg.proposer_style == "explore":
        system += "\\n\\n" + PROPOSER_EXPLORE
    if "coordinate" in ctx.cfg.stages:
        k = ctx.n_select()
        system += (
            f"\\n\\nA COORDINATOR agent will choose which {k} of your {n} proposals are trained, so give a spread of distinct "
            f"bets, each worth training, rather than {n} variants of one idea."
        )
    return system, "\\n".join(parts)''')
rep('''def coder_prompt(ctx: AgentCtx, item: dict) -> tuple[str, str]:''', '''def coordinator_prompt(ctx: AgentCtx, plan: dict) -> tuple[str, str]:
    n, k = len(plan["items"]), ctx.n_select()
    done = sorted(ctx.records, key=lambda m: -m.val)[:15]
    parts = ["TRAINED SO FAR (validation r, train seconds; best first):"]
    parts += [f" - val r {m.val:.4f}, {m.train_s:.0f}s, {m.n_params:,} params: {m.idea[:90]}" for m in done] or [" - nothing yet"]
    failed = [h_ for h_ in ctx.history if not h_.get("ok")][-8:]
    if failed:
        parts.append("\\nRECENT FAILURES:")
        parts += [f" - {h_['idea'][:80]}: {str(h_.get('reason', ''))[:100]}" for h_ in failed]
    if ctx.lessons:
        parts.append("\\nLESSONS (kept by a reflector from validation results; evidence, not rules):\\n" + ctx.lessons)
    parts.append(f"\\nCANDIDATES this round ({n}):")
    for i, it in enumerate(plan["items"], 1):
        parts.append(f"[{i}] IDEA: {it['idea']}\\n    WHY: {it['why']}\\n    DESIGN: {it['design'][:900]}\\n    TRAIN: {_fmt_recipe(it['train_spec'])}")
    parts.append(f"\\nChoose {k} of the {n} candidates now.")
    return COORDINATOR_SYSTEM.replace("{n}", str(n)).replace("{k}", str(k)), "\\n".join(parts)


@register_stage("coordinate")
def stage_coordinate(ctx: AgentCtx, plan: dict) -> None:
    if plan.get("coordinated"):
        return  # resumed inside this round
    rd, k = plan["round"], ctx.n_select()
    items = plan["items"]
    if len(items) <= k:
        plan["coordinated"] = True
        plan["coordinator"] = {"selected": list(range(len(items))), "text": "", "note": "no more candidates than slots"}
        save_plan(ctx.out_p, plan)
        return
    sys_p, user = coordinator_prompt(ctx, plan)
    text = ctx.llm.call("coordinator", rd, sys_p, user, n_candidates=len(items), k=k)
    chosen = parse_selection(text, len(items), k)
    for i, it in enumerate(items):
        if i not in chosen:
            it.update(status="skipped", reason="not selected by the coordinator", coordinator_skipped=True)
    plan["coordinated"] = True
    plan["coordinator"] = {"selected": chosen, "text": text[:2000]}
    save_plan(ctx.out_p, plan)
    ctx.say(f"  [round {rd}] coordinator: kept {[i + 1 for i in chosen]} of {len(items)}")


def coder_prompt(ctx: AgentCtx, item: dict) -> tuple[str, str]:''')
# report: do not list coordinator-skipped ideas as failures; mention the choice
rep('''        else:
            why = it["reason"] or it["status"]
            lines.append(f" - NOT TRAINED ({it['status']}): {why[:160]}; idea: {it['idea'][:80]}")''',
    '''        elif it.get("coordinator_skipped"):
            continue
        else:
            why = it["reason"] or it["status"]
            lines.append(f" - NOT TRAINED ({it['status']}): {why[:160]}; idea: {it['idea'][:80]}")''')
open(p, "w").write(s)
print("patched", p)
