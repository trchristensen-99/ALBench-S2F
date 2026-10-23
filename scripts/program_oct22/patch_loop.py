"""Add (1) a harness-level precision switch AL_PRECISION (fp32 default, tf32 / fp16 / bf16) and (2) --style prompt options (explore / reflect / explore_reflect)
to albench/arch_search/arch_autoresearch.py (and its scratchpad copy). Defaults are byte-identical to the old behaviour."""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (s.count(a), a[:90])
    s = s.replace(a, b)


# ------------------------------------------------------------------ precision (harness-level: proposals cannot choose it)
rep("""    lossf = torch.nn.MSELoss()
    Xva_t = torch.from_numpy(Xva).to(dev); yva_t = torch.from_numpy(yva).to(dev)
""", """    lossf = torch.nn.MSELoss()
    # Precision is set by the harness for every model alike (env AL_PRECISION); a proposal cannot opt in or out, so no model gets a private speed-up.
    prec = os.environ.get("AL_PRECISION", "fp32").lower()
    amp_on = prec in ("fp16", "bf16") and dev.type == "cuda"
    amp_dt = torch.float16 if prec == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (prec == "fp16" and dev.type == "cuda") else None
    if prec == "tf32" and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    Xva_t = torch.from_numpy(Xva).to(dev); yva_t = torch.from_numpy(yva).to(dev)
""")
rep("""            opt.zero_grad()
            out = model(xb).squeeze()
            # RC consistency: average fwd + reverse-complement prediction in the loss
            out_rc = model(rc(xb)).squeeze()
            loss = 0.5 * (lossf(out, yb) + lossf(out_rc, yb))
            if not torch.isfinite(loss):
                return None, {"error": f"non-finite loss ep{ep}"}, used
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
""", """            opt.zero_grad()
            with torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
                out = model(xb).squeeze()
                # RC consistency: average fwd + reverse-complement prediction in the loss
                out_rc = model(rc(xb)).squeeze()
            loss = 0.5 * (lossf(out.float(), yb) + lossf(out_rc.float(), yb))
            if not torch.isfinite(loss):
                return None, {"error": f"non-finite loss ep{ep}"}, used
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
""")
rep("""        with torch.no_grad():
            vp = torch.cat([(model(Xva_t[i:i+512]).squeeze() + model(rc(Xva_t[i:i+512])).squeeze()) / 2
                            for i in range(0, len(Xva_t), 512)])
            vmse = float(((vp - yva_t) ** 2).mean())""", """        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
            vp = torch.cat([(model(Xva_t[i:i+512]).squeeze() + model(rc(Xva_t[i:i+512])).squeeze()) / 2
                            for i in range(0, len(Xva_t), 512)]).float()
            vmse = float(((vp - yva_t) ** 2).mean())""")

# ------------------------------------------------------------------ prompt styles
rep('''def _round_note(args, rd, cap_min):''', '''STYLE_EXPLORE = ("EXPLORATION STYLE: be bold. Spend most of your proposals on structurally different ideas that the history does not yet contain "
                 "(different block types, receptive-field layouts, pooling, attention placement, heads, widths and depths), not small variations of the current best. "
                 "Each proposal should differ from the others and from earlier ones in more than one design choice, while still being a model you expect to train well.")
STYLE_REFLECT = ("REFLECTION STYLE: before the proposals, write one short paragraph that starts with the word REFLECTION: and states what the results so far show about which design "
                 "choices raise or lower validation Pearson r (cite specific earlier ideas and their scores) and which hypotheses are still untested. Then give the proposals, "
                 "chosen to test those hypotheses.")


def _style_parts(args, lessons):
    st = getattr(args, "style", "none")
    parts = []
    if "explore" in st:
        parts.append(STYLE_EXPLORE)
    if "reflect" in st:
        parts.append(STYLE_REFLECT)
        if lessons:
            parts.append("YOUR EARLIER REFLECTIONS (oldest first):\\n" + "\\n".join(f"- {x}" for x in lessons[-5:]))
    return parts


def _round_note(args, rd, cap_min, lessons=()):''')
rep('''    if args.guidance != "known_good":
        return (ctx + "\\n\\n" + base) if ctx else base
''', '''    sp = _style_parts(args, lessons)
    if args.guidance != "known_good":
        return "\\n\\n".join(([ctx] if ctx else []) + sp + [base])
''')
rep('''    parts.append(base)
    return "\\n\\n".join(parts)''', '''    parts += sp
    parts.append(base)
    return "\\n\\n".join(parts)''')
rep('''    ap.add_argument("--resume", action="store_true")''', '''    ap.add_argument("--style", default="none", choices=["none", "explore", "reflect", "explore_reflect"])  # extra prompt persona; none = unchanged prompt
    ap.add_argument("--resume", action="store_true")''')
rep('''    history, midx, t_start = [], 0, time.time()
''', '''    history, midx, t_start = [], 0, time.time()
    lessons = []
    if args.resume and (Path(args.out_dir) / "reflections.jsonl").exists():
        lessons = [json.loads(l)["text"] for l in open(Path(args.out_dir) / "reflections.jsonl") if l.strip()]
''')
rep('''note=_round_note(args, rd, cap_min))''', '''note=_round_note(args, rd, cap_min, lessons))''')
rep('''            text = call_model(sys_p, user)
            props = parse_proposals(text)''', '''            text = call_model(sys_p, user)
            if "reflect" in args.style:
                import re as _re
                _m = _re.search(r"REFLECTION:\\s*(.+?)(?:\\n\\s*\\n|\\Z)", text, _re.S)
                if _m:
                    lessons.append(_m.group(1).strip()[:900])
                    with open(Path(args.out_dir) / "reflections.jsonl", "a") as _fh:
                        _fh.write(json.dumps({"round": rd, "text": lessons[-1]}) + "\\n")
            props = parse_proposals(text)''')
open(p, "w").write(s)
print("patched", p)
