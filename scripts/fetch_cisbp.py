"""Fetch the CIS-BP human PFMs and convert them to MEME format.

CIS-BP has no direct download URL for its bulk archives: the bulk page POSTs a form
to ``bulk_archive.php``, the server generates a timestamped zip under ``tmp/``, and
the response is an HTML page containing a link to it. This script performs that
two-step exchange, then converts the per-motif PWM files into a single MEME file.

WHY CIS-BP RATHER THAN JASPAR. JASPAR CORE has ~700 human matrices; CIS-BP covers
far more human TFs, which matters because our motif arms are limited by how many of
the ChIP'd factors we have a matrix for -- with JASPAR only ~57 of the 292 TFs
ChIP'd in both K562 and HepG2 survive into the vocabulary.

MOTIF STATUS. CIS-BP marks each motif as Directly determined (D) or Inferred (I)
from a homologous TF. Inferred motifs are real information but weaker evidence, so
``--status`` controls which are kept and the default is direct-only.

Usage:
    python scripts/fetch_cisbp.py                     # download, convert, install
    python scripts/fetch_cisbp.py --archive local.zip # skip the download
    python scripts/fetch_cisbp.py --status all        # include inferred motifs
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://cisbp.ccbr.utoronto.ca/"
BULK = BASE + "bulk_archive.php"
UA = {"User-Agent": "Mozilla/5.0 (albench data fetcher)"}


def request_archive(species: str, timeout: int) -> str:
    """POST the bulk form and return the URL of the generated archive."""
    fields = [
        ("selSpec", species),
        ("Spec[]", "PWMs"),
        ("Spec[]", "TF_Information"),
        ("submit", "Download Species Archive!"),
    ]
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(BULK, data=data, headers=UA)
    print(f"  POST {BULK}  selSpec={species}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf8", "replace")
    m = re.search(r"href=[\"']((?:tmp/)?[^\"']+\.zip)[\"']", html)
    if not m:
        snippet = " ".join(re.sub(r"<[^>]+>", " ", html).split())[:300]
        raise RuntimeError(
            "CIS-BP did not return an archive link. The form may have changed.\n"
            f"Response text began: {snippet}"
        )
    return urllib.parse.urljoin(BASE, urllib.parse.quote(m.group(1), safe="/:"))


def download(url: str, dest: Path, timeout: int) -> Path:
    print(f"  GET  {url}")
    req = urllib.request.Request(url, headers=UA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            total += len(chunk)
    print(f"  saved {total / 1e6:.1f} MB -> {dest}")
    if not zipfile.is_zipfile(dest):
        raise RuntimeError(f"{dest} is not a zip -- CIS-BP may have returned an error page")
    return dest


def parse_tf_info(zf: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    """Motif_ID -> (TF name, status D/I). Prefers the first row seen per motif."""
    name = next((n for n in zf.namelist() if n.endswith("TF_Information.txt")), None)
    if name is None:
        raise RuntimeError("archive has no TF_Information.txt")
    out: dict[str, tuple[str, str]] = {}
    with zf.open(name) as fh:
        header = fh.readline().decode("utf8", "replace").rstrip("\n").split("\t")
        col = {c: i for i, c in enumerate(header)}
        need = ("Motif_ID", "TF_Name", "TF_Status")
        missing = [c for c in need if c not in col]
        if missing:
            raise RuntimeError(f"TF_Information.txt lacks columns {missing}; has {header[:12]}")
        for raw in fh:
            p = raw.decode("utf8", "replace").rstrip("\n").split("\t")
            if len(p) <= max(col[c] for c in need):
                continue
            mid = p[col["Motif_ID"]].strip()
            if not mid or mid == "." or mid in out:
                continue
            out[mid] = (p[col["TF_Name"]].strip() or mid, p[col["TF_Status"]].strip())
    return out


_INFLATING = re.compile(r"inflating:\s*\S*?([A-Za-z0-9_.\-]+)\.txt", re.I)


def _rows_from_block(lines: list[str]) -> list[list[float]] | None:
    """One ``Pos A C G T`` block -> rows of [A,C,G,T], renormalised."""
    head = next((i for i, ln in enumerate(lines) if ln.startswith("Pos")), None)
    if head is None:
        return None
    cols = lines[head].split("\t")
    try:
        idx = [cols.index(b) for b in ("A", "C", "G", "T")]
    except ValueError:
        return None
    rows = []
    for ln in lines[head + 1 :]:
        p = ln.split("\t")
        if len(p) <= max(idx):
            continue
        try:
            vals = [float(p[i]) for i in idx]
        except ValueError:
            continue
        t = sum(vals)
        if t <= 0:
            return None
        rows.append([v / t for v in vals])
    return rows or None


def read_all_pwms(zf: zipfile.ZipFile) -> dict[str, list[list[float]]]:
    """Motif_ID -> PWM rows, handling both layouts CIS-BP has served.

    The current bulk archive packs every matrix into ONE member named
    ``PWMs_all_motifs`` which is the captured stdout of an unzip run: each matrix is
    preceded by a line like ``inflating: pwms_all_motifs/M00099_3.10.txt``. That is a
    server-side packaging quirk rather than a documented format, so the older layout
    (a directory of per-motif .txt files) is still handled as a fallback.
    """
    out: dict[str, list[list[float]]] = {}

    members = [n for n in zf.namelist() if "pwm" in n.lower() and n.lower().endswith(".txt")]
    if members:  # per-file layout
        for n in members:
            rows = _rows_from_block(zf.read(n).decode("utf8", "replace").splitlines())
            if rows:
                out[Path(n).stem] = rows
        return out

    blob = next((n for n in zf.namelist() if "pwms" in n.lower()), None)
    if blob is None:
        raise RuntimeError(f"no PWM member found; archive has {zf.namelist()}")
    lines = zf.read(blob).decode("utf8", "replace").splitlines()
    starts = [(i, m.group(1)) for i, ln in enumerate(lines) if (m := _INFLATING.search(ln))]
    if not starts:
        raise RuntimeError(
            f"{blob} has neither per-file members nor 'inflating:' markers; "
            "the CIS-BP packaging may have changed"
        )
    for j, (i, mid) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        rows = _rows_from_block(lines[i:end])
        if rows:
            out[mid] = rows
    return out


def convert(archive: Path, out: Path, status: str, min_len: int) -> int:
    kept = skipped_empty = skipped_status = 0
    seen_tf: set[str] = set()
    with zipfile.ZipFile(archive) as zf, open(out, "w") as w:
        info = parse_tf_info(zf)
        print(f"  TF_Information: {len(info):,} motif records")
        pwms = read_all_pwms(zf)
        print(f"  PWM matrices parsed: {len(pwms):,}")

        w.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        w.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for mid, rows in sorted(pwms.items()):
            tf, st = info.get(mid, (mid, "D"))
            if status == "direct" and st.upper().startswith("I"):
                skipped_status += 1
                continue
            if len(rows) < min_len:
                skipped_empty += 1
                continue
            # One matrix per TF: CIS-BP lists many near-identical motifs per factor,
            # and the vocabulary clusters anyway, so keeping all of them only slows
            # the O(n^2) clustering without adding binding preferences.
            if tf in seen_tf:
                continue
            seen_tf.add(tf)
            w.write(f"MOTIF {mid} {tf}\n")
            w.write(f"letter-probability matrix: alength= 4 w= {len(rows)} nsites= 20 E= 0\n")
            for r in rows:
                w.write("  " + "  ".join(f"{v:.6f}" for v in r) + "\n")
            w.write("\n")
            kept += 1
    if kept == 0:
        # Do not leave a header-only file behind: `albench doctor` would then report
        # the asset as present and the failure would resurface later as a parse error.
        out.unlink(missing_ok=True)
        print("\n  wrote NOTHING; removed the empty output file")
        return 0
    print(f"\n  wrote {kept:,} motifs ({len(seen_tf):,} distinct TFs) -> {out}")
    print(f"  skipped {skipped_empty:,} empty/short, {skipped_status:,} inferred-status")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="Homo_sapiens")
    ap.add_argument("--archive", default=None, help="use a local zip instead of downloading")
    ap.add_argument("--out", default=None, help="default: <ALBENCH_DATA>/motifs/CISBP_human.meme")
    ap.add_argument(
        "--status",
        choices=("direct", "all"),
        default="direct",
        help="direct = only directly-determined motifs (default)",
    )
    ap.add_argument("--min-len", type=int, default=4, help="drop matrices shorter than this")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from albench.paths import ASSETS, data_root

    out = Path(args.out) if args.out else data_root() / ASSETS["cisbp_meme"].default_relpath
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.archive:
        archive = Path(args.archive)
        if not archive.exists():
            raise SystemExit(f"--archive not found: {archive}")
        print(f"using local archive {archive}")
    else:
        print(f"requesting CIS-BP archive for {args.species} ...")
        url = request_archive(args.species, args.timeout)
        archive = download(url, out.parent / f"cisbp_{args.species}.zip", args.timeout)

    n = convert(archive, out, args.status, args.min_len)
    if not args.archive and not args.keep_archive:
        archive.unlink(missing_ok=True)
    if n == 0:
        print("\nNo motifs written -- check --status and the archive contents.", file=sys.stderr)
        return 1
    print(
        "\nalbench will now prefer CIS-BP over JASPAR automatically "
        "(see albench/motifs/vocabulary.py::_default_meme)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
