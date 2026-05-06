#!/bin/bash
# Download Sahu et al. 2022 (GSE180158) STARR-seq supplementary data
# from NCBI GEO. Targets the random N170/N150 library files which are
# what we actually need (truly random sequences with measured episomal
# activities — for the CpG-vs-activity question).
#
# Strategy: list the GEO supplementary directory, identify
# small-to-medium files (processed counts, oligo design tables) and
# skip raw FASTQs (each ~10-50GB). The Nature supplementary tables
# would be downloaded separately if needed.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
OUT_DIR="$REPO/external/sahu_geo"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

GEO=GSE180158
BASE_FTP="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/${GEO}/suppl"

echo "=== Listing $BASE_FTP ==="
curl -s "$BASE_FTP/" -o suppl_listing.html
# Extract file links + sizes
python3 - <<'PY'
import re
from pathlib import Path
html = Path("suppl_listing.html").read_text()
# Apache-style listing: <a href="filename">filename</a> ... size
matches = re.findall(r'<a href="([^"?][^"]+)">([^<]+)</a>\s*</td><td[^>]*>[^<]+</td><td[^>]*>\s*([0-9.A-Z]+)', html)
print(f"Found {len(matches)} entries")
for href, name, size in matches[:60]:
    print(f"  {size:>10}  {name}")
# Save list of non-FASTQ small/medium files
with open("download_targets.txt", "w") as f:
    for href, name, size in matches:
        # Skip raw FASTQ which are huge
        is_fastq = ".fastq" in name.lower() or ".fq" in name.lower()
        is_dir = href.endswith("/")
        if is_dir or is_fastq:
            continue
        f.write(href + "\n")
print(f"\nSaved {sum(1 for _ in open('download_targets.txt'))} non-FASTQ files to download_targets.txt")
PY

echo
echo "=== Downloading non-FASTQ files ==="
n=0
while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ -f "$f" ]; then
        echo "  [skip] $f already present"
        continue
    fi
    echo "  fetching $f ..."
    curl -sf "${BASE_FTP}/${f}" -o "$f" || echo "    FAIL $f"
    n=$((n + 1))
done < download_targets.txt
echo
echo "=== Downloaded $n files ==="
ls -lah "$OUT_DIR" | head -30
