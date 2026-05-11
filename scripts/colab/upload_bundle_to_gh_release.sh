#!/bin/bash
# Upload bundle_d20k.tar.gz + the notebook to a GitHub release.
# Creates the release if it doesn't exist; appends the assets.
#
# Prerequisites: `gh` CLI authenticated.

set -euo pipefail

REPO=trchristensen-99/ALBench-S2F
TAG=colab-d20k
TITLE="K562 D=20k MPRA subset (Colab notebook)"

# Make sure files exist
BUNDLE=$(realpath "$(dirname "$0")/bundle_d20k.tar.gz")
NB=$(realpath "$(dirname "$0")/k562_d20k_hpsearch.ipynb")
[ -f "$BUNDLE" ] || { echo "Missing $BUNDLE — run make_data_bundle.py first"; exit 1; }
[ -f "$NB" ]     || { echo "Missing $NB — run make_colab_notebook.py first"; exit 1; }

# Check if release exists; create if not, upload otherwise
if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
    echo "Release $TAG exists — uploading assets (overwriting if same name)"
    gh release upload "$TAG" "$BUNDLE" "$NB" -R "$REPO" --clobber
else
    echo "Creating release $TAG"
    gh release create "$TAG" "$BUNDLE" "$NB" \
        -R "$REPO" \
        --title "$TITLE" \
        --notes "Self-contained Colab notebook + data bundle. See README in tarball."
fi

echo ""
echo "Bundle URL: https://github.com/$REPO/releases/download/$TAG/$(basename $BUNDLE)"
echo "Notebook URL: https://github.com/$REPO/releases/download/$TAG/$(basename $NB)"
echo ""
echo "Open in Colab: https://colab.research.google.com/github/$REPO/blob/main/scripts/colab/k562_d20k_hpsearch.ipynb"
