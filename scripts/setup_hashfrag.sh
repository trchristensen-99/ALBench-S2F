#!/bin/bash
# Set up HashFrag and BLAST+ for K562 homology-aware splits.
#
# HashFrag creates homology-aware data splits by:
# 1. Using BLAST to find similar sequences
# 2. Computing Smith-Waterman alignment scores
# 3. Clustering homologous sequences together
# 4. Ensuring clusters don't span train/val/test splits

set -euo pipefail

echo "=== Setting up HashFrag ==="
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check/install BLAST+
if ! command -v blastn &> /dev/null; then
    echo "BLAST+ not found. Attempting to install..."
    if command -v apt-get &> /dev/null; then
        echo "Installing via apt-get..."
        sudo apt-get update && sudo apt-get install -y ncbi-blast+
    elif command -v brew &> /dev/null; then
        echo "Installing via Homebrew..."
        brew install blast
    elif command -v module &> /dev/null; then
        echo "Attempting to load BLAST+ module..."
        module load blast-plus 2>/dev/null || module load BLAST+ 2>/dev/null || \
            module load ncbi-blast+ 2>/dev/null || {
                echo "ERROR: Could not find BLAST+ module."
                echo "Try: module avail | grep -i blast"
                echo "Then: module load <module_name>"
                exit 1
            }
    else
        echo "ERROR: Cannot install BLAST+ automatically."
        echo ""
        echo "Install manually from:"
        echo "  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
        echo ""
        echo "Or on supported systems:"
        echo "  Ubuntu: sudo apt-get install ncbi-blast+"
        echo "  macOS:  brew install blast"
        echo "  HPC:    module load blast-plus"
        exit 1
    fi
fi

echo "✓ BLAST+ found: $(blastn -version 2>/dev/null | head -1)"
echo ""

# Clone HashFrag if not present
mkdir -p external

if [ -d "external/hashFrag" ]; then
    echo "✓ HashFrag already cloned at external/hashFrag"
    cd external/hashFrag
    echo "  Commit: $(git rev-parse --short HEAD)"
    cd "$PROJECT_ROOT"
else
    echo "Cloning HashFrag repository..."
    git clone https://github.com/de-Boer-Lab/hashFrag.git external/hashFrag
    echo "✓ HashFrag cloned"
fi

# Make executable
if [ -f "external/hashFrag/src/hashFrag" ]; then
    chmod +x external/hashFrag/src/hashFrag
    echo "✓ HashFrag executable"
fi

echo ""
echo "=== HashFrag Setup Complete ==="
echo ""
echo "To use in current shell:"
echo "  export PATH=\"\$PATH:$(pwd)/external/hashFrag/src\""
echo ""
echo "To create K562 splits:"
echo "  python scripts/create_hashfrag_splits.py"
echo ""
