#!/bin/bash
# Setup environment for ALBench-S2F on CSHL HPC
# Usage: source setup_env.sh

if [[ $(hostname) == *"bam"* || $(hostname) == *"gpu"* || $(hostname) == *"elzar"* ]]; then
    echo "Configuring for CSHL HPC..."
    
    # Load EasyBuild modules (Python, CUDA, etc.)
    if command -v module &> /dev/null; then
        module load EB5
    else
        # Try sourcing modules if function not present
        [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
        module load EB5
    fi

    # Add project bin to PATH (for HashFrag/BLAST if installed locally)
    export PATH=$PWD/external/hashFrag/src:$PWD/external/hashFrag:$PATH
    export PATH=$PWD/external/ncbi-blast-2.17.0+/bin:$PWD/external/ncbi-blast-2.16.0+/bin:$PATH
    
    # Check for BLAST
    if command -v blastn &> /dev/null; then
        echo "✓ BLAST+ found: $(blastn -version | head -1)"
    else
        echo "⚠ BLAST+ not found. Attempting to load module..."
        module load blast-plus 2>/dev/null || module load BLAST+ 2>/dev/null
    fi
fi

# General setup
export PATH=$PWD/scripts:$PATH
echo "Environment setup complete."
