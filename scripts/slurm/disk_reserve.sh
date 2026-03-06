#!/bin/bash
# Manually create or remove the disk space reservation.
#
# Usage:
#   scripts/slurm/disk_reserve.sh claim   # Reserve 5GB placeholder
#   scripts/slurm/disk_reserve.sh release # Free the placeholder
#   scripts/slurm/disk_reserve.sh status  # Show current state

PLACEHOLDER="/grid/wsbs/home_norepl/christen/.disk_placeholder.dat"
SIZE="5G"

case "${1:-status}" in
    claim|create)
        if [ -f "$PLACEHOLDER" ]; then
            echo "Reservation already exists: $(du -h "$PLACEHOLDER" | cut -f1)"
        else
            fallocate -l "$SIZE" "$PLACEHOLDER" && \
                echo "Created ${SIZE} reservation at $PLACEHOLDER" || \
                echo "ERROR: fallocate failed (disk full?)"
        fi
        ;;
    release|free)
        if [ -f "$PLACEHOLDER" ]; then
            rm -f "$PLACEHOLDER"
            echo "Released reservation"
        else
            echo "No reservation to release"
        fi
        ;;
    status)
        if [ -f "$PLACEHOLDER" ]; then
            echo "Reservation ACTIVE: $(du -h "$PLACEHOLDER" | cut -f1)"
        else
            echo "Reservation INACTIVE (space not claimed)"
        fi
        df -h /grid/wsbs/home_norepl/christen/ | tail -1
        ;;
    *)
        echo "Usage: $0 {claim|release|status}"
        exit 1
        ;;
esac
