#!/usr/bin/env bash
# v_long: same arch as v1, 1.5x training epochs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${HERE}/archive"
mkdir -p "$ARCHIVE_DIR"
PIPELINE_EPOCHS_SCALE=1.5 python3 "${HERE}/compress.py" --batch-size 4 "$@"
cd "$ARCHIVE_DIR"
zip -0 "${HERE}/archive.zip" model.pt.br mask.obu.br pose.npy.br
