#!/usr/bin/env bash
# End-to-end demo on synthetic data with a known answer key (labels come from the key).
# Use it to check the install. It says nothing about the real data.
set -euo pipefail
export PYTHONPATH=src
C="--config config.synthetic.json"
python -m cel make-synthetic $C
python -m cel prepare $C
python -m cel link $C
python -m cel label-sheet $C --labels labels_synth --force
python -m cel oracle-label $C --labels labels_synth
python -m cel evaluate $C --labels labels_synth
python -m cel cluster $C
python -m cel export-graph $C
python -m cel summarize $C --use-truth
