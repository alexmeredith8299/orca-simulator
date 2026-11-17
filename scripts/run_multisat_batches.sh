#!/usr/bin/env bash
# Run the multi-satellite fault batches (tx_fade and the magnetometer faults) and their
# detectors. Usage: scripts/run_multisat_batches.sh OUT_DIR [WORKERS]
#
# Run from a clean checkout of branch multisat-faults, with the conda env active
# (conda activate orca; pip install -e '.[detect-ml,detect-dl]'). Keep OUT_DIR outside the repo,
# and do not edit params/ or src/ while it runs: every seed reloads params/ from disk.
# Finished seeds are skipped on a rerun; add FORCE=1 to redo them.
# Set WITH_NEGATIVES=1 to also re-simulate the spec-fault (F1..F10) negatives that the
# posterior needs, with the new nav, RSSI and magnetometer columns.
set -euo pipefail

OUT=${1:?usage: $0 OUT_DIR [WORKERS]}
WORKERS=${2:-60}
SEEDS=0:200       # simulate 200 seeds per batch
TUNE=0:70         # detector calibration seeds
TEST=100:200      # scoring seeds
FORCE_FLAG=${FORCE:+--force}

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "working tree is dirty; commit first so the manifests record a clean commit" >&2
    exit 1
fi
mkdir -p "$OUT"

# batch name : fault : onset shape
BATCHES=(
    "tx_fade_step:tx_fade:step"
    "tx_fade_ramp:tx_fade:ramp"
    "mag_bias_step:mag_bias:step"
    "mag_bias_ramp:mag_bias:ramp"
    "mag_noise_ramp:mag_noise:ramp"
    "mag_noise_intermittent:mag_noise:intermittent"
    "mag_coupling_step:mag_coupling:step"
    "mag_coupling_ramp:mag_coupling:ramp"
)

# detectors per fault
detectors_for() {
    case "$1" in
        tx_fade) echo "mcdn_txfade_sig,mcdn_txfade_sig_comms" ;;
        mag_*)   echo "mcdn_mag,mcdn_mag_sig,mcdn_mag_sig_comms" ;;
    esac
}

echo "== simulation =="
for spec in "${BATCHES[@]}"; do
    IFS=: read -r name fault profile <<<"$spec"
    echo "-- $name ($(date +%T))"
    expand3 run --seeds "$SEEDS" --fault "$fault" --fault-set extended --profile "$profile" \
        --workers "$WORKERS" --out "$OUT/$name" $FORCE_FLAG
done

if [[ -n "${WITH_NEGATIVES:-}" ]]; then
    echo "-- spec negatives ($(date +%T))"
    expand3 run --seeds "$SEEDS" --fault-set spec --workers "$WORKERS" \
        --out "$OUT/spec_negatives" $FORCE_FLAG
fi

echo "== detection =="
for spec in "${BATCHES[@]}"; do
    IFS=: read -r name fault profile <<<"$spec"
    echo "-- $name ($(date +%T))"
    expand3 detect "$OUT/$name" --detector "$(detectors_for "$fault")" \
        --tune "$TUNE" --seeds "$TEST" --workers "$WORKERS" \
        --out "$OUT/$name/detect/multisat"
done

echo "done $(date +%T): results in $OUT/<batch>/detect/multisat"
