#!/usr/bin/env bash
set -euo pipefail
wavelet_tt_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
wavelet_architecture="${1:-wormhole}"
if (( $# )); then shift; fi
if [[ "$wavelet_architecture" != wormhole && "$wavelet_architecture" != blackhole ]]; then
  echo 'Usage: overnight.sh wormhole|blackhole' >&2
  exit 2
fi
wavelet_board=auto
if [[ "$wavelet_architecture" == wormhole ]]; then wavelet_board=n150; fi
"$wavelet_tt_root/benchmarks/run_campaign.sh" --architecture "$wavelet_architecture" \
  --device-model "$wavelet_board" --timing-mode wall \
  --dispatch-mode normal_fast_dispatch "$@"
wavelet_run="$(cat "$wavelet_tt_root/logs/latest-wall-run")"
"$wavelet_tt_root/benchmarks/run_campaign.sh" --architecture "$wavelet_architecture" \
  --device-model "$wavelet_board" --timing-mode device_profile \
  --dispatch-mode normal_fast_dispatch --resume "$wavelet_run"
