#!/usr/bin/env bash
set -euo pipefail
wavelet_tt_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
wavelet_shared_root="$(cd -- "$wavelet_tt_root/../wavelets" && pwd)"
wavelet_command=("$wavelet_tt_root/.venv/bin/python"
  "$wavelet_shared_root/benchmarks/run_campaign.py"
  --backend tenstorrent --config "$wavelet_shared_root/benchmark.toml"
  --native-binary "$wavelet_tt_root/build-bench/benchmarks/ttwt_benchmark"
  --runtime-root "$wavelet_tt_root/third_party/tt-metal"
  --results-root "$wavelet_tt_root/results" "$@")
wavelet_timing=wall
wavelet_arguments=("$@")
for (( wavelet_i=0; wavelet_i<${#wavelet_arguments[@]}; wavelet_i++ )); do
  if [[ "${wavelet_arguments[wavelet_i]}" == --dry-run || "${wavelet_arguments[wavelet_i]}" == --help ]]; then
    exec "${wavelet_command[@]}"
  fi
  if [[ "${wavelet_arguments[wavelet_i]}" == --timing-mode ]]; then
    wavelet_timing="${wavelet_arguments[wavelet_i+1]}"
  fi
done
mkdir -p "$wavelet_tt_root/logs"
wavelet_console="$(mktemp "$wavelet_tt_root/logs/$wavelet_timing-console-XXXXXX.log")"
printf '%s\n' "$wavelet_console" > "$wavelet_tt_root/logs/latest-$wavelet_timing-console"
set +e
"${wavelet_command[@]}" 2>&1 | tee "$wavelet_console"
wavelet_status=${PIPESTATUS[0]}
set -e
sed -n 's/^RUN_DIR=//p' "$wavelet_console" | tail -n 1 > "$wavelet_tt_root/logs/latest-$wavelet_timing-run"
if (( wavelet_status == 0 )); then
  wavelet_run="$(cat "$wavelet_tt_root/logs/latest-$wavelet_timing-run")"
  "$wavelet_tt_root/.venv/bin/python" "$wavelet_shared_root/benchmarks/process_campaign.py" check "$wavelet_run" --require-available
fi
exit "$wavelet_status"
