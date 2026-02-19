#!/usr/bin/env bash
set -euo pipefail

# Edit this list directly.
TRAYS=(1 10 20 30)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANGLE_JSON="${SCRIPT_DIR}/angle.json"

if [ ! -f "${ANGLE_JSON}" ]; then
  echo "angle.json not found: ${ANGLE_JSON}"
  exit 1
fi

cd "${PROJECT_ROOT}"

get_cv_angle() {
  local tray_num="$1"
  python - "${tray_num}" "${ANGLE_JSON}" <<'PY'
import json
import pathlib
import sys

tray_num = str(sys.argv[1])
angle_path = pathlib.Path(sys.argv[2])
data = json.loads(angle_path.read_text(encoding="utf-8"))

saved = None
for item in data.get("angles", []):
    if str(item.get("tray_num")) == tray_num:
        saved = float(item.get("angle"))
        break

if saved is None:
    raise SystemExit(f"No angle found for tray_num={tray_num} in {angle_path}")

# Match web pipeline behavior: OpenCV uses inverted sign.
print(-saved)
PY
}

if [ "${#TRAYS[@]}" -eq 0 ]; then
  echo "TRAYS is empty. Edit app/run_trayseg_build_template.sh and set TRAYS=(...)."
  exit 1
fi

for tray in "${TRAYS[@]}"; do
  cv_angle="$(get_cv_angle "${tray}")"
  echo "=================================================="
  echo "[build_template] tray=${tray}, manual_angle=${cv_angle}"
  echo "=================================================="
  python -m app.angle.tray_seg \
    --mode build_template \
    --tray_num "${tray}" \
    --manual_angle "${cv_angle}"
done

echo "All build_template jobs completed."
