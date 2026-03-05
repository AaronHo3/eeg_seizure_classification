#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://physionet.org/files/chbmit/1.0.0"
OUT_DIR="${1:-data/raw/chb-mit}"
shift || true

if [ "$#" -eq 0 ]; then
  PATIENTS=(chb01 chb02 chb03)
else
  PATIENTS=("$@")
fi

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

download_with_wget() {
  local patient="$1"
  wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" "${BASE_URL}/${patient}/"
}

download_with_curl() {
  local patient="$1"
  local listing
  listing="$(curl -fsSL "${BASE_URL}/${patient}/")"

  if [ -z "$listing" ]; then
    echo "Failed to read listing for ${patient}" >&2
    return 1
  fi

  mkdir -p "$patient"
  echo "$listing" \
    | grep -Eo 'href="[^"]+\.(edf|txt)"' \
    | sed -E 's/href="([^"]+)"/\1/' \
    | while read -r fname; do
        [ -z "$fname" ] && continue
        curl -fL "${BASE_URL}/${patient}/${fname}" -o "${patient}/${fname}"
      done
}

if command -v wget >/dev/null 2>&1; then
  FETCHER="wget"
elif command -v curl >/dev/null 2>&1; then
  FETCHER="curl"
else
  echo "Neither wget nor curl is installed." >&2
  exit 1
fi

echo "Output directory: $OUT_DIR"
echo "Patients: ${PATIENTS[*]}"
echo "Fetcher: $FETCHER"

for patient in "${PATIENTS[@]}"; do
  echo "Downloading ${patient}..."
  if [ "$FETCHER" = "wget" ]; then
    download_with_wget "$patient"
  else
    download_with_curl "$patient"
  fi
  echo "Completed ${patient}."
  ls -1 "$patient" | head -n 5 || true
  echo "---"
done

echo "Download complete."
