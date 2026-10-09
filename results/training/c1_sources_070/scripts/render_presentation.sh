#!/usr/bin/env bash
# Render a PPTX with the privately installed LibreOffice runtime.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
office_bin="$project_root/.tools/libreoffice/root/opt/libreoffice25.2/program/soffice"
input_path="${1:?Usage: scripts/render_presentation.sh INPUT.pptx [OUTPUT_DIRECTORY]}"
input_path="$(realpath "$input_path")"
output_dir="${2:-$(dirname "$input_path")/office_render}"
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"

if [[ ! -x "$office_bin" ]]; then
  echo "Private LibreOffice runtime missing: $office_bin" >&2
  exit 1
fi

profile_dir="$(mktemp -d "${TMPDIR:-/tmp}/fusion-office-profile.XXXXXX")"
trap 'rm -rf "$profile_dir"' EXIT
SAL_USE_VCLPLUGIN=svp "$office_bin" \
  "-env:UserInstallation=file://$profile_dir" \
  --headless --nologo --nodefault --nofirststartwizard \
  --convert-to pdf:impress_pdf_Export --outdir "$output_dir" "$input_path"

pdf_path="$output_dir/$(basename "${input_path%.*}").pdf"
test -s "$pdf_path"
echo "$pdf_path"
