#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

mapfile -t slurm_files < <(find . -maxdepth 1 -type f -name "*.slurm" | sort)

if [[ ${#slurm_files[@]} -eq 0 ]]; then
  echo "No .slurm files found in $SCRIPT_DIR"
  exit 1
fi

echo "Found ${#slurm_files[@]} slurm files in $SCRIPT_DIR"

for f in "${slurm_files[@]}"; do
  job_file="${f#./}"
  if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] sbatch $job_file"
  else
    echo "Submitting $job_file"
    sbatch "$job_file"
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run complete. No jobs were submitted."
else
  echo "All submissions finished."
fi
