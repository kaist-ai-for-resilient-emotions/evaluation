#!/bin/bash

set -euo pipefail

THERAPIST_MODELS=("gpt-chat" "gpt-normal" "gpt-safety")
CLIENT_MODEL="fast"
PERSONAS_DIR="$(dirname "$0")/personas"
OUTPUT_DIR="transcriptions_generated"
MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
persona_files=("${PERSONAS_DIR}"/*.txt)
shopt -u nullglob

if [[ ${#persona_files[@]} -eq 0 ]]; then
  echo "No persona files found in ${PERSONAS_DIR}" >&2
  exit 1
fi

declare -a JOB_PIDS=()
declare -A JOB_DESC=()

wait_for_slot() {
  local max_jobs=$1
  while (( $(jobs -rp | wc -l) >= max_jobs )); do
    sleep 0.5
  done
}

launch_job() {
  local persona="$1"
  local therapist="$2"
  local output_file="${OUTPUT_DIR}/${therapist}.${CLIENT_MODEL}.${persona}.log"

  if [[ -f "${output_file}" ]]; then
    echo "[skip] ${output_file}"
    return
  fi

  echo "[run ] therapist=${therapist} client=${CLIENT_MODEL} persona=${persona}"
  (
    PYTHONUNBUFFERED=1 python3 gen_virtual.py "${persona}" \
      --model "${therapist}" \
      --client-model "${CLIENT_MODEL}" \
      | stdbuf -o0 tee "${output_file}"
  ) &

  local pid=$!
  JOB_PIDS+=("${pid}")
  JOB_DESC["${pid}"]="${therapist}/${persona}"
}

for persona_path in "${persona_files[@]}"; do
  persona_name="$(basename "${persona_path}" .txt)"
  for therapist_model in "${THERAPIST_MODELS[@]}"; do
    wait_for_slot "${MAX_JOBS}"
    launch_job "${persona_name}" "${therapist_model}"
  done
done

exit_code=0
for pid in "${JOB_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[fail] ${JOB_DESC[${pid}]}" >&2
    exit_code=1
  else
    echo "[done] ${JOB_DESC[${pid}]}"
  fi
done

exit "${exit_code}"
