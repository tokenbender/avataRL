#!/usr/bin/env bash
set -euo pipefail

# Common Weights & Biases metadata
export WANDB_PROJECT=${WANDB_PROJECT:-nanogpt-avatarl}
export EXPERIMENT_GROUP=${EXPERIMENT_GROUP:-pretrain_size_sweep}

log() {
  printf '\n[%-24s] %s\n' "${1}" "${2}"
}

run_regular() {
  local cfg="$1"
  local path="experiments/pretrain/${cfg}/config.py"
  log "regular" "Starting ${cfg}"
  EXPERIMENT_NAME="${cfg}" python train.py "${path}"
}

run_avatarl() {
  local cfg="$1"
  local path="experiments/pretrain/${cfg}/config.py"
  log "avatarl" "Starting ${cfg}"
  EXPERIMENT_NAME="${cfg}" python avatarl.py "${path}"
}

regular_configs=(
  regular_30m
  regular_70m
  regular_150m
  regular_300m
)

avatarl_configs=(
  avatarl_30m
  avatarl_70m
  avatarl_150m
  avatarl_300m
)

for cfg in "${regular_configs[@]}"; do
  run_regular "${cfg}"
  log "regular" "Finished ${cfg}"

done

for cfg in "${avatarl_configs[@]}"; do
  run_avatarl "${cfg}"
  log "avatarl" "Finished ${cfg}"

done

log "all done" "Ablation sweep complete"
