#!/usr/bin/env bash
set -euo pipefail

# Common Weights & Biases metadata
export WANDB_PROJECT="${WANDB_PROJECT:-nanogpt-avatarl}"
export EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-pretrain_size_sweep}"

# Torchrun launch arguments (override via REGULAR_TORCHRUN_ARGS / AVATARL_TORCHRUN_ARGS)
REGULAR_TORCHRUN_ARGS="${REGULAR_TORCHRUN_ARGS:---standalone --nproc_per_node=1}"
AVATARL_TORCHRUN_ARGS="${AVATARL_TORCHRUN_ARGS:---standalone --nproc_per_node=1}"

log() {
  printf '\n[%-24s] %s\n' "${1}" "${2}"
}

set_run_metadata() {
  local cfg="$1"
  local sha diff
  sha=$(git rev-parse HEAD)
  diff=$(git diff --stat HEAD || true)
  if [[ -z "${diff}" ]]; then
    diff="clean working tree"
  fi
  export GIT_SHA="${sha}"
  export GIT_DIFF="${diff}"
  export WANDB_TAGS="sweep,${cfg}"
  printf -v WANDB_NOTES 'commit:%s\n%s' "${sha}" "${diff}"
  export WANDB_NOTES
}

run_regular() {
  local cfg="$1"
  local path="experiments/pretrain/${cfg}/config.py"
  set_run_metadata "${cfg}"
  read -r -a args <<< "${REGULAR_TORCHRUN_ARGS}"
  log "regular" "Starting ${cfg}"
  EXPERIMENT_NAME="${cfg}" torchrun "${args[@]}" train.py "${path}"
}

run_avatarl() {
  local cfg="$1"
  local path="experiments/pretrain/${cfg}/config.py"
  set_run_metadata "${cfg}"
  read -r -a args <<< "${AVATARL_TORCHRUN_ARGS}"
  log "avatarl" "Starting ${cfg}"
  EXPERIMENT_NAME="${cfg}" torchrun "${args[@]}" avatarl.py "${path}"
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
