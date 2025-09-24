# Pretrain Experiments

This directory organizes GPT pretraining variants so both `train.py` and
`avatarl.py` can be launched with explicit configuration files.

## Layout

- `regular/` – Vanilla GPT pretrain setup (~250M params, mirrors original defaults).
- `regular_30m/`, `regular_70m/`, `regular_150m/`, `regular_300m/` – Size sweep for baseline GPT models.
- `avatarl/` – AvataRL dual-expert pretrain configuration (~250M params).
- `avatarl_30m/`, `avatarl_70m/`, `avatarl_150m/`, `avatarl_300m/` – Size sweep for AvataRL runs with matching reward knobs.
- `baseline_small/` – Lightweight variant for smoke tests (fewer layers,
  smaller batch, lower LR).

Each subdirectory exposes a `config.py` that can be passed directly to
the training scripts, e.g.:

```bash
torchrun --standalone --nproc_per_node=1 train.py experiments/pretrain/regular_30m/config.py
torchrun --standalone --nproc_per_node=1 train.py experiments/pretrain/regular_300m/config.py
torchrun --standalone --nproc_per_node=1 avatarl.py experiments/pretrain/avatarl_70m/config.py
```

Checkpoints now land beside each config under an `out/` folder
(e.g. `experiments/pretrain/regular_30m/out/`).

Override individual hyperparameters on the CLI as usual:

```bash
torchrun --standalone --nproc_per_node=1 train.py experiments/pretrain/baseline_small/config.py --max_iters=500
```

Feel free to add more variant folders following the same pattern.
