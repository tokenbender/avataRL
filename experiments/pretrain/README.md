# Pretrain Experiments

This directory organizes GPT pretraining variants so both `train.py` and
`avatarl.py` can be launched with explicit configuration files.

## Layout

- `regular/` – Vanilla GPT pretrain setup (mirrors the original train.py defaults).
- `avatarl/` – AvataRL dual-expert pretrain configuration.
- `baseline_small/` – Lightweight variant for smoke tests (fewer layers,
  smaller batch, lower LR).

Each subdirectory exposes a `config.py` that can be passed directly to
the training scripts, e.g.:

```bash
python train.py experiments/pretrain/regular/config.py
python avatarl.py experiments/pretrain/avatarl/config.py
```

Override individual hyperparameters on the CLI as usual:

```bash
python train.py experiments/pretrain/baseline_small/config.py --max_iters=500
```

Feel free to add more variant folders following the same pattern.
