# avatarl

training a tinyshakespeare model from random initialization using pure reinforcement learning - no cross entropy loss based pretraining.

## overview

normal process for building models for any task is typically pretrain + midtrain + sft + rl, where my understanding is that

- pretrain = builds a world model
- midtrain = adds extensive domain knowledge for any area absent in organic pretrain corpus
- sft = rich demonstration data on instruction following, alignment, establishing persona and style
- rl = optimize the demonstrable to what can get fuzzy. improve reasoning, honesty and harmlessness. teaching the model to read between the lines.

this repository contains an optimized implementation of gpt2 but getting trained purely with grpo (group relative policy optimization). 

so you can see we are skipping all stages directly. but for the objective we have, i.e. tinyshakespeare level, we are skipping just one stage - pretraining.
the implementation is hyperoptimized, as all good things should be.

and we might be creating lots of unorthodox things here, as any fun loving man should.

## key files

- **config.py**: all configuration parameters (model size, training hyperparameters, features)
- **train_core.py**: core training logic and grpo implementation
- **train.py**: local training entry point (single or multi-gpu)
- **train_modal.py**: modal cloud deployment for distributed training

## requirements

- python 3.11 (recommended for modal parity)
- pytorch 2.5.0
- flash attention 2.6.3
- cuda-capable gpu (a100/h100 recommended)

## quick start

```bash
# setup
bash setup_server.sh

# local training
python train.py

# multi-gpu training
torchrun --nproc_per_node=8 train.py

# modal cloud training
modal run train_modal.py
```

edit `config.py` to change any parameters before running.


## progress
**28may25**] so since here i am starting with random weights, i wished to find a way to bootstrap this. and for rl, i am starting with grpo algo. 

**29may25** i was able to get positive rewards, the approach i tried was to have a bigram objective for partial scoring of predictions and groundtruth, i am getting positive rewards and after like 80% of the run, it is matching bigram level performance and then drops off. see [wandb runs](https://wandb.ai/ahm-rimer/gpt2-grpo-v2/reports/avatarl-runs--vmlldzoxmzazotu3mw) for detailed metrics.

**30may25** so i figured i could ramp up the ngrams. trigram after bigram and so on, but this approach is going to scale badly. so i decided to think deeper on this. since i need to run many experiments with limited personal budget, i improved speed from 27-30min previously to 2min50s current. now i can do 10x more experiments and it is a good base.

**03june25** shared insights on the progress made in bootstrapping a random weights system with rl pretrain in bootstrapping md file.

**current** bridge the gap between bootstrapped level of performance and groundtruth accuracy.


## contributing

contributions are welcome! please feel free to submit a pull request.

## license

this project is licensed under the apache 2.0 license - see the [license](license) file for details.

## citation

if you find this work useful in your research, please consider citing:
```bibtex
@software{avatarl2025,
  author = {tokenbender},
  title = {avatarl: training language models from scratch with pure reinforcement learning},
  year = {2025},
  publisher = {github},
  journal = {github repository},
  url = {https://github.com/tokenbender/avatarl}
}
```