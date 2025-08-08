# avatarl

training a natural language model from random initialization using pure reinforcement learning.

## overview

normal process for building models for any task is typically pretrain + midtrain + sft + rl, where my understanding is that

- pretrain = builds a world model
- midtrain = adds extensive domain knowledge for any area absent in organic pretrain corpus
- sft = rich demonstration data on instruction following, alignment, establishing persona and style
- rl = optimize the demonstrable to what can get fuzzy. improve reasoning, grounded behaviour and harmlessness. essentially learning generalisation over memorization.

this repository contains an optimized implementation of gpt2 but getting trained purely with reinforcment learning.

and we might be creating lots of unorthodox things here, as any fun loving person should.

## key files

- `avatarl.py` - main training script implementing avatarl reinforcement learning algorithm for language model pretraining
- `model.py` - gpt model architecture with transformer blocks, attention, and language modeling head
- `config/train_avatarl.py` - training configuration for avatarl experiments (hyperparameters, model size, optimizer settings)
- `configurator.py` - command-line configuration override system for experiment management
- `modal_train.py` - modal cloud deployment for distributed training, profiling and ~~benchmaxxing~~ benchmarking.
- `start.sh` - local training launcher to run experiments with environment setup and multi-gpu support
- `docs/avatarl.md` - technical documentation explaining the avatarl framework and positive reinforce approach

## requirements

### modal cloud (recommended)
- modal account ([sign up free](https://modal.com))
- modal cli: `pip install modal`
- authenticate: `modal setup`

### local training (optional)
- python 3.12
- pytorch 2.6.0
- cuda 12.6+ capable gpu (h200/h100/a100 recommended)
- flash attention (auto-installed)

## quick start

### local training

```bash
# setup environment
bash start.sh

# single gpu
python avatarl.py --compile=False

# multi-gpu (8 gpus)
torchrun --nproc_per_node=8 avatarl.py
```

edit `config/train_avatarl.py` to change hyperparameters before running.

### modal cloud training

```bash
# install and authenticate modal
pip install modal
modal setup

# run training
modal run modal_train.py:train_avatarl_single_node
```

## progress
**28may25**] so since here i am starting with random weights, i wished to find a way to bootstrap this. and for rl, i am starting with grpo algo. 

**29may25** i was able to get positive rewards, the approach i tried was to have a bigram objective for partial scoring of predictions and groundtruth, i am getting positive rewards and after like 80% of the run, it is matching bigram level performance and then drops off. see [wandb runs](https://wandb.ai/ahm-rimer/gpt2-grpo-v2/reports/avatarl-runs--vmlldzoxmzazotu3mw) for detailed metrics.

**30may25** so i figured i could ramp up the ngrams. trigram after bigram and so on, but this approach is going to scale badly. so i decided to think deeper on this. since i need to run many experiments with limited personal budget, i improved speed from 27-30min previously to 2min50s current. now i can do 10x more experiments and it is a good base.

**03june25** shared insights on the progress made in bootstrapping a random weights system with rl pretrain in bootstrapping md file.

**04june - 01july25** various efforts like curriculum learning and reward chaining to bridge the gap between bootstrapped level of performance and groundtruth accuracy.

**1july - 11july25** all previous experiments hit a plateau, briefly consider changing the experiment to minimal pretrain + rl. start in a completely new direction. 

**11july - 24july25** mostly boostrapping from own judgement to get partial rewards other than gold token. but initial learning is very noisy as model judgement is not establishment this early in training. a method works where i scale partial reward from zero to 0.3 but i don't go with it because it would be very slow and start converging well only late. also it looks very similar to pretrain + rl. lots of time and compute loss due to hidden bugs.

**25july - 06aug25** cleaned up the codebase, essentially i gave avataRL a new avatar. not sorry for the pun. pick up a new idea - referee model that is trained on groundtruth data and is used to score the predictions of the model. this works nicely, converging with reasonable compute expense and referee model does not need to be bigger than the model in training. i announce success of the project.

**07aug25** update everything in public codebase.

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