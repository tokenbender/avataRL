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

## key features

- **improved distributed training with grpo**: optimized from 27-30min to 2min50s per run per epoch
- **distributed training**: supports both single-gpu and multi-gpu setups via torchrun
- **bootstrapping with ngram** rewards: starts with bigram rewards and ramps up to trigrams, etc.
- **curriculum learning**: stagewise rewards for testing language understanding, adaptive kl penalty for stable convergence

## implementation details

### grpo (group relative policy optimization)
our grpo implementation extends beyond typical reference implementations (like [tiny-grpo](https://github.com/open-thought/tiny-grpo))

#### architectural optimizations
- **model configuration**: 6 layers, 6 attention heads, 384 embedding dimension (~10m parameters)
- **fused qkv projection**: single linear layer for q,k,v instead of separate projections
- **pre-norm architecture**: applies normalization before attention/ffn blocks for better stability
- **weight tying**: shares weights between input embeddings and output projection

#### custom training techniques
- **temperature-based sampling**: controls generation diversity during rollouts
- **k-sample generation**: generates multiple samples per context for better advantage estimation
- **adaptive kl coefficient**: dynamically adjusts kl penalty based on divergence history
- **ppo-style clipping**: prevents destructive policy updates
- **entropy regularization**: encourages exploration during training
- **minimum variance threshold**: prevents numerical instability in advantage normalization

#### features beyond standard grpo
- **bigram reference model**: lightweight baseline for reward computation
- **sophisticated reward system**:
  - exact match rewards for correct predictions
  - partial credit based on reference probability
  - combined reward with configurable scaling
- **old policy updates**: updates reference policy every 5 iterations for stability

#### key differences from reference implementations
1. **character-level modeling** on tinyshakespeare (vs token-level on larger models)
2. **no negative rewards** for not discouraging any exploration
3. **dense architecture** optimized for single-gpu training

## requirements

- python 3.11 (recommended for modal parity)
- pytorch 2.5.0
- flash attention 2.6.3
- cuda-capable gpu (a100/h100 recommended)

## quick start

### server setup (recommended)
```bash
# 1. run automated setup script
bash setup_server.sh

# 2. use generated launch scripts
./launch_single_gpu.sh    # for single gpu
./launch_multi_gpu.sh     # auto-detects and uses all gpus
```

the setup script will:
- verify cuda/gpu availability
- create python 3.11 virtual environment
- install pytorch 2.5.0 with appropriate cuda support
- install flash attention 2.6.3 (precompiled or from source)
- download tinyshakespeare dataset
- create convenient launch scripts

### manual installation
```bash
# create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# install dependencies (cuda 12.x)
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy tqdm wandb requests matplotlib nvidia-ml-py3
pip install flash-attn==2.6.3 --no-build-isolation

# download dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### modal deployment
```bash
pip install modal
modal run train_modal.py
```


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