# avataRL

training a tinyshakespeare model -> zero pretrain, all rl baby.

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

- custom implementation with performance-focused design choices
- implementation of group relative policy optimization for improved training stability with ppo style clipping and entropy minimisation
- flash attention
- rmsnorm instead of layernorm
- tight memory management

## implementation details

### grpo (group relative policy optimization)
our grpo implementation extends beyond typical reference implementations (like [tiny-grpo](https://github.com/open-thought/tiny-grpo))

#### architectural optimizations
- **rmsnorm instead of layernorm**: custom rms normalization for ~2x faster normalization
- **fused qkv projection**: single linear layer for q,k,v instead of separate projections
- **flash attention 2 integration**: o(n) memory complexity for attention computation
- **pre-norm architecture**: applies normalization before attention/ffn blocks for better stability
- **weight tying**: shares weights between input embeddings and output projection
- **gelu with tanh approximation**: faster activation function without accuracy loss

#### custom training techniques
- **temperature-based sampling** (t=1.2): controls generation diversity during rollouts
- **k-sample generation** (k=4): generates multiple samples per context for better advantage estimation
- **adaptive kl coefficient**: dynamically adjusts kl penalty based on divergence history
- **ppo-style clipping** (ratio=0.5): prevents destructive policy updates
- **entropy regularization** (coef=0.01): encourages exploration during training
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

- python 3.8+
- pytorch 2.0+
- flash attention 2

## installation
monolith hackable script for modal deployment, just install modal via pip


## progress
**28May25**] so since here i am starting with random weights, i wished to find a way to bootstrap this. and for rl, i am starting with grpo algo. 

**29May25** i was able to get positive rewards, the approach i tried was to have a bigram objective for partial scoring of predictions and groundtruth, i am getting positive rewards and after like 80% of the run, it is matching bigram level performance and then drops off. (TODO - add wandb link and link the older script used for it.)

**30May25** So i figured i could ramp up the ngrams. trigram after bigram and so on, but this approach is going to scale badly. so i decided to think deeper on this. since i need to run many experiments with limited personal budget, i improved speed from 27-30min previously to 2min50s current. now i can do 10x more experiments and it is a good base.

**current** thinking of how to boostrap with sparse rewards to match pretrain level performance. would create a thinking md file.

all the progress and ideas are documented in the THINKING.md file for reference.


## contributing

contributions are welcome! please feel free to submit a pull request.

## license

this project is licensed under the apache 2.0 license - see the [license](license) file for details.