continuing from where the thinking.md file left off, focusing on implementing the outlined strategy steps.

the idea is that rl typically elicits latent abilities that the weights has from the training data (pretrain/midtrain).

but if you have random weights as starting point then the prediction is also supposedly random and we start rewarding or penalising the predictions with nothing to reinforce.

the way i look at this thing, this is a design experiment actually.

## why is this thing harder than learning from pretraining?

because pretraining provides a structured initialization, while starting from random weights lacks any prior knowledge, making convergence slower and more challenging.

also rl is inherently exploratory, highly susceptible to noise, affects small subnetworks and dependent on really sparse rewards - https://arxiv.org/abs/2505.11711.

it may seem like you can just use next token prediction as reference, but sparse redistribution of nothing is nothing - https://wandb.ai/ahm-rimer/avataRL/reports/organic-reinforcement-of-nothing--VmlldzoxMzA3OTAwNw

behold the most randomness you have seen in an RL experiment probably.

if you are coming from a background without much information on rl, it would be hard to follow so i share nathan lambert's rl primer here - https://rlhfbook.com/

rl algorithms are designed to let the system grow like weed, while our rewards guide and shape the growth towards the patterns we desire, ensuring alignment with the intended objectives.

so our success depends on these factors: smallest unit of memorization that random weights can do with rl, reward design, and noise management.

## language modelling is a game
we can define languuage modelling as a game,
- where the context is the current state
- the predicted char is the action
- the reward is a value assigned based on the accuracy of the predicted token
- the two players are the model competing with itself - in parallel and with its past checkpoints.

to simplify my experiments, i focus on smaller models and shorter sequences to reduce computational complexity and isolate key behaviors.

as i develop confidence over things i want to scale, i increase the model size and try bolder targets.

## initialization
the smallest unit of intelligence in a text corpus is the character frequency or better to say ngram frequency.

so my objective is going to be using a mix of hard signal and soft signal to guide the model's learning process.

my choice of hard signal is the tinyshakespeare dataset which is like a groundtruth signal in an rlvr experiment.

my choice of soft signal is bigram statistics derived from the training corpus to provide probabilistic guidance.

my kl divergence reference distribution is the bigram frequency distribution.

model size - 20m param gpt2 arch model.

## experiment design
when the model starts with random weights, it is likely to predict tokens with uniform probability, leading to high entropy outputs.

and gpt2 tokenizer size is 50000, which means i have a chance of being right once by accident if i sample over 50000 times.

do i really expect utf-8 sequences and all kinds of wild patterns over simple old english texts? no.

so instead i pick character level tokenization, basically a unigram lm approach. 

there are only 65 possible character choices for my predictions, corresponding to the english alphabet (both cases), digits, and a few special symbols.

and turns out unigram LMs are not so bad if our goals are not to capture long-range dependencies but rather to model short-term patterns effectively - https://ndingwall.github.io/blog/tokenization

also, claude deep research suggests char-level encoding is probably better choice if our challenge is not scaling but a small corpus - https://claude.ai/public/artifacts/12312dbd-f33f-4ac8-9b85-3600c003d4d2

now i only have to explore and exploit over the 65-token space to guide the model's learning process effectively.

but starting from random weights, the model would need strong density-based guidance to converge effectively.

so here is my design - exhaustive exploration. 

for each prediction, we calculate the rewards, advantages for all 65 characters and update the policy.

the context is a sliding window of character-level predictions, focusing on short-term dependencies.

the choice of rl algorithm is grpo here for starters.

## the experiment
to start as baseline, in each iteration we grab a batch of 8-char context window and ask: "what comes next?"

reference is a key aspect of the design, ensuring that the model effectively utilizes past predictions to inform future actions. 

even if i reduced the sample space to 65 characters i can't really expect to visit over all of them per iteration so i deploy some tricks as i share here.

## eval
for eval, we calculate perplexity and accuracy to assess model performance.

## results curated over effective experiment batches
trying to extract out the most valuable experiments and things i tried out while i was throwing a bunch of ideas just to calibrate my own understanding. my experiment management skills are optimized for just my learning but understood a few key points about how i can do better for next major update.

## temp varied sampling with bigram guidance
result - the model learns to imitate bigram predictions around 0.8 epoch and then rewards start going down again because overfitting to bigram is continuously getting penalised wrt the ground truth
wandb - https://wandb.ai/ahm-rimer/gpt2-grpo-v2/reports/avataRL-runs--VmlldzoxMzAzOTU3Mw

conclusions - you do not need just bigram, performance can be improved by using 3gram and 4gram as well probably, but is that enough? how would you scale past that? so need to figure out a way to let the model bootstrap from minimal signal and bridge the gap between bootstrap and groundtruth on its own.
another thing i observed was that the horizon window was too big and i felt the bigram level learning can happen faster.

since like bigram, 3gram and 4gram can also overfit upto a point and fall-off, i want to try out other ideas before i elevate bootstrap signal quality.


## exhaustive exploration - try all 65 characters, 65 rollouts per prediction step to calculate rewards
result - every iteration all 65 characters are evaluated and the composition of one correct and rest being graded on partial correctness remains the same. 

this looked like a dense signal by itself first but if you look at the reward composition over all the characters you realise the rewards have low standard deviation. basically we always expect one winner and 64 losers and that kinda keeps the reward roughly the same across the entire run with minor fluctuations. 

conclusion - while i can see that the model probability distribution for the characters increases and stabilises it does not communicate the growth of the model properly.

wandb - https://wandb.ai/ahm-rimer/avataRL/reports/rewards-fixed-with-low-std--VmlldzoxMzA4MTc5OQ

so i come up with a method to actually look into model's growth even with my exhaustive eexploration design.

then this is where my design is different - instead of letting the model generate hundreds of rollouts and hope to cover entire sample space i try all 65 characters.

to ensure i am understanding and weighing what my model's top set of choices are - i calculate log probs for each char and use it as a confidence estimate.

top choice is most probable character and also the most confident choice.

## reward calculation
reward is a combination of how close is the character to bigram distribution, exact match to groundtruth, confidence bonus (if confidence and correct).

## customisations
other than typical reward maxxing, i apply 
- a scaling factor to the confidence bonus to ensure its predictions and accuracy of guessing how correct it is get a boost as the training goes forward.
- another change is ppo-style clipping, which ensures that updates to the policy remain within a trust region, preventing overly large updates that could destabilize training.
- and entropy bonus, to avoid the model getting overconfident and be more exploratory rather than answer from just fewer options.

to be honest, there is a min-maxxing approach needed here for entropy not just monotonically letting it decrease or increase but for now our model tries to avoid entropy collapse by assigning bonuses for exploration.

## a health monitor to my strategy
i add 3 warning metrics
- calibration error: is confidence matching accuracy?
- confidence collapse: is the model becoming overconfident and ignoring exploration?
- gradient stability: are the updates to the model parameters smooth and not wild.

## confidence scaling and its impact on exploration
result - this has the best results so far, my rewards seem to continue improving with each iteration over multiple epochs and do not start dropping after just reaching bigram level, effective it trying to bridge the gap. though everything stabilises and it struggles to improve accuracy.

conclusion - it is possible to teach a model to guess better what would be correct and improve its performance at the same time. entropy collapse or accuracy stagnation are still problems.

wandb - https://wandb.ai/ahm-rimer/avataRL/reports/3-epoch-confidence-scaled-calibrated-model-growth--VmlldzoxMzA4MTk2MA

## next set of ideas to refine:
- avoid entropy collapse by continuously removing low entropy predictions and study its impact on confidence scaling.
- precompute upto 4gram and create a lookup table for efficient reward assignment
- experiments aimed at boostrapping with ngram but increasing performance over ngram level and getting closer to groundtruth accuracy.