# avatarl: training language models from scratch with pure reinforcement learning

> "if you know the way broadly you will see it in all things" — Miyamoto Musashi

## abstract

standard language model pre-training relies on a cross-entropy objective, which rigidly teaches the model that only a single token is the correct continuation for a given context. this is a fundamentally incomplete representation of language, where multiple continuations are often plausible.

avatarl addresses this by replacing the traditional cross-entropy objective with a sophisticated **reinforcement learning (rl) framework**. during the pre-training phase, a player model learns from a **continuous reward signal** derived from a pre-trained critic and ground-truth reality. this allows the player to learn a rich, distributional understanding of language, rewarding it for predicting tokens that are not just the single 'correct' answer but are also plausible alternatives, as judged by the critic. this approach represents a more principled pre-training objective, designed to create models with a deeper and more nuanced understanding of language from the very beginning [1].


---

## understanding avatarl

imagine you're teaching a child (the player model) how to play a piano. 

**traditional pretraining**: the piano student sits with a strict teacher who slaps their hand every time they hit the wrong key. "No! Only C# here! Never D!" The student memorizes thousands of songs note-by-note and is terrified of mistakes. this creates a technically perfect but robotic player who can only reproduce what they've memorized.

**avatarl's rl approach**: the student plays in a jazz club with an experienced critic watching. instead of punishment, they get applause. loud cheers for perfect notes, moderate clapping for harmonious alternatives, polite claps for creative attempts. the critic (a seasoned jazz pianist) whispers suggestions: "try a B♭ there, it creates tension" the student learns not just what's "correct" but what's musically meaningful.

in rl terms:
- **state**: the musical context (previous notes played)
- **action**: choosing the next note
- **reward signal**: continuous applause (0 to 1)
- **policy**: the student's evolving musical intuition
- **critic**: provides expertise without dictating every note

the best part starts when the student starts improvising - they play their own ideas (student top-k), consider the critic's suggestions (critic top-k), while keeping the original melody in mind (ground truth). over time, they develop their own style while respecting musical theory. this is reinforcement learning i.e. learning through rewards rather than rigid correction, discovering patterns through exploration rather than memorization. 

### standard pretraining => one-hot encoded lottery
there is RL everywhere for those with the eyes to see.
this is your traditional pretraining but you can also think of it as REINFORCE with binary rewards.
```ascii
   standard cross-entropy loss
   
   context: "the cat..."
   
   [sat] [jumped] [ran] [slept]
     ↓      ↓       ↓      ↓
    WIN    LOSE    LOSE   LOSE
   
   payout rules:
   - hit the exact word from training data = win $1
   - hit anything else = lose
   
```

### avatarl => progressive reward lottery
we can reimagine the same problem as a progressive reward lottery.

```ascii
   avatarl's continuous rewards
   
   context: "the cat..."
   
   [sat] [jumped] [ran] [slept]
     ↓      ↓       ↓      ↓
    HIGH    MED     LOW    TINY
   
   payout rules (from expert consensus):
   - ground truth (sat) = $82
   - critic's favorites = $10-45  
   - plausible alternatives = $1-9
   - noise = nothing
   
```
now how do we design this around some of the challenges we face in RL?
---

## the problems with RL in pre-training language modeling

### intuition

let us look at how we teach the model traditionally. standard pretraining with cross-entropy loss is actually a special case of REINFORCE where only the gold token gets reward 1.0 and all others get 0 [3]. so we can at least create a binary reward where gold token gets 1 and all others get 0. the challenge of framing pretraining as an rl problem isn't hard however there are a few concerning points.

```ascii
   broadly RL looks like this:

   step 1: sample action (probability of each action - inverse of size of action space)
   step 2: run rollout (reward for each action)
   step 3: update policy (gradient descent)
```

the first problem is that **the action space is too large for a language model** on token level because the entire vocab size is ~50k. and one of the biggest reasons we do pretraining before RL is to make sure the most probable actions are the ones that are most likely to be correct. so if we expect chances of correct token to be present in top 64 choices, we can rollout 64 times and get the reward for each action. however, without pretraining, the probability of next token being correct is random chance - 1/50k i.e. 0.002% so if you do rollouts you'd get 0.002% * 64 = 0.128% chance of getting the correct token. huge number of rollouts per step and such low reward signal makes it computationally extremely expensive to train a model using pure RL.

the second problem is that the **win condition is not absolutely clear**. traditional RL in games is something with deterministic win or loss. each action being rewarded or punished enforces the model to learn the correct action over a limited space. however, in language modeling, the win condition is not clear, given the context the right way to predict the next token could vary a lot. and this creates a challenge that if you wanted to have a clear win condition or if you wanted to have a smooth probabilistic range of partial reward, then that becomes extremely hard with large language models. the model is not trying to win a game, it is trying to minimise the loss over predicting the next token. and the bigger the context is, the more variation can be expected in what could be the correct next token. 

at its core, avatarl reframes language model pre-training as a reinforcement learning problem. the student model is an rl agent that learns to make decisions (predicting the next token) in an environment (the text sequence so far). this approach aligns with the reincarnated rl paradigm [8], where we reuse prior computational work (the pre-trained critic) rather than starting from scratch. 

as a solution to first problem, we can use a smaller action space. but reducing the action space arbitrarily does nothing. we apply something which is called **active token filtering** to eliminate the inconsequential tokens. we use the critic model to share its knowledge of the world (critic top k) and the student does the same (student top k). we combine this with the ground truth to create a subset of actions that are most likely to be correct (max `2*k + 1` action space). the student learns from the highly likely actions. for sampling-related context in language models, see top-k and nucleus (top-p) sampling analyses [5].

for the second problem, we need something to act as a mechanism to provide a smooth reward range which can provide appropriately reward the actions. simply relying on just gold token keeps the reward signal sparse and relying on just critic distribution either becomes distillation if the critic is a bigger teacher model. it also puts a cap on the value from the reward signal, the student model does not derive value after reaching similar level as the critic. we need an option to leverage the reward signal from the critic and the student. and ideally create a method which teaches the model richer distribution than just one-hot encoding and also does not remain capped on the value from the critic reward.

the entire process can be visualized as a simple loop:
```ascii
+-------------------+      +-------------------+
|   state s_t       | ---> | student policy    |
|  (context tokens) |      | π_θ(s_t) → logits |
+-------------------+      +-------------------+
            |                         |
            |         +---------------+---------------+
            |         |                               |
            v         v                               v
     +-----------+  +-----------+               +-------------+
     | top-k_s   |  | top-k_c   |               | gold token  |
     | (student) |  | (critic)  |               | a*_t        |
     +-----------+  +-----------+               +-------------+
            \            |   |                       /
             \           |   |                      /
              \          v   v                     /
               +----------------------------------+
               | filtered actions A_t             |
               | (≤ 2k + 1, deduped)             |
               +----------------------------------+
                            |
                            v
               +----------------------------------+
               | reward model (expert consensus):|
               | reality + critic (blend);       |
               | active-token smoothing          |
               +----------------------------------+
                            |
                            v
               +----------------------------------+
               | r_i over A_t: renorm on A_t      |
               | reward only above-mean; clamp max|
               +----------------------------------+
                            |
                            v
               +----------------------------------+
               | REINFORCE update                 |
               | L = -Σ_i r_i · log π_θ(a_i|s_t) |
               | + entropy bonus                  |
               +----------------------------------+
                            |
                            v
               +----------------------------------+
               | update θ; pick a_t ∈ A_t         |
               | append to context → s_{t+1}      |
               +----------------------------------+
                            |
                            v
                          (loop)
```

notation used above: `s_t` = context at time t (tokens so far); `a*_t` = ground-truth next token; `A_t` = student top-k ∪ critic top-k ∪ `{a*_t}` (deduped, ≤ `2k+1`); `π_θ(a|s_t)` = student policy over tokens; `r_i` = reward for action `a_i ∈ A_t`.

in **avatarl reward mechanism**. instead of a sparse `+1` for the correct token and `0` for all others, we construct an "ideal" reward model at each step that provides a dense, continuous reward signal for any potential action. this reward model is an **expert consensus**.

## how to combine the reality and the critic?

### the expert consensus problem → active-token smoothing
there’s a subtle problem with expert consensus if we’re not careful. if the reality expert
is one-hot, rewards become ultra-sparse and the student gets almost no exploration signal.
if we lean only on the critic, we cap the student at the critic’s quality and inherit its
biases. if we smooth over the full vocabulary, we sprinkle reward on implausible junk and
dilute the gradient.

the rationale is simple: give plausible alternatives a fair chance without rewarding noise.
most of the probability should remain on what actually happened (the ground truth), but a
small ε should be shared only among tokens that are likely in this context. this keeps
exploration focused, plays nicely with the critic’s judgement, and stabilizes the
REINFORCE update.

the process inside expert consensus looks like this:
```ascii
active-token smoothing within expert consensus (reality + critic)

context s_t
   |
student top-k_s     critic top-k_c     gold a*
        \             |      |            /
         \            |      |           /
          +--------------------------------------+
          | A_t = dedup(top-k_s ∪ top-k_c ∪ {a*}) |
          | (≤ 2k + 1 tokens)                      |
          +--------------------------------------+
                      |
                      v
reality distribution over A_t:
- a* gets 1 - ε
- others in A_t share ε / (|A_t| - 1)
- outside A_t gets 0
                      |
                      v
critic distribution (restricted to A_t)
                      |
                      v
expert consensus (blend reality + critic)
- renormalize only on A_t
- above-mean mask + clamp
                      |
                      v
final rewards r_i over A_t → used by REINFORCE
```

#### 1. the ground truth expert

we need to create a reward signal for the ground truth token. we can do this by using a one-hot encoding over the full vocabulary. however, this is not a good signal for the model to learn from. we need to create a reward signal that is more informative. we can do this by using a label smoothing technique.

```ascii

   before: one-hot over full vocab       after: active-token label smoothing (A_t only)
   ┌──────────────────────────────┐      ┌──────────────────────────────────────────────┐
   │ ground truth a* = 100%      │      │ A_t = dedup(top-k_s ∪ top-k_c ∪ {a*})       │
   │ everything else = 0%        │  →   │ |A_t| ≤ 2k+1 (≈ 33 when k=16)               │
   └──────────────────────────────┘      │                                              │
                                         │ p_reality(a*) = 1 - ε                        │
                                         │ p_reality(a ≠ a*, a ∈ A_t) = ε/(|A_t|-1)     │
                                         │ p_reality(a ∉ A_t) = 0                       │
                                         │                                              │
                                         │ example: k=16, ε=0.10, |A_t|=33               │
                                         │   a* = 90.000%                                │
                                         │   each other in A_t ≈ 0.3125%                 │
                                         └──────────────────────────────────────────────┘

   "we already have the ground truth, but we’ll share a small ε only among the ~2*k + 1 tokens
    that actually matter in this context — nothing for the rest."
```

#### 2. using the critic expert as the calibrated judge

to create plausible alternatives, we can use the critic model. the critic model is a pre-trained model that has been trained to predict the next token. we can use the critic model to create a reward signal for the purpose.

```ascii
   
   ┌─────────────────────┐
   │ based on model priors:   │
   │ "sat" = 60%            │
   │ "slept" = 30%          │
   │ "jumped" = 8%          │
   │ "flew" = 2%            │
   └─────────────────────┘
   
   "many things could work here; we score them consistently."
```

### the expert consensus

we can combine their reward signals using a **weighted geometric mean**:

```ascii
   
   reality (70% weight)         critic (30% weight)
          \                          /
           \    geometric mean      /
            \    p^0.7 × p^0.3     /
             \                    /
              [combined wisdom]
                      |
                      v
            ┌─────────────────┐
            │ final rewards:   │
            │ "sat" = 82%     │  <-- ground truth still wins
            │ "slept" = 15%   │  <-- but critic's ideas get credit
            │ "jumped" = 2%   │  <-- even small ideas matter
            │ "flew" = 0.1%   │  <-- tiny but non-zero to promote exploration
            └─────────────────┘
```

```ascii
   the formula visualized
   
   reality expert says:          critic expert says:
   ┌──────────────┐             ┌──────────────┐
   │ "sat" = 0.9  │             │ "sat" = 0.6  │
   │ others = 0.1 │             │ "slept" = 0.3│
   └──────────────┘             │ "jumped"= 0.08│
          ↓                      └──────────────┘
          ↓ power of 0.7                ↓ power of 0.3
          ↓                             ↓
      0.9^0.7 = 0.93            0.6^0.3 = 0.86
          ↓                             ↓
          └─────────────┬───────────────┘
                        │
                    multiply
                        │
                   0.93 × 0.86
                        │
                     = 0.80
                        │
                  (normalize all)
                        │
                        ↓
              ┌─────────────────┐
              │ "sat" = 82%     │  <-- ideal reward
              │ "slept" = 15%   │  <-- decent reward
              │ "jumped" = 2%   │  <-- try again
              └─────────────────┘
                        ↓
                 scale by 100
                        ↓
              rewards then sparsified (above-mean),
              and proportionally capped per position (≤ 1.5)
```

**intuition**: for a token to receive higher rewards, both experts must agree. the smoothed reality expert ensures the ground truth is strongly preferred without completely vetoing alternatives, while the critic expert provides a smooth gradient of preferences for all plausible tokens. the resulting probabilities serve directly as positive rewards.

### the mathematical model of the reward signal

a key insight of avatarl is that we do not create a reward for a *single* action the student takes. instead, we first construct a **dense reward landscape** for *active filtered tokens*, and then we evaluate the student's policy against this landscape.

**1. defining the experts mathematically**

let `a` be a potential action (a token) and `a*` be the ground-truth token from the dataset.

-   **the groundtruth expert, `p_reality(a | s)`**: this expert strongly prefers the ground-truth token `a*` but uses **active token label smoothing** to maintain a continuous distribution.
    -   `p_reality(a | s) = 1 - ε` if `a = a*` (where `ε = 0.1`)
    -   `p_reality(a | s) = ε / (num_active - 1)` if `a ∈ active_tokens` and `a ≠ a*`
    -   `p_reality(a | s) = 0` if `a ∉ active_tokens` (i.e. not in the active filtered tokens)
    
    where `active_tokens` = student's top-k ∪ critic's top-k ∪ {ground truth} (typically ≤32 tokens after deduplication)
    
    this gives 90% probability to the ground truth and distributes 10% **only across active tokens** rather than the entire vocabulary. this concentrates the exploration signal on relevant alternatives instead of wasting it on 50,000+ irrelevant tokens.

-   **the critic expert, `p_critic(a | s)`**: the critic model's probability distribution over the vocabulary.
    -   `p_critic(a | s) = softmax(critic_logits)`

**2. the expert consensus formulation**

the ideal distribution, `p_ideal`, is the **weighted geometric mean** of the two expert distributions:

`p_ref(a | s) ∝ [p_reality(a | s)]^w_r * [p_critic(a | s)]^w_m`

here `w_r` (reality_weight) and `w_m` (mentor_weight) are the expert weights; defaults are 0.7 and 0.3 respectively. after normalization on the action set A, this gives us a proper probability distribution. the resulting `p_ref` serves as our **positive reward signal**—a vector over A containing the calibrated "ideal" probabilities (and thus rewards).

**3. evaluating the student's policy**

the student's policy, `π_student(a | s)`, is judged against this reward landscape. the policy gradient loss aims to move the student's distribution closer to the ideal one. the reinforce rule is `loss = -e_{a ~ π_student} [ r(a) ]`, where our reward `r(a) = p_ideal(a | s)`—we also add an entropy bonus `−β·H[π_θ(·|s)]` to encourage exploration, with `β = entropy_coefficient` (default 0.01) [2, 4]. in practice, stable policy-gradient updates often use PPO-style clipping and variance reduction such as GAE [6, 7].

#### learning to calibrate the critic internally

by including the student's top-k predictions in the action space, we create a self-reinforcement mechanism. not only the model learns to predict the ground truth, but also learns to calibrate the critic's predictions via its own top-k predictions.

**concurrent top-k evaluation**

instead of sampling actions and performing k sequential rollouts, avatarl concurrently evaluates the student's top-k, the critic's top-k, and the gold token — an efficient 2k+1 rollout-equivalent with guided exploration. specifically:
- we evaluate all actions in the expanded action space (~2k+1 tokens) simultaneously
- each action gets its reward from the pre-computed expert consensus reward model
- the student's entire probability distribution over these actions is updated via policy gradient

```ascii
   typical rl (k sequential rollouts):        avatarl (2k+1 rollout-equivalent, concurrent):
   
   sample action → run rollout (×k)       gather student top-k + critic top-k + gold
   sample action → run rollout            ↓
   sample action → run rollout            compute rewards for all selected actions
   ...repeat n times...                   ↓
   ↓                                      update policy on all actions
   average rewards                        simultaneously
   ↓                                      
   update policy                          one concurrent pass, guided exploration
   
   [slow & high variance]                [fast & low variance]
```

for efficient pretraining we avoid monte carlo sampling, and instead do **exhaustive evaluation** over the reduced action space.

**the mathematical truth**:
```python
# what avatarl doesn't do (sequential per-token rollouts):
for rollout in range(num_rollouts):  # no sequential rollouts per token
    action = sample_from_student()   # no stochastic rollout sampling at train time
    reward = run_episode(action)     # no multi-step episodes per token

# what avatarl actually does (k-rollout-equivalent, concurrent):
action_space = student_top_k ∪ critic_top_k ∪ ground_truth  # ~32 actions
rewards = consensus_model(action_space)  # pre-computed for all actions
loss = -sum(log_prob(action) * reward for action in action_space)  # update all at once
```

**per-prediction statistics**:
- **rollout-equivalent**: 2k+1 (student k + critic k + gold), evaluated concurrently
- **actions evaluated**: ~2k+1 (entire expanded action space)
- **samples drawn**: deterministic top-k selection (or top-p if enabled)
- **reward computations**: ~2k+1 (one per action in space)
- **gradient updates**: 1 (single policy gradient step)

## results

avatarl has been completely open sourced and researched in the open from day 1.
hence what you see are the earliest results after getting it to work.

we successfully validated the approach on a model in size range of 30M-250M parameters using avatarl. in each case, we used a critic model of size 30M parameters. around 2B tokens were used for training where the dataset used for this was [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext). 

for the same param size, the avatarl trained model can reach similar cross entropy loss as a standard pretrained model.

avatarl also allows us to train the models for equivalent of 50-60B tokens on 2B tokens of data because of the dense reward signal.

this is how the training and validation curves look like for a 250M parameter model, validated every 500 steps.

### Training loss against combined distribution (2k+1)
<img src="../assets/train_loss.png" alt="train_curves" width="50%">

### Average Reward Progress
<img src="../assets/avg_reward.png" alt="avg_reward" width="50%">

### Cross-Entropy Loss
<img src="../assets/ce_loss.png" alt="ce_loss" width="50%">

## Samples

### Generated Text Examples

**Prompt:** *"The true meaning of life is often"*

---

#### Sample 1
> The true meaning of life is often more complex than the actual cause, but we can think so much about each of them if we take into account the real and possible work that underlies an individual's character and personality.
>
> When we do that, the very nature of our relations with other people takes us into their distinct forms and carries them far into their various circumstances. When a particular person is isolated from the "ideal" circle of people with whom we have worked – and others who are displaced from the

#### Sample 2
> The true meaning of life is often an all-consuming exchange of goods — real commodities such as gold, services such as medicine, clothes and appliances — while also building up a certain personal meaning for all: this change helps us begin to transcend some material barrier to the search for meaning. Rather than content, though, we are simply noticing how the changes our material conditions are also creating.
>
> Today, it's easy to underestimate how life can transition, if we didn't think so. Increasingly, health and education are more about

#### Sample 3
> The true meaning of life is often put in terms of one's relationship with us. "There is life of this for us as well," Mr. Adams says. "For me, the meaning of life was precisely this."<|endoftext|>I was delighted to be offered the opportunity to see David Koretzner, one of my favorite television reporters who came into the world on such a busy note that I couldn't turn down this season's worth of tweets. Only four of them came

#### Sample 4
> The true meaning of life is often malleable by the way they can change and change more naturally. If no matter how many lives they provide, life continues to be a process of change, and thus human life is often incomplete.
>
> Is Life Any Better? If so, with all the information about human nature you would like to know about but have no knowledge about human nature, you might ask yourself why. It is not, in order to understand it, to practice and follow as best we can. The journey of your

#### Sample 5
> The true meaning of life is often unknown: Death through illness has in fact brought death from one's own body into another.
>
> Many now seem to see this in the light of a number of ways to die. But rather than focus on dying, we look to be on the look out for what death will bring to us. The notion of death is less cruel than life; death is more honest in each of its parts.
>
> If the death of evil goes on all around us, our act of anger may well follow

#### Sample 6
> The true meaning of life is often determined by circumstances and relationships. As such, personal freedom is a process unto itself and not something else.
>
> This might seem like a completely false premise to be a "mystical situation." However, the fact that every species has at least some sort of a right to life outside of its senses and attitudes shows it, quite literally, to be a person.
>
> In other words, what the individual doesn't have is a right to life of their own free

#### Sample 7
> The true meaning of life is often one of 'making up your mind'.
>
> It takes a lot of hard work and dedication every day for somebody to realize they are doing something amazing. After three years of planning and constant training it was almost clear that they had formed one of the best teams I've ever been given.
>
> My team worked hard. I had a pretty amazing package coming out of the kit room after sending home a huge package. Doing the team thing took time because there were times when there was absolutely nothing I

#### Sample 8
> The true meaning of life is often about a certain general understanding of our existence and how we cope with our suffering. This is how people of a certain age, or a particular religious persuasion, perceive our existence.
>
> We are now reminded that our understanding is limited in comparison to the scope of relationships within us. The way that we may assess ourselves under the ideal of our being the sum of our self-interest, we may do it to others and to oneself in order to make our bodies and identities relevant, but it will be

## future work
this is a very early stage of the research.
i understand that to many trained eyes this feels incomplete.
i am also primarily a hacker and you may sniff it across my work.

the agi lab is understaffed (1 person - me) and we are planning to do the following:
- avatarl training objective as gradually sequence level training objective not just tokens
- train for tool use and minimal world knowledge with pure RL - how small of a banger tool calling model can we get?

if you wish to pick these ideas for yourself or if you wish to collaborate, let me know.
i would appreciate any support in the form of sponsorship, compute or research collaboration.

i would be sharing discussion on the approaches that i took but didn't succeed as well.

## acknowledgements
[Chinmay Kak](https://x.com/ChinmayKak), who discussed this with me on call and instantly started hacking the idea using GRPO. The skeleton code for the earliest version was written by him.

[Telt](https://x.com/twofifteenam) for the support and compute.

[Ravi Theja](https://x.com/ravithejads) for taking initiative on his own to put me in touch with @except_raised.

huge thanks to everyone who reposted, shared and got it the initial attention, i appreciate all of you.

modal cloud is awesome. it makes running experiments simple. no worrying about instance management or forgotten resources - it just works. it kinda suits my adhd hacker brain very well.

the infrastructure is solid too. no cold starts to slow things down.
the pricing is also very reasonable.


## sponsorship
this work was sponsored for compute credits by [@except_raised](https://x.com/except_raised), [@twofifteenam](https://x.com/twofifteenam) and [@modal_labs](https://x.com/modal_labs)


## code

everything is open sourced and available on [github](https://github.com/tokenbender/avataRL).

## references

[1] Dong, Q., Dong, L., Tang, Y., Ye, T., Sun, Y., Sui, Z., & Wei, F. (2025). Reinforcement Pre-Training. arXiv:2506.08007.

[2] Hong, J., Dragan, A., & Levine, S. (2024). Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning. arXiv:2411.05193.

[3] Norouzi, M., Bengio, S., Chen, Z., Jaitly, N., Schuster, M., Wu, Y., & Schuurmans, D. (2016). Reward Augmented Maximum Likelihood for Neural Structured Prediction. NeurIPS 2016.

[4] Shen, S., Cheng, Y., He, Z., He, W., Wu, H., Sun, M., & Liu, Y. (2016). Minimum Risk Training for Neural Machine Translation. Proceedings of ACL 2016.

[5] Ranzato, M. A., Chopra, S., Auli, M., & Zaremba, W. (2016). Sequence Level Training with Recurrent Neural Networks (MIXER). ICLR 2016.

[6] Bahdanau, D., Brakel, P., Xu, K., Goyal, A., Lowe, R., Pineau, J., Courville, A., & Bengio, Y. (2017). An Actor–Critic Algorithm for Sequence Prediction. ICLR 2017.

[7] Peters, J., & Schaal, S. (2007). Reinforcement Learning by Reward-Weighted Regression for Operational Space Control. ICML 2007.

[8] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2022). Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress. NeurIPS 2022. arXiv:2206.01626.

[9] Furlanello, T., Lipton, Z., Tschannen, M., Itti, L., & Anandkumar, A. (2018). Born Again Neural Networks. Proceedings of ICML 2018, PMLR 80:1607-1616. arXiv:1805.04770.