# avatarl - training shakespearean text generator from scratch using reinforcement learning

## the problem statement

"train a language model from scratch using only reinforcement learning. no pretraining, no supervised learning. just rewards and exploration. make it generate coherent shakespeare."

i loved conceptualising the problem but as experiments had to be designed my first reaction: *"bruh in language models we learnt that we need strong base model and strong priors in it to get rl to work. how are we going to make this happen from literally nothing?"*

but tbh once something has been designated as an objective, it is always possible to hack your way to it.

## many people questioned if it is possible

let me count the ways this should fail:

1. **the space is massive**: you start with random weights and want to explore an astronomical number of possibilities to find meaningful patterns. what do you reward, what do you penalize, and how do you guide exploration?

2. **no structure**: language has grammar, semantics, pragmatics. rl only has rewards. how do you learn that 'the' comes before nouns without being told what a noun is?

3. **sparse rewards**: in chess, every move changes win probability. in language, 31 characters might be garbage before one makes a word. and even if we tried brute forcing using gpt2 tokenizer, we would have 50k vocab, how do you permute and compute anything feasibly?

4. **no opponent**: alphazero or any other field with pure rl learning has had self-play. language doesn't compete with itself. it is too vast.

## but wait... why does pure rl work elsewhere?

so i hit some deep research on various kinds of cases that have worked with pure rl and why they succeeded.

my deep research query - "all cases where pure rl worked for learning things from scratch, in any domain. and why did it work? what was the methodology and learning and challenges overcome?"

### alphazero (chess/go)
- started from: random moves
- achieved: superhuman play in 24 hours
- secret sauce: clear win/loss signal + self-play curriculum

### openai five (dota 2) 
- started from: aimless wandering
- achieved: beat world champions
- secret sauce: 180 years of self-play per day + emergence

### curiosity-driven robotics
- started from: random motor commands
- achieved: infant-like exploration behaviors  
- secret sauce: intrinsic motivation + prediction error

ok, so there's a pattern. they all had:
1. clear feedback (win/loss, prediction error)
2. natural curriculum (self-play, increasing difficulty, lego block equivalents)
3. massive scale (years of simulated experience) making up for the lack of innate priors or pre-training

## i can't really do scale, i need to improvise smart exploration strategies

openai five's biggest lesson: "we expected to need sophisticated algorithms. we were wrong. we needed scale."

but that's for dota. for language, scale isn't the only challenge. wait, let me think differently.

i need to design a strategy that balances exploration and exploitation effectively, while keeping exploration diverse enough to avoid local optima and small enough to ensure efficient convergence.

what if language is just a game where:
- the "board" is the context
- the "moves" are character choices  
- the "win" is matching shakespeare's distribution

after all i already have the pretraining dataset, can i not leverage it to my advantage as a strong groundtruth signal?

## and i can't do tokenizer based

i just can't use gpt2 tokenizer, it has 50k vocab and that makes a really large exploration space when you are trying to hit rewards from randomly initialized weights.
but what is 50k tokens is actually just 65 characters. sixty. five.

let me do the math:
- trying all next characters: 65 forward passes
- that's... nothing. my laptop can do that.
- we can literally try everything

this changes the challenge difficulty a lot. 

it's not "explore the infinite space of language." 

it's "optimize over a finite, manageable space."

## how do i leverage the pretraining shakespeare dataset?

cutting myself more slack, i am going to create a self-supervised rl scheme. 

we have the entire tinyshakespeare dataset. we can compute:
- p(char | any context) - the true distribution
- bigram frequencies  
- trigram patterns
- word boundaries

my engineering brain is also optimizing suffix trees in the background to efficiently compute substring frequencies and patterns, enabling faster lookups and reducing computational overhead. 

wow, this looks doable.

let me combine everything:

1. **small vocab (65)** → can try all possibilities
2. **have true data** → can compute perfect rewards  
3. **rl framework** → learns *why* not just *what*

## the plan roughly looks like this
1. start with character frequency matching using kl divergence as a reward.
2. progress to learning bigram dynamics with transition probabilities as a reward.
3. encourage word emergence by rewarding complete words and common word boundaries.
4. develop syntactic patterns by rewarding punctuation rules and capitalization.
5. finally reward for shakespeare-specific patterns and iambic hints.

i may or may not stick with grpo and this exact plan.

might create something else if that looks more fun or p(foom) maxxer. we'll see.

creating a shakespeare but it's evolution, not instruction!

*"we are such stuff as dreams are made on, and our little life is rounded with a sleep."* - soon to be generated by pure rl!