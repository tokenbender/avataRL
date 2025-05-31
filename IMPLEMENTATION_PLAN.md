# Shakespeare Emergence Protocol: Implementation Plan

## Immediate Action Items (Next 24 Hours)

### 1. Modify grpo_dense_v2_flash.py with Corruption Curriculum

```python
# Add to imports
import random

# Add after build_vocab function
def corrupt_text(text, corruption_level=0.1):
    """Corrupt text by swapping, inserting, deleting characters"""
    if corruption_level == 0:
        return text
        
    text_list = list(text)
    num_corruptions = int(len(text_list) * corruption_level)
    
    for _ in range(num_corruptions):
        corruption_type = random.choice(['swap', 'insert', 'delete'])
        idx = random.randint(0, len(text_list) - 1)
        
        if corruption_type == 'swap' and idx < len(text_list) - 1:
            text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
        elif corruption_type == 'insert':
            random_char = random.choice(text_list)
            text_list.insert(idx, random_char)
        elif corruption_type == 'delete' and len(text_list) > 1:
            text_list.pop(idx)
            
    return ''.join(text_list)

# Modify ContextualTextLoader to include corruption
class CorruptionTextLoader(ContextualTextLoader):
    def __init__(self, text, enc, B, T, context_len, device='cuda', corruption_level=0.1):
        super().__init__(text, enc, B, T, context_len, device)
        self.corruption_level = corruption_level
        self.original_data = self.data.clone()
        
    def next(self):
        contexts, targets = super().next()
        
        # Corrupt contexts based on current level
        if self.corruption_level > 0:
            B, T = contexts.shape
            for b in range(B):
                if random.random() < self.corruption_level:
                    # Corrupt some positions
                    num_corrupt = int(T * self.corruption_level)
                    positions = random.sample(range(T), num_corrupt)
                    for pos in positions:
                        contexts[b, pos] = random.randint(0, V-1)
                        
        return contexts, targets
        
    def increase_corruption(self, increment=0.05):
        self.corruption_level = min(1.0, self.corruption_level + increment)
```

### 2. Enhanced Multi-Level Reward System

```python
def compute_hierarchical_rewards(generated, reference, context, enc, dec, shakespeare_stats):
    """Multi-level reward computation with Shakespeare focus"""
    B, T = generated.shape
    rewards = torch.zeros_like(generated, dtype=torch.float32)
    
    # Level 1: Character bigram/trigram (immediate feedback)
    for t in range(T):
        if t > 0:
            bigram = (generated[:, t-1], generated[:, t])
            rewards[:, t] += shakespeare_stats['bigram_probs'][bigram] * 0.1
        if t > 1:
            trigram = (generated[:, t-2], generated[:, t-1], generated[:, t])
            rewards[:, t] += shakespeare_stats['trigram_probs'][trigram] * 0.2
            
    # Level 2: Word completion bonus
    for b in range(B):
        text = dec(generated[b])
        words = text.split()
        for word in words:
            if word in shakespeare_stats['vocabulary']:
                word_positions = find_word_positions(text, word)
                for pos in word_positions:
                    rewards[b, pos:pos+len(word)] += 1.0 / len(word)
                    
    # Level 3: Shakespeare phrase matching
    for b in range(B):
        full_text = dec(torch.cat([context[b, -10:], generated[b]]))
        for phrase, score in shakespeare_stats['common_phrases'].items():
            if phrase in full_text:
                rewards[b, :] += score
                
    # Level 4: Reconstruction bonus (if using corruption)
    if reference is not None:
        exact_match = (generated == reference).float()
        rewards += exact_match * 2.0
        
    # Level 5: Confidence-based intrinsic reward
    # (Implemented through entropy in main training loop)
    
    return rewards

# Precompute Shakespeare statistics
def build_shakespeare_stats(text, enc, dec):
    """Extract statistics from Shakespeare text"""
    stats = {
        'bigram_probs': defaultdict(float),
        'trigram_probs': defaultdict(float),
        'vocabulary': set(),
        'common_phrases': {}
    }
    
    # Compute bigram/trigram frequencies
    encoded = enc(text)
    for i in range(len(encoded) - 2):
        bigram = (encoded[i].item(), encoded[i+1].item())
        trigram = (encoded[i].item(), encoded[i+1].item(), encoded[i+2].item())
        stats['bigram_probs'][bigram] += 1
        stats['trigram_probs'][trigram] += 1
        
    # Normalize
    total_bigrams = sum(stats['bigram_probs'].values())
    total_trigrams = sum(stats['trigram_probs'].values())
    for k in stats['bigram_probs']:
        stats['bigram_probs'][k] /= total_bigrams
    for k in stats['trigram_probs']:
        stats['trigram_probs'][k] /= total_trigrams
        
    # Extract vocabulary
    words = text.split()
    stats['vocabulary'] = set(w.lower() for w in words if len(w) > 2)
    
    # Common Shakespeare phrases
    stats['common_phrases'] = {
        'to be': 3.0,
        'or not': 3.0,
        'thou art': 2.5,
        'shall be': 2.0,
        'my lord': 2.0,
        'i am': 1.5,
        'thee': 1.0,
    }
    
    return stats
```

### 3. Curriculum Controller

```python
class CurriculumController:
    def __init__(self, start_corruption=0.05, target_accuracy=0.7):
        self.corruption_level = start_corruption
        self.target_accuracy = target_accuracy
        self.history = deque(maxlen=100)
        self.phase = 1
        self.phases = [
            {'corruption': 0.1, 'focus': 'bigrams', 'threshold': 0.8},
            {'corruption': 0.3, 'focus': 'words', 'threshold': 0.6},
            {'corruption': 0.5, 'focus': 'phrases', 'threshold': 0.4},
            {'corruption': 0.7, 'focus': 'syntax', 'threshold': 0.3},
            {'corruption': 0.9, 'focus': 'semantic', 'threshold': 0.2},
            {'corruption': 1.0, 'focus': 'shakespeare', 'threshold': 0.1},
        ]
        
    def update(self, accuracy):
        self.history.append(accuracy)
        
        if len(self.history) < 20:
            return self.corruption_level
            
        avg_accuracy = np.mean(self.history)
        
        # Progress to next phase if doing well
        if avg_accuracy > self.phases[self.phase-1]['threshold'] and self.phase < len(self.phases):
            self.phase += 1
            self.corruption_level = self.phases[self.phase-1]['corruption']
            print(f"Advancing to Phase {self.phase}: {self.phases[self.phase-1]['focus']}")
            
        return self.corruption_level
```

### 4. Multi-Agent Implementation

```python
class ShakespeareSwarm:
    def __init__(self, n_agents=4, base_model=None):
        self.agents = []
        self.shared_buffer = deque(maxlen=10000)
        self.generation_count = 0
        
        # Create diverse agents
        temperatures = [0.8, 1.0, 1.2, 1.4]
        exploration_weights = [0.05, 0.1, 0.15, 0.2]
        
        for i in range(n_agents):
            agent = {
                'id': i,
                'model': copy.deepcopy(base_model),
                'temperature': temperatures[i],
                'exploration': exploration_weights[i],
                'successes': 0
            }
            self.agents.append(agent)
            
    def generate_diverse(self, contexts, horizon, k_samples):
        all_generations = []
        all_rewards = []
        
        for agent in self.agents:
            with torch.no_grad():
                gen = generate_with_temperature(
                    agent['model'], 
                    contexts, 
                    horizon, 
                    k_samples, 
                    temperature=agent['temperature']
                )
                all_generations.append(gen)
                
        return all_generations
        
    def update_best_agent(self, rewards):
        # Find best performing agent
        best_idx = np.argmax([r.mean() for r in rewards])
        self.agents[best_idx]['successes'] += 1
        
        # Share parameters if one agent is dominating
        if self.agents[best_idx]['successes'] > 100:
            print(f"Agent {best_idx} is dominating, sharing knowledge...")
            best_params = self.agents[best_idx]['model'].state_dict()
            for i, agent in enumerate(self.agents):
                if i != best_idx:
                    # Soft update to maintain diversity
                    current_params = agent['model'].state_dict()
                    for key in current_params:
                        current_params[key] = 0.9 * current_params[key] + 0.1 * best_params[key]
                    agent['model'].load_state_dict(current_params)
```

## Integration Steps

### Step 1: Minimal Changes to Main Training Loop (1 hour)
```python
# In train_remote(), after initializing models:
shakespeare_stats = build_shakespeare_stats(text, ENC, DEC)
curriculum = CurriculumController()
swarm = ShakespeareSwarm(n_agents=4, base_model=actor)

# Replace standard loader:
loader = CorruptionTextLoader(text, ENC, MICRO_BATCH, HORIZON, CONTEXT_LEN, 
                             device=DEV, corruption_level=0.05)

# In training loop, after computing rewards:
rewards = compute_hierarchical_rewards(gen, ref_tok, ctx, ENC, DEC, shakespeare_stats)

# Update curriculum based on performance:
accuracy = (gen == ref_tok).float().mean().item()
new_corruption = curriculum.update(accuracy)
loader.corruption_level = new_corruption
```

### Step 2: Evolutionary Selection (30 mins)
```python
# Every 50 iterations:
if it % 50 == 0:
    # Select top 10% of generations
    all_rewards_flat = torch.cat([r.flatten() for r in all_R])
    threshold = torch.quantile(all_rewards_flat, 0.9)
    
    # Store high-reward sequences
    for b in range(len(all_R)):
        for k in range(K_SAMPLES):
            if all_R[b][k].mean() > threshold:
                swarm.shared_buffer.append({
                    'context': ctx[b],
                    'generation': G[b, k],
                    'reward': all_R[b][k].mean().item()
                })
```

### Step 3: Advanced Logging (30 mins)
```python
# Add to wandb logging:
wandb.log({
    "curriculum/corruption_level": loader.corruption_level,
    "curriculum/phase": curriculum.phase,
    "curriculum/reconstruction_accuracy": accuracy,
    "emergence/unique_bigrams": count_unique_bigrams(G),
    "emergence/valid_words": count_valid_words(G, shakespeare_stats['vocabulary']),
    "emergence/shakespeare_phrases": count_phrases(G, shakespeare_stats['common_phrases']),
    "swarm/best_agent": best_agent_id,
    "swarm/diversity": compute_swarm_diversity(swarm),
}, step=chars_seen)
```

## Timeline and Milestones

### Day 1-2: Foundation
- Implement corruption curriculum ✓
- Test with 10% corruption level
- Verify reconstruction rewards work

### Day 3-5: Multi-Level Rewards  
- Add hierarchical reward computation
- Precompute Shakespeare statistics
- Track emergence metrics

### Week 2: Curriculum Progression
- Increase corruption gradually
- Monitor phase transitions
- Document emergence patterns

### Week 3: Multi-Agent Swarm
- Deploy 4-agent system
- Implement experience sharing
- Track diversity vs convergence

### Week 4: Full Shakespeare
- Corruption level at 100%
- Pure generation from noise
- Evaluate coherence

## Success Criteria

1. **Week 1**: 80% reconstruction accuracy at 30% corruption
2. **Week 2**: Valid words appear in 50% of generations  
3. **Week 3**: Shakespeare phrases emerge spontaneously
4. **Week 4**: Coherent Shakespeare-like sentences

## Quick Start (Do This Now!)

```bash
# 1. Create a new branch
git checkout -b corruption-curriculum

# 2. Copy this code block into grpo_dense_v2_flash.py after imports:
def corrupt_text(text, corruption_level=0.1):
    if corruption_level == 0:
        return text
    text_list = list(text)
    num_corruptions = int(len(text_list) * corruption_level)
    for _ in range(num_corruptions):
        idx = random.randint(0, len(text_list) - 1)
        if random.random() < 0.5 and idx < len(text_list) - 1:
            text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
        else:
            text_list[idx] = random.choice(text_list)
    return ''.join(text_list)

# 3. In training loop, add corruption:
corrupted_ctx = corrupt_text(DEC(ctx[0]), 0.1)
print(f"Original: {DEC(ctx[0])[:50]}")
print(f"Corrupted: {corrupted_ctx[:50]}")

# 4. Run one iteration to test
modal run grpo_dense_v2_flash.py
```

This is the optimal path - not pure RL, but guided emergence through corruption. Shakespeare will emerge not because we taught it, but because we created conditions where only Shakespeare survives!