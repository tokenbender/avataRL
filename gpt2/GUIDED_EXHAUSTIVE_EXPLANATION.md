# Guided Exhaustive GRPO: Smart Exploration with Context

## Key Improvements Over Previous Versions

### 1. **100-Character Sliding Window Context**
```python
CONTEXT_LEN = 100  # Instead of just 1 character!
```

**Why this matters:**
- With "Hamlet, Prince of Denmark", we can predict meaningful continuations
- Not just "H" → ?, but full context → informed prediction
- Eliminates the degenerate space-only solution

### 2. **3-Gram Reference Model**
```python
class TrigramRef(nn.Module):
    # Looks at 3 characters: "the" → likely next char
```

**Advantages over bigram:**
- Captures common word patterns ("the", "ing", "and")
- Provides stronger initial signal
- Better baseline for comparison

### 3. **Hybrid Generation Strategy**

Instead of exhaustive on ALL 65 characters:

```python
# Smart selection based on uncertainty
if model_is_uncertain:
    try top 10 likely chars + 15 random exploration
else:
    just try top 5-10 most likely
```

**Benefits:**
- Focuses compute on promising options
- Still explores when uncertain
- Scales better with vocabulary size

### 4. **Enhanced Reward Function**

```python
def enhanced_reward(gen, ref, model_logits, ref_logits):
    exact_match = (gen == ref).float()  # 100% for correct
    
    # NEW: Partial credit if char is in reference's top-5
    in_top5 = is_generated_char_in_ref_top5()
    partial_reward = in_top5 * 0.2  # 20% for reasonable guess
    
    return exact_match + partial_reward * (1 - exact_match)
```

**Why this helps:**
- Rewards "reasonable but wrong" predictions
- E.g., predicting 'a' when 'e' was correct but both are common after 'th'
- Provides denser learning signal

### 5. **Adaptive Exhaustive Search**

```python
def select_exhaustive_chars(model, context, V):
    entropy = compute_entropy(model(context))
    
    if entropy > threshold:  # High uncertainty
        # Try more characters exhaustively
        return top_k_chars + random_exploration
    else:  # Confident prediction
        # Just refine top choices
        return top_k_chars
```

**Smart exploration:**
- Measures model uncertainty via entropy
- Explores more when confused
- Exploits when confident

## Architecture Improvements

### Enhanced GPT Model
```python
class ContextualGPT(nn.Module):
    def __init__(self):
        # 8 layers (vs 6)
        # 8 heads (vs 6)  
        # 512 dims (vs 384)
        # Dropout for regularization
        # Proper causal masking
```

### Better Data Loading
```python
class ContextualTextLoader:
    # Provides sliding windows of text
    # Each sample has full context + target sequence
    # No more single-character contexts!
```

## Training Strategy

### 1. **Split Batch Processing**
```python
EXHAUSTIVE_RATIO = 0.5  # 50% exhaustive, 50% normal

# High uncertainty contexts → exhaustive search
# Low uncertainty contexts → normal generation
```

### 2. **Guided Generation**
```python
def guided_generate(model, context, exhaustive_chars=None):
    if exhaustive_chars:
        # Try each specified character as first
        # Then continue normally
    else:
        # Standard autoregressive generation
```

### 3. **Better Metrics**
- Tracks entropy (uncertainty)
- Monitors max/min rewards
- Shows reward distribution
- Generates samples with context

## Expected Improvements

### Before (1-char context):
```
Context: "H"
Best char: " " (space) - 15.7%
Overall: 1.5% (random performance)
Model collapsed to always predicting space
```

### After (100-char context):
```
Context: "...whether 'tis nobler in the mind to suffer\nThe slings and arrows of"
Best continuation: " outrageous fortune" - 65%+
Overall: 25-35% accuracy expected
Model learns actual patterns
```

## Why This Works Better

1. **Context Solves Degeneracy**
   - "H" → many valid next chars
   - "Hamlet" → fewer valid options
   - "To be or not to" → very specific continuations

2. **3-gram Captures Patterns**
   - "th" → 'e' (the)
   - "in" → 'g' (ing)  
   - "an" → 'd' (and)

3. **Adaptive Search Balances**
   - Exploration when needed
   - Exploitation when confident
   - Computational efficiency

4. **Partial Rewards Guide Learning**
   - Not just right/wrong
   - Degrees of correctness
   - Smoother gradient signal

## Running the Improved Version

```bash
modal run --detach grpo_guided_exhaustive.py
```

## Key Hyperparameters

```python
CONTEXT_LEN = 100          # Full context window
EXHAUSTIVE_RATIO = 0.5     # Balance exhaustive/normal
UNCERTAINTY_THRESHOLD = 2.0 # When to explore more
PARTIAL_LP = 0.1           # Partial credit weight
BETA_KL = 5e-3            # Higher for 3-gram reference
```

## Expected Training Dynamics

### Early (0-50 iterations):
- High entropy (model uncertain)
- Lots of exhaustive exploration
- Rapid improvement from 1.5% → 10-15%

### Mid (50-200 iterations):
- Entropy decreases
- More targeted exploration
- 15% → 25-30% accuracy

### Late (200-500 iterations):
- Low entropy (confident model)
- Mostly exploitation
- Fine-tuning to 30-40%+ accuracy

This approach combines the best of both worlds:
- **Exhaustive search** where it matters (uncertain predictions)
- **Efficient generation** where model is confident
- **Rich context** for meaningful predictions
- **Better reference** for stronger initial signal