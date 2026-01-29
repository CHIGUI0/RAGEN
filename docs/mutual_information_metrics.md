# Mutual Information Metrics for Collapse Detection

This document provides a comprehensive explanation of the mutual information (MI) based metrics used in RAGEN for detecting training collapse phenomena.

## 1. Overview: Two Types of Collapse

We distinguish two collapse phenomena that can occur during RL training of language model agents:

| Collapse Type | Definition | Diagnostic Metric |
|--------------|------------|-------------------|
| **Entropy Collapse** | Model becomes more deterministic per input | Low $H(R \mid X)$ |
| **Template Collapse** | Reasoning becomes input-independent | Low $I(X; R)$ |

**Key Insight**: We compute MI under the batch's empirical input distribution (uniform over prompts), not the true $p(x)$. This is exactly what's needed for diagnosing template collapse.

---

## 2. Design Decision: Partitioning $X$ and $R$

The choice of how to partition the sequence into conditioning context $X$ and reasoning $R$ is crucial for meaningful collapse detection. The partition answers: **"Which segment of generation depends on which segment of input context?"**

### 2.1 Design Goal

We want to diagnose **reasoning template collapse**: whether the reasoning content becomes increasingly input-independent (i.e., ignoring the environment state and producing template-like outputs).

### 2.2 Recommended Partition

For a typical agent turn with structure:
```
[System Prompt] [User: State] [Assistant:] <think> reasoning content </think> <answer> action </answer>
```

We define:

| Variable | Content | Rationale |
|----------|---------|-----------|
| **$X$** | System prompt + User turn (state) + Assistant prefix + `<think>` tag | Everything the model sees *before* generating reasoning content |
| **$R$** | Reasoning content tokens (between `<think>` and `</think>`, **excluding both tags**) | The actual reasoning we want to measure dependency for |

### 2.3 Why Include `<think>` in $X$?

The `<think>` tag should be part of $X$ (conditioning context), not $R$ (reasoning):

1. **Semantic role**: `<think>` is a control token meaning "start generating reasoning" — it's a boundary marker, not reasoning content itself.

2. **Near-constant token**: `<think>` appears identically in every sample, so including it in $R$ would:
   - Add no discriminative information between prompts
   - Dilute entropy/MI statistics with high-probability constant tokens

3. **Clean separation**: With `<think>` in $X$, the partition becomes: "everything before reasoning starts" vs "reasoning content itself"

### 2.4 Why Exclude `</think>` from $R$?

The `</think>` closing tag should also be excluded from $R$:

1. **Structural boundary**: Like `<think>`, it's a format token, not reasoning content.

2. **Format stability signal**: If `</think>` is included in $R$, MI/entropy metrics would conflate:
   - Reasoning content dependency (what we want)
   - Format stability (whether the model reliably closes tags)

3. **Cleaner interpretation**: Excluding both tags means $R$ purely measures "does the reasoning *content* depend on the input state?"

### 2.5 Implementation Mapping

In the codebase, this corresponds to:

| Field | Content |
|-------|---------|
| `first_turn_prompt_ids` | Tokens up to and including `<think>` |
| `first_turn_reasoning_ids` | Reasoning content tokens only (no `<think>`, no `</think>`) |

### 2.6 Extension: Separate Action Analysis (Optional)

For more fine-grained diagnosis, one can compute MI for multiple variables:

| Metric | Measures | Collapse Type |
|--------|----------|---------------|
| $I(X; R_{\text{think}})$ | Does reasoning depend on input? | Reasoning template collapse |
| $I(X; A)$ | Does action depend on input? | Policy/action collapse |
| $I(X; [R_{\text{think}}, A])$ | Does full output depend on input? | Overall mode collapse |

This allows distinguishing:
- "Reasoning is templated but actions are still correct" (reasoning collapse only)
- "Both reasoning and actions are templated" (full policy collapse)

---

## 3. Notation and Definitions

### 3.1 Random Variables

| Symbol | Description |
|--------|-------------|
| $X$ | Input context: system prompt + user turn + assistant prefix + `<think>` |
| $R$ | Reasoning content tokens (between `<think>` and `</think>`, excluding tags) |
| $x_j$ | The $j$-th unique prompt in the batch, $j \in \{1, \ldots, N\}$ |
| $r_{i,k}$ | The $k$-th reasoning sample for trajectory $i$ |
| $N$ | Number of unique prompts in the batch |
| $K$ | Number of reasoning samples per prompt (group size) |

### 3.2 Probability Distributions

| Symbol | Definition | Description |
|--------|------------|-------------|
| $p(r \mid x)$ | $\prod_{t=1}^{T} p_\theta(r_t \mid x, r_{<t})$ | Conditional probability of reasoning $r$ given prompt $x$ under policy $\pi_\theta$ |
| $p_{\text{mix}}(r)$ | $\frac{1}{N} \sum_{j=1}^{N} p(r \mid x_j)$ | Marginal probability under uniform prompt mixture |
| $\hat{p}(x)$ | $\frac{1}{N}$ | Empirical (uniform) distribution over batch prompts |

---

## 4. Core Information-Theoretic Quantities

### 4.1 Conditional Entropy $H(R \mid X)$

**Definition**: The expected uncertainty in the reasoning $R$ given the prompt $X$.

$$H(R \mid X) = -\mathbb{E}_{x \sim \hat{p}(x)} \mathbb{E}_{r \sim p(r|x)} \left[ \log p(r \mid x) \right]$$

**Estimation**: Using sampled (prompt, reasoning) pairs:

$$\hat{H}(R \mid X) = -\frac{1}{NK} \sum_{i,k} \log p(r_{i,k} \mid x_i)$$

**Interpretation**:
- **High $H(R \mid X)$**: Model generates diverse responses for each prompt (stochastic policy)
- **Low $H(R \mid X)$**: Model generates deterministic/repetitive responses (**Entropy Collapse**)

**Code Reference** (`collapse_metrics.py:676-701`):
```python
conditional_entropy = -matched.mean().item()  # H(R|X) estimate
```

### 4.2 Marginal Entropy $H(R)$

**Definition**: The total entropy of reasoning under the marginal distribution.

$$H(R) = -\mathbb{E}_{r \sim p_{\text{mix}}(r)} \left[ \log p_{\text{mix}}(r) \right]$$

**Estimation**: Using the mixture distribution:

$$\hat{H}(R) = -\frac{1}{NK} \sum_{i,k} \log p_{\text{mix}}(r_{i,k})$$

where:

$$p_{\text{mix}}(r) = \frac{1}{N} \sum_{j=1}^{N} p(r \mid x_j)$$

**Code Reference** (`collapse_metrics.py:676-701`):
```python
reasoning_entropy = -marginal.mean().item()  # H(R) estimate
```

### 4.3 Mutual Information $I(X; R)$

**Definition**: The amount of information that the reasoning $R$ contains about the prompt $X$.

$$I(X; R) = H(R) - H(R \mid X)$$

Equivalently:

$$I(X; R) = \mathbb{E}_{x, r} \left[ \log \frac{p(r \mid x)}{p_{\text{mix}}(r)} \right]$$

**Estimation**:

$$\hat{I}(X; R) = \frac{1}{NK} \sum_{i,k} \left[ \log p(r_{i,k} \mid x_i) - \log p_{\text{mix}}(r_{i,k}) \right]$$

**Interpretation**:
- **High $I(X; R)$**: Reasoning is input-dependent (healthy)
- **Low $I(X; R)$**: Reasoning is input-independent (**Template Collapse**)
- **Upper Bound**: $I(X; R) \leq H(X) = \log N$ (when $X$ is uniform)

**Code Reference** (`collapse_metrics.py:564-591`):
```python
def _compute_mi_estimate(self, matched, marginal, N_prompts):
    mi = matched.mean().item() - marginal.mean().item()
    return {
        "collapse/mi_estimate": mi,
        "collapse/mi_upper_bound": math.log(N_prompts),
    }
```

---

## 5. Computation Pipeline

### 5.1 Cross Log-Probability Matrix

For each reasoning $r_{i,k}$ and each prompt $x_j$, we compute the cross log-probability:

$$\ell_j(r_{i,k}) = \log p(r_{i,k} \mid x_j) = \sum_{t=1}^{T} \log p_\theta(r_{i,k,t} \mid x_j, r_{i,k,<t})$$

This forms a matrix $\mathbf{L} \in \mathbb{R}^{NK \times N}$ where:
- Rows index (trajectory, sample) pairs
- Columns index unique prompts

**Code Reference** (`collapse_metrics.py:453-547`):
```python
def _compute_cross_log_probs(self, ...):
    """
    For each reasoning r_{i,k} and each prompt x_j:
    1. Construct sequence [x_j | r_{i,k}]
    2. Compute teacher-forcing log prob
    3. Sum over reasoning tokens → ℓ_j(r_{i,k})
    """
    cross_log_probs = torch.zeros(NK, N, device=device)      # per-token mean
    cross_log_probs_sum = torch.zeros(NK, N, device=device)  # per-sequence sum
```

### 5.2 Matched vs Marginal Log-Probabilities

**Matched**: Log-probability of reasoning under its true prompt:
$$\text{matched}_{i,k} = \ell_i(r_{i,k}) = \log p(r_{i,k} \mid x_i)$$

**Marginal**: Log-probability under uniform prompt mixture:
$$\text{marginal}_{i,k} = \log p_{\text{mix}}(r_{i,k}) = \log \left( \frac{1}{N} \sum_{j=1}^{N} \exp(\ell_j(r_{i,k})) \right)$$

Using log-sum-exp for numerical stability:
$$\text{marginal}_{i,k} = \text{logsumexp}_j(\ell_j(r_{i,k})) - \log N$$

**Code Reference** (`collapse_metrics.py:549-562`):
```python
def _compute_log_prob_stats(self, cross_log_probs, col_ids):
    NK, N = cross_log_probs.shape
    matched = cross_log_probs[torch.arange(NK), col_ids]  # diagonal elements
    marginal = torch.logsumexp(cross_log_probs, dim=1) - math.log(N)
    return matched, marginal
```

---

## 6. Per-Token vs Per-Sequence Metrics

We compute two variants of each metric:

| Variant | Normalization | Use Case |
|---------|--------------|----------|
| **Per-token** (`_est`) | Divide by sequence length | Length-invariant comparison |
| **Per-sequence** (`_seq_est`) | Sum over tokens | Total information content |

### 6.1 Per-Token (Length-Normalized)

$$\bar{\ell}_j(r) = \frac{1}{T} \sum_{t=1}^{T} \log p(r_t \mid x_j, r_{<t})$$

This reduces length bias when comparing reasoning of different lengths.

### 6.2 Per-Sequence (Sum)

$$\ell_j(r) = \sum_{t=1}^{T} \log p(r_t \mid x_j, r_{<t})$$

This captures total log-probability without normalization.

**Metrics Output**:
- `collapse/mi_estimate` — Per-token MI
- `collapse/mi_seq_estimate` — Per-sequence MI
- `collapse/conditional_entropy_est` — Per-token $H(R|X)$
- `collapse/conditional_entropy_seq_est` — Per-sequence $H(R|X)$
- `collapse/reasoning_entropy_est` — Per-token $H(R)$
- `collapse/reasoning_entropy_seq_est` — Per-sequence $H(R)$

---

## 7. Additional Diagnostic Metrics

### 7.1 Retrieval Accuracy

**Definition**: Fraction of samples where the highest cross-log-probability matches the true prompt.

$$\text{Acc} = \frac{1}{NK} \sum_{i,k} \mathbf{1}\left[ \arg\max_j \ell_j(r_{i,k}) = i \right]$$

**Interpretation**:
- **High Accuracy** ($\approx 1$): Reasoning is highly prompt-specific
- **Chance Level** ($\approx 1/N$): Reasoning is prompt-independent (template collapse)

**Code Reference** (`collapse_metrics.py:593-674`):
```python
def _compute_retrieval_accuracy(self, cross_log_probs, col_ids, N_prompts):
    predicted_cols = torch.argmax(cross_log_probs, dim=1)
    correct = (predicted_cols == col_ids).float()
    accuracy = correct.mean().item()
    chance_level = 1.0 / N_prompts
```

**Metrics Output**:
- `collapse/retrieval_accuracy` — Top-1 accuracy
- `collapse/retrieval_accuracy@k` — Top-k accuracy (k ∈ {2, 4, 8})
- `collapse/retrieval_chance_level` — Expected accuracy under random guessing
- `collapse/retrieval_above_chance` — Accuracy improvement over chance

### 7.2 MI Z-Score

**Definition**: Standardized MI using the marginal log-probability standard deviation.

$$\text{MI-ZScore} = \frac{\text{matched} - \text{marginal}}{\sigma_{\text{marginal}} + \epsilon}$$

where $\sigma_{\text{marginal}} = \text{std}(\text{marginal}_{i,k})$ and $\epsilon = 10^{-3}$ for stability.

**Interpretation**: Measures how many standard deviations the matched log-prob is above the marginal. More robust to scale changes during training.

**Code Reference** (`collapse_metrics.py:303-321`):
```python
marginal_std = marginal.std(unbiased=False)
metrics["collapse/mi_zscore"] = ((matched - marginal) / (marginal_std + self.std_eps)).mean().item()
```

### 7.3 EMA-Normalized MI Z-Score

To handle variance drift during training, we track an exponential moving average of the marginal standard deviation:

$$\sigma_{\text{EMA}}^{(t)} = \alpha \cdot \sigma_{\text{EMA}}^{(t-1)} + (1 - \alpha) \cdot \sigma_{\text{marginal}}^{(t)}$$

where $\alpha = 0.9$ (default decay rate).

**Metrics Output**:
- `collapse/marginal_std` — Current batch marginal std
- `collapse/marginal_std_ema` — EMA of marginal std
- `collapse/mi_zscore_ema` — MI Z-score normalized by EMA std

---

## 8. Multi-Turn Sampling Strategies

For multi-turn trajectories, we support two sampling strategies:

### 8.1 Trajectory-Uniform Sampling

**Probability**: $\Pr(m, t) = \frac{1}{M} \cdot \frac{1}{T_m}$

- First sample trajectory $m$ uniformly
- Then sample turn $t$ uniformly within trajectory
- Each trajectory has equal weight regardless of length

**Code Reference** (`collapse_metrics.py:781-814`):
```python
def _sample_trajectory_uniform(self, ...):
    """Each trajectory has equal weight regardless of length."""
    for _ in range(num_to_sample):
        m = np.random.randint(M)  # uniform over trajectories
        t = np.random.randint(turn_counts[m])  # uniform over turns
```

### 8.2 Turn-Uniform Sampling (Disabled by Default)

**Probability**: $\Pr(m, t) = \frac{1}{\sum_m T_m}$

- Uniform over all (trajectory, turn) pairs
- Longer trajectories contribute more samples

---

## 9. Summary of All Metrics

| Metric Name | Formula | Healthy | Collapsed |
|-------------|---------|---------|-----------|
| `mi_estimate` | $\mathbb{E}[\log p(r \mid x) - \log p_{\text{mix}}(r)]$ | High | $\to 0$ |
| `mi_upper_bound` | $\log N$ | — | — |
| `conditional_entropy_est` | $-\mathbb{E}[\log p(r \mid x)]$ | Moderate | Very low |
| `reasoning_entropy_est` | $-\mathbb{E}[\log p_{\text{mix}}(r)]$ | High | $\approx H(R \mid X)$ |
| `retrieval_accuracy` | $\Pr[\arg\max_j \ell_j(r) = \text{true}]$ | High | $\to 1/N$ |
| `retrieval_above_chance` | Accuracy $- 1/N$ | Positive | $\to 0$ |
| `mi_zscore` | $(matched - marginal) / \sigma$ | High | $\to 0$ |
| `marginal_std` | $\text{std}(\text{marginal})$ | Moderate | Very low |

---

## 10. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compute_freq` | 5 | Compute metrics every N steps |
| `micro_batch_size` | 128 | Batch size for cross-scoring |
| `first_turn_enabled` | True | Compute first-turn metrics |
| `multi_turn_enabled` | True | Enable multi-turn sampling |
| `num_samples` | 64 | Number of (x, r) pairs to sample |
| `std_eps` | 1e-3 | Stability constant for std normalization |
| `ema_decay` | 0.9 | EMA decay for cross-time std tracking |

**Configuration in `base.yaml`**:
```yaml
collapse_detection:
  compute_freq: 5
  micro_batch_size: 128
  first_turn_enabled: true
  multi_turn_enabled: true
  num_samples: 64
```

---

## 11. Mathematical Derivations

### 11.1 MI Estimation via Importance Sampling

The mutual information is:

$$I(X; R) = \mathbb{E}_{p(x,r)} \left[ \log \frac{p(r \mid x)}{p(r)} \right]$$

Under the empirical distribution $\hat{p}(x) = 1/N$ (uniform over batch prompts):

$$I(X; R) = \mathbb{E}_{x \sim \hat{p}(x)} \mathbb{E}_{r \sim p(r|x)} \left[ \log \frac{p(r \mid x)}{p_{\text{mix}}(r)} \right]$$

where $p_{\text{mix}}(r) = \sum_j \hat{p}(x_j) p(r \mid x_j) = \frac{1}{N} \sum_j p(r \mid x_j)$.

Monte Carlo estimate with $K$ samples per prompt:

$$\hat{I}(X; R) = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \left[ \log p(r_{i,k} \mid x_i) - \log p_{\text{mix}}(r_{i,k}) \right]$$

### 11.2 Information-Theoretic Identity

The fundamental identity relating our metrics:

$$I(X; R) = H(R) - H(R \mid X)$$

This means:
- If $H(R|X)$ drops but $H(R)$ stays constant → MI increases (good)
- If both $H(R)$ and $H(R|X)$ drop equally → MI stays constant (entropy collapse)
- If $H(R) \to H(R|X)$ → MI → 0 (template collapse)

---
