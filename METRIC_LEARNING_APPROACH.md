# Metric Learning for Building Age Estimation
**Project:** Age of Buildings — Amsterdam & Madrid  
**Date:** 2026-05-25

---

## The Core Problem

We want to estimate building construction dates from Landsat spectral time series. The challenge is:

1. **City-specificity**: The mapping from spectral features to age is not the same across cities. Amsterdam's 1960s brick rowhouses look spectrally different from Madrid's 1960s concrete apartments.
2. **Few labels**: We can only afford to label a small number of buildings per new city.
3. **Generalizability**: The model should work for any city worldwide, not just Amsterdam and Madrid.

A direct regression approach (predict age from features) trained on Amsterdam will fail in Madrid because the absolute feature values differ. We need to learn something more fundamental.

---

## The Key Insight

Instead of learning "what does a 1965 building look like?", we learn "what makes two buildings the same age?"

This is a **relative** judgment. If the aging process — weathering, material degradation, urban densification around older buildings — produces consistent spectral signals regardless of city, then a model trained to recognize *same-era vs different-era* pairs in Amsterdam should generalize to Madrid.

** Metric learning** is **one possible approach** to address this learning problem.  There are other possible approaches. you are free to explore alternatives!

---

## How Metric Learning Works

### Step 1: Feature Engineering
For each building, compress its full spectral time series (6 bands × ~43 years) into a compact feature vector. Part of your own design will involve making an intelligent choice of feature set.  We have made suggestions in Notebook 3, but you are not obliged to hold to them.

### Step 2: Train an Embedding Network on Amsterdam
Train a small neural network `f()` that maps the feature vector to a lower-dimensional **embedding space** (e.g., 16 dimensions):

```
f: R^30 → R^16
```

The network is trained with **triplet loss**:

```
For each triplet (anchor, positive, negative):
  - anchor:   a building with known construction year
  - positive: another building in the same class
  - negative: a building from a different class

Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

Where `d()` is Euclidean distance in embedding space. The loss pushes same-era buildings together and different-era buildings apart.

**Training data needed:** ~200–500 Amsterdam buildings with known construction years. Triplets are generated automatically from these.

### Step 3: Apply Embedding to Madrid
Once trained, apply `f()` to all Madrid buildings. Each Madrid building gets mapped to the same 16-dimensional embedding space — without any Madrid labels.

### Step 4: Few-Shot Classification in Madrid
With just 20–50 labeled Madrid buildings, classify the rest using one of:

**Option A — k-Nearest Neighbors:**  
For each unlabeled Madrid building, find its k nearest neighbors among the labeled Madrid buildings in embedding space. Assign the majority era class.

**Option B — Prototype Matching (simplest, works with very few labels):**  
Compute the mean embedding for each era class from the labeled buildings (the "prototype"). Classify each unlabeled building by its nearest prototype. Works with as few as 2–3 labeled examples per class.

**Option C — Lightweight Classifier:**  
Train a logistic regression or small decision tree on the labeled Madrid embeddings. The embedding does the heavy lifting; the classifier just learns the Madrid-specific offset.

### Step 5: Active Learning (optional, maximizes label efficiency)
Rather than randomly choosing which Madrid buildings to label:
1. Apply the embedding to all Madrid buildings
2. Fit a Gaussian Process on the labeled subset
3. Identify buildings with the highest prediction uncertainty
4. Label those buildings specifically
5. Retrain

This gives more accuracy per label than random selection.

---

## Why This Generalizes

The embedding network is trained to ignore features that vary by city but not by age (different baseline reflectance, different climate, different atmospheric conditions). It amplifies features that vary by age regardless of city (temporal change patterns, weathering rates, spectral trajectories).

If the physical aging process is similar across cities — which is a reasonable assumption for modern construction materials — the embedding space will be shared, and few Madrid labels are enough to orient the classifier.

---

## Validation Strategy

### Within Amsterdam (cross-validation)
- Hold out 20% of Amsterdam buildings as test set
- Train on 80%, evaluate on 20%
- Measure: mean absolute error in years, accuracy per decade class

### Cross-city transfer
- Train entirely on Amsterdam
- Apply embedding to Madrid
- Use 20 labeled Madrid buildings to fit a prototype classifier
- Evaluate on remaining labeled Madrid buildings
- Compare to: (a) random baseline, (b) direct regression trained on Amsterdam

### Embedding quality check
After training on Amsterdam, compute embeddings for labeled Madrid buildings and visualize with t-SNE or PCA. If same-era Madrid buildings cluster together in the Amsterdam-trained space, transfer is working. If not, more Madrid labels or a domain adaptation step is needed.

---

## Architecture Recommendation

For the feature vector size (~30 inputs) and dataset size (~200–500 buildings), a simple network is appropriate:

```
Input (30) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16) → L2 normalize → Embedding
```

- L2 normalization constrains embeddings to a unit hypersphere, which stabilizes triplet loss training
- No dropout needed at this scale
- Train with Adam optimizer, lr=0.001, for ~100 epochs
- Batch size: 32–64 triplets per batch

Total parameters: ~3,000. This is intentionally small — the feature engineering does the heavy lifting, not the network.

---

## Era Classes

Suggested discretization for Amsterdam and Madrid:

| Class | Years | Historical context |
|-------|-------|--------------------|
| Pre-war | before 1940 | Traditional masonry, pre-modernist |
| Post-war reconstruction | 1940–1959 | Rapid rebuilding, uniform materials |
| Modernist expansion | 1960–1979 | Concrete, prefab, urban growth |
| Late 20th century | 1980–1999 | Mixed construction, renovation era |
| Contemporary | 2000–present | Modern materials, energy efficiency codes |

5 classes is a reasonable starting point. Boundaries can be adjusted based on local urban history.

---

## Comparison to Alternative Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Metric learning** (recommended) | Generalizes across cities, few labels needed, interpretable features | Assumes shared aging process |
| Direct GP regression | Uncertainty quantification, works with small N | City-specific, breaks down in new city |
| Transfer learning (fine-tune NN) | Can capture complex patterns | Needs large N per city, black box |
| Hierarchical Bayesian | Principled, handles city offsets | Complex to implement, slow |

---

## Implementation Steps

1. Extract feature vectors from parquet dataset (one row per building per year → aggregate to one row per building)
2. Assign era class labels from construction year
3. Generate triplets from Amsterdam labeled buildings
4. Train embedding network
5. Visualize Amsterdam embeddings (t-SNE) to verify clustering by era
6. Apply to Madrid, visualize
7. Fit prototype classifier on 20–50 labeled Madrid buildings
8. Evaluate on held-out Madrid buildings
