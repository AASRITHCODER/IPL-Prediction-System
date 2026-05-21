# IPL Dataset Loader Documentation

## Purpose

`IPLDataset` is the primary PyTorch sequence dataset used for training the IPL Cricket Intelligence Platform models.

The dataset is designed for:

- Sequential cricket prediction
- Multi-task learning
- Chronologically safe training
- Future embedding integration
- RL-compatible state generation

It converts processed IPL ball-by-ball data into fixed-length sequential tensors suitable for LSTM-based architectures.

The current delivery target is NEVER included inside the input sequence targets/features.
Only historically available contextual information is used.

---

# Core Design Principles

## 1. No Future Data Leakage

Training data is filtered chronologically:

```python
df = df[df["season"] < train_year]
```

This ensures:

* 2025 models only train on <= 2024
* embeddings remain historically safe
* validation/test realism maintained

---

## 2. Sequence-Based Learning

The dataset creates rolling ball-by-ball sequences.

Each sample contains:

```text
Last N deliveries -> Predict current delivery outcome
```

Current architecture:

```text
sequence_length = 30
```

---

## 3. Match & Innings Isolation

Sequences are strictly separated by:

```python
(matchId, inning)
```

This prevents illegal sequence transitions such as:

* match A -> match B
* innings 1 -> innings 2

---

## 4. Left Zero Padding

Early deliveries in innings use left-side zero padding.

Example:

```text
Ball 1:
[0,0,0,...,Ball_1]

Ball 2:
[0,0,...,Ball_1,Ball_2]
```

Padding index:

```text
0 reserved for embeddings
```

---

# Dataset Architecture

## Input Types

The dataset separates:

| Type                 | Description                            |
| -------------------- | -------------------------------------- |
| Numerical Features   | Continuous/contextual cricket features |
| Categorical Features | Integer IDs for embeddings             |

This architecture supports:

* Embedding layers
* TabTransformer integration
* Hybrid sequence models

---

# Leakage Prevention Strategy

The dataset intentionally separates:

* input context features
* prediction targets

Features prefixed with `prev_` are generated during feature engineering using strictly historical information (`shift(1)` logic).

This prevents current-ball target leakage.

---

# Numerical Features

Current numerical features:

```python
[
    "inning",
    "over",
    "total_balls",
    "balls_remaining",
    "phase_pp",
    "phase_middle",
    "phase_death",
    "target",
    "last_over_runs",
    "balls_since_boundary",
    "prev_batsman_runs",
    "prev_isWide",
    "prev_isNoBall",
    "prev_is_wicket",
    "prev_total_runs",
    "score_before",
    "wickets_before",
    "percentage_target_achieved",
    "current_run_rate",
    "required_run_rate",
    "is_pacer",
    "season",
    "sin_ball",
    "cos_ball",
]
```

---

# Categorical Features

Categorical features are integer encoded.

Current categorical IDs:

```python
[
    "batter_id",
    "non_striker_id",
    "bowler_id",
    "venue_id",
]
```

---

# Stable Global Mapping System

Mappings are loaded from persistent global files:

```python
all_players.json
all_venues.json
```

This guarantees:

* consistent embedding indices
* stable inference behavior
* deployment safety
* train/validation/test consistency

---

# Embedding Index Rules

| Value | Meaning      |
| ----- | ------------ |
| 0     | Padding      |
| >=1   | Valid entity |

Separate vocabularies are maintained for:

* players
* venues

Player vocabulary is shared across:

* batter
* non_striker
* bowler

---

# Multi-Task Learning Targets

The dataset currently supports 4 prediction tasks.

| Target              | Type        |
| ------------------- | ----------- |
| batsman_runs_target | Multi-class |
| isWide_target       | Binary      |
| isNoBall_target     | Binary      |
| is_wicket_target    | Binary      |

---

# Returned Dataset Structure

Each dataset sample returns:

```python
{
    "numerical_features": tensor,
    "categorical_features": tensor,

    "runs_target": tensor,
    "wide_target": tensor,
    "noball_target": tensor,
    "wicket_target": tensor,
}
```

---

# Tensor Shapes

## Numerical Features

```python
(sequence_length, numerical_feature_dim)
```

Example:

```python
(30, 24)
```

---

## Categorical Features

```python
(sequence_length, categorical_feature_dim)
```

Example:

```python
(30, 4)
```

---

# Target Shapes

## Runs

```python
scalar integer
```

Range:

```text
0-6
```

---

## Wide / NoBall / Wicket

```python
scalar binary float
```

Range:

```text
0 or 1
```

---

# Sequence Construction Logic

For each delivery:

```python
start_idx = max(0, idx - sequence_length + 1)
```

The dataset extracts:

```python
previous deliveries + current delivery context
```

Then applies:

```python
left zero padding
```

if sequence length is insufficient.

---

# Data Ordering

Data is sorted using:

```python
["matchId", "inning", "over", "total_balls"]
```

This preserves temporal delivery order.

---

# Memory Strategy

Current implementation pre-builds all sequences into RAM.

Advantages:

* simpler implementation
* faster training iteration
* easier debugging

Disadvantages:

* higher memory usage
* less scalable for very large datasets

Chosen because:

* IPL dataset size is manageable
* project timeline prioritizes execution speed

---

# Metadata Stored

The dataset stores important metadata for model initialization.

## Feature Metadata

```python
self.numerical_dim
self.categorical_dim
```

---

## Embedding Metadata

```python
self.num_players
self.num_venues
```

Used for:

```python
nn.Embedding(num_embeddings, embedding_dim)
```

---

# Architectural Decisions

## Why Integer IDs Instead of One-Hot Encoding?

One-hot encoding was intentionally avoided because:

* dimensional explosion
* inefficient embeddings
* poor scalability

Integer IDs are required for embedding-based architectures.

---

## Why Separate Numerical & Categorical Features?

This enables:

* embedding layers
* transformer compatibility
* cleaner feature processing
* easier model experimentation

---

## Why Multi-Task Targets?

The simulator requires prediction of:

* runs
* wickets
* wides
* no-balls

A shared encoder with multiple heads is more efficient and realistic.

---

# Future Planned Model Architecture

Planned architecture:

```text
Categorical IDs
    ↓
Embedding Layers
    ↓
Concatenate with Numerical Features
    ↓
LSTM Encoder
    ↓
Multi-Task Heads
```

Prediction heads:

* Runs Head
* Wicket Head
* Wide Head
* NoBall Head

---

# Current Limitations

## 1. High RAM Usage

Entire sequence dataset is stored in memory.

Future optimization may move sequence generation into `__getitem__()`.

---

## 2. No Embedding Generation Yet

Current dataset only prepares integer IDs.

Embedding training occurs later.

---