# IPL Cricket Intelligence Platform — Feature Specification

**Scope**: IPL Only | **Format**: T20 (20 overs per inning)

## Executive Summary

ML-powered cricket intelligence platform for IPL T20 matches with:
- **Tab Transformer learned embeddings** (player & venue, seasonal, chronologically safe)
- **Wicket prediction** (binary classifier)
- **Run prediction** (Unified zero-padded LSTM for all balls)
- **RL agent** (bowling strategy optimization)

**Core Principles**: Simplicity (no wicket types, fielding, toss) | Legal deliveries only | Single unified model | No future data leakage

## 1. Feature Specification & Normalization

### 1.1 Match Context Features

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `inning` | int | 0 - 1 | None | Categorical identifier |
| `over` | int | 0 - 19 | `over / 20` | Normalized to [0, 0.95] |
| `total_balls` | int | 1 - 17 | `total_balls / 10` | Normalized to [0.1, 1.0+] |
| `sin_ball` | float | -1 to 1 | `sin(2π × legal_ball / 6)` | Cyclic encoding of ball position |
| `cos_ball` | float | -1 to 1 | `cos(2π × legal_ball / 6)` | Cyclic encoding of ball position |
| `balls_remaining` | int | 0 - 120 | `balls_remaining / 120` | Balls remaining before ball bowled, Normalized to [0, 1]|
| `score_before` | int | 0 - 300 | ` score_before / 200` | Test during training |
| `wickets_before` | int | 0 - 10 | `wickets_before / 10` | Normalized to [0, 0.9] |
| `current_run_rate` | float | 0 - 66 | `current_run_rate / 36` | Normalized to [0, 1.833] |
| `required_run_rate` | float | 0 - 700 | `required_run_rate / 36`, clip at 2 | Normalized to [0, 2] |
| `target` | int | 0 - 288 | `target / 200` | 0 for inning 1 |
| `phase_pp` | int | 0 - 1 | None | powerplay |
| `phase_middle` | int | 0 - 1 | None | middle |
| `phase_death` | int | 0 -1 | None | death |

**Notes**
- **Formulae**: balls_remaining = (20 - current_over) * 6 - current_ball
- **For Phase**: 0: powerplay (1 - 6 overs) / 1: middle (7-15 overs)/ 2: death (16 - 20 overs)
- **legal_ball**: a dynamically reconstructed index (1-6) that ignores extras
- **inning Switch**: Acts as a binary switch to zero out chase weights
- **Ball Encoding Rationale**: Using `sin_ball` and `cos_ball` instead of raw `ball` captures the cyclic nature (ball 6 → ball 1 transition) and ensures smooth gradients

### 1.2 Extended Context Features

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `last_over_runs` | int | 0 - 37 | `last_over_runs / 36` | Recent momentum |
| `balls_since_boundary` | int | 0 - 120 | `balls_since_boundary / 120` | Pressure indicator |
| `percentage_target_achieved` | float | 0 - 1 | None | 0.0 for inning 1; `score_before / target` for inning 2 |
| `prev_batsman_runs` | float | 0 - 6     | `prev_batsman_runs / 6` | Runs scored by batsman on previous ball        |
| `prev_total_runs`   | float | 0 - 7+    | `prev_total_runs / 6`   | Total runs from previous ball including extras |
| `prev_isWide`       | int   | 0 - 1     | None                    | Indicates previous delivery was a wide         |
| `prev_isNoBall`     | int   | 0 - 1     | None                    | Indicates previous delivery was a no-ball      |
| `prev_is_wicket`    | int   | 0 - 1     | None                    | Indicates wicket fell on previous ball         |


### 1.3 Player Features

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `batsman_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `non_striker_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `bowler_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `is_pacer` | int | 0 - 1 | None | 0 for Spinner |

### 1.4 Venue & Metadata

| Feature | Type | Raw Range | Normalization | Notes |
|---------|------|-----------|---------------|-------|
| `venue_embedding` | vector | [-1, 1] | None | Pre-normalized |
| `season` | int | 2008 - 2030 | `(season - 2007) / 20` | Normalized to [0.05, 0.9] |

## 2. Player & Venue Embeddings

**Tab Transformer learned offline** and placed in dataset as fixed vectors (NOT trainable during model training).

### 2.1 Embedding Strategy

**Initial embedding dimensions**:

- batter embedding → 20
- bowler embedding → 20
- venue embedding → 10

These values are starting points and may be tuned based on validation performance.

**Generation**:
- Computed externally using historical delivery data
- Embeddings by TabTransformer
- Season N embeddings: Use data from seasons 2008 to N-1 only
- Dimension: 20-70 (players), 10-15 (venues)

**Cold Start**:
- New players (That Season): Use role-based league average
- New venues (That Season): Use league-average venue embedding

**Chronological Safety**: Embeddings for 2024 use 2008-2023 data; 2025 uses 2008-2024 data.

## 3. Extras Handling

**Extras Prediction Models**: Extras are not handled probabilistically. Instead, they are predicted directly via machine learning models utilizing the binary target variables (`isWide_target`, `isNoBall_target`) generated during the feature engineering pipeline.

### 3.1 Extras Modeling Strategy

- **Wide Model**: Binary classifier predicting P(wide) ∈ [0, 1] for the current delivery.
- **No-Ball Model**: Binary classifier predicting P(no-ball) ∈ [0, 1] for the current delivery.
- **Input Space**: These models utilize the exact same normalized feature space (Match Context, Player/Venue Embeddings, Sequence Momentum) as the Wicket model.

### 3.2 Simulator Logic

The match outcome emerges from the interaction of these models inside the simulator. Wides and No-balls disrupt the standard legal delivery flow and inject sequence momentum (`prev_*` features).

1. **Predict Extras**: Query the Wide and No-Ball models before the Wicket/Run models.
   - If **Wide** occurs: Add 1 run, set `prev_isWide` = 1, repeat the legal ball index.
   - If **No-Ball** occurs: Add 1 run, set `prev_isNoBall` = 1, repeat the legal ball index, and trigger the Free Hit state for the next delivery.
2. **Get Core Predictions**: If the delivery is legal (or a legal Free Hit), execute the Wicket model → execute the LSTM Run model.
3. **Update State**: Append results to sequence history, update scoreboards, and process strike rotation.

**Edge Cases & Constraints**: 
- Free hit state explicitly disables the Wicket model for that specific delivery. 
- Consecutive or multiple wides are possible on the same ball index. 
- A delivery cannot simultaneously be a legal wide and a no-ball (simulator logic dictates No-ball takes precedence in sequence logging).

## 4. Model Architecture

### 4.1 Wicket Model
- **Type**: Binary classifier (wicket / no wicket)
- **Input**: All normalized features from Sections 1.1-1.4
- **Output**: P(wicket) ∈ [0, 1]
- **Architecture**: During Building Phase

### 4.2 Unified LSTM Run Model (All Balls)
- **Type**: Sequence-based multi-class classifier
- **Input**: Last 30 balls (30 × feature_dim)
- **Output**: P(runs) for {0, 1, 2, 3, 4, 5, 6}
- **Main Target Variable**: `batsman_runs_target`
- **Architecture**: During Building Phase
- **Padding**: Left-zero-padding used for dummy historical deliveries when predicting balls 1 through 29

**Sequence Injection Strategy**:
The `prev_*` tracking features (runs, extras, wickets from the immediate preceding ball) are concatenated with the static tabular embeddings at every time step of the LSTM sequence. This allows the model to capture immediate momentum alongside match context.

**Model Transition**:
```
Sequence predicting Ball 1:  [Pad, Pad, ..., Pad (x29), Ball_1_Features]
Sequence predicting Ball 2:  [Pad, Pad, ..., Pad (x28), Ball_1_Features, Ball_2_Features]
...
Sequence predicting Ball 30: [Ball_1, Ball_2, Ball_3, ..., Ball_30_Features]
Sequence predicting Ball 31: [Ball_2, Ball_3, Ball_4, ..., Ball_31_Features]
```
LSTM sequence resets at start of each inning

## 5. Match Simulation Flow

### Target Variables Definition

Each ML component has a clearly defined target variable:

Each ML component has a clearly defined target variable matching the feature pipeline:

1. Wicket Model → target: `is_wicket_target`
2. Run Model → target: `batsman_runs_target`
3. Wide Model → target: `isWide_target`
4. No Ball Model → target: `isNoBall_target`
5. RL Bowler Selection Model → target: reward signal based on match outcome

### 5.1 Strike Rotation Logic

| Event | Action |
|-------|--------|
| Odd runs (1, 3, 5) | Swap striker ↔ non-striker |
| End of over | Swap striker ↔ non-striker |
| Wicket | New batter takes striker's end |
| Even runs (0, 2, 4, 6) | No swap |

### 5.2 Bowling Constraints

| Constraint | Rule |
|------------|------|
| Max overs | 4 per bowler per inning |
| Consecutive | Cannot bowl overs N and N+1 in same inning |
| Validity | Must be in playing XI |
| Minimum | < 5 bowlers allowed (flexibility) |

### 5.3 Match Structure

- **Format**: 2 inning, 20 overs each, 120 legal balls per inning
- **Termination**: 20 overs complete OR 10 wickets fall OR target reached (inning 2)
- **Target**: `inning_1_score + 1` for inning 2; `0` for inning 1
- **RL Control**: Selects bowler at start of each over for both teams

## 6. Reinforcement Learning Environment

### 6.1 RL Agent Role

**Objective**: Select optimal bowler at the start of each over to maximize win probability.

- **Decision frequency**: Once per over (~20 per inning)
- **Scope**: Controls bowling for both teams

### 6.2 RL State Space

**To be finalized during RL training phase.** Initial proposal includes:

**Core Features**: All match context, player, venue features from Section 1

**Bowler-Specific Features**:
- `available_bowlers`: List of valid bowler IDs
- `overs_bowled_per_bowler`: Dict of overs bowled
- `overs_left_per_bowler`: Dict of overs remaining

### 6.3 RL Action & Reward

- **Action**: Select `bowler_id` from `available_bowlers`
- **Validation**: Simulator enforces constraints; invalid action → negative penalty
- **Reward**: To be decided during training

## 7. Team Composition

### 7.1 Playing XI & Batting Order

| Mode | Playing XI | Batting Order |
|------|------------|---------------|
| **Developer** | Fixed from database | Historical order |
| **User** | Custom selection | Custom order (1-11) |

**Validation**: All 11 players must have embeddings for the season.

### 7.2 Bowling Order

**Not predefined** — RL agent selects bowler each over subject to constraints in Section 6.2.

## 8. Dataset Requirements

### 8.1 Data Scope

- **League**: IPL only (2008-present)
- **Granularity**: Ball-by-ball deliveries
- **Structure**: Data processed in overlapping sliding windows of exactly 30 balls.
- **Extras Handling (Wides)**: To mirror the simulator's logic of repeating the ball index, wide deliveries are explicitly row-expanded (repeated) in the dataset before reconstructing the legal ball count.

### 8.2 Chronological Split

| Set | Description |
|-----|-------------|
| **Training** | Seasons 2010 to 2024 |
| **Validation** | Season 2025 (First Half) |
| **Test** | Season 2025 (Second Half) |

**Critical**: No future data leakage in embeddings or features.<br>
**Note**: 2023 and 2024 are critical training years to ensure the model adapts to the modern ~185+ par score meta introduced by the Impact Player rule. Validation and Test are split mid-season 2025 to evaluate the model's performance on pitch degradation.

### 8.3 Data Quality

**Required Fields**: Player IDs, venue ID, match outcome, ball outcome (runs/wicket), over/ball counts
**Action if Missing**: Exclude ball/match from dataset

## 9. Implementation Pipeline

### Phase 1: Data Pipeline
1. Extract ball-by-ball data from IPL sources
2. Clean data, expand wide deliveries into multiple sequence rows (repeating the ball index), and reconstruct a strict 1-6 legal ball count
3. Engineer features (`phase`, `last_over_runs`, etc.)
4. Create chronological train/val/test splits

### Phase 2: Embedding Generation (External)
1. Compute player embeddings (batters, bowlers) from historical data
2. Compute venue embeddings
3. Generate league-average embeddings for cold-start
4. Place embeddings in dataset as fixed vectors
5. Ensure chronological constraint (season N uses data up to N-1)

### Phase 3: ML Models
1. Train wicket model (binary classifier with class weights)
2. Train unified LSTM using overlapping sliding window sequences to prevent data starvation (using zero-padding for early balls)
3. Validate on validation set, tune hyperparameters
4. Evaluate on test set

### Phase 4: Simulator
1. Integrate wicket and LSTM run models
2. Implement strike rotation and bowling constraints
3. Build full 2-inning match loop
4. Test with historical match replays

### Phase 5: RL Environment
1. Wrap simulator as RL environment
2. Define state encoding and action space
3. Train RL agent (methodology TBD)
4. Evaluate vs historical bowling orders

### Phase 6: Deployment
1. Build microservices for models (wicket, LSTM, RL)
2. Create API endpoints for simulation and prediction
3. Deploy embedding lookup service
4. Integrate with frontend

## 10. Model Outputs Summary

| Model | Output |
|-------|--------|
| **Wicket** | P(wicket) ∈ [0, 1] |
| **Unified LSTM** | P(runs) for {0,1,2,3,4,5,6} |
| **RL Agent** | bowler_id from available_bowlers |

**Excluded**: Wicket types (bowled/caught/LBW), fielding positions, shot types

## 11. Key Design Decisions

### Finalized
- **LSTM Sequence Strateg**y: Overlapping sliding windows of 30 balls to massively increase training data volume and capture multi-over momentum.
- **Player Representation**: Concatenation of TabTransformer interactions and normalized season stats at the sequence input.
- **inning Handling**: 0 for inning 1
- **Main Model**: Multi-model cricket simulation system with specialized models interacting through a simulator
- **Ball Encoding**: Cyclic (sin/cos) to capture position
- **Embeddings**: TabTransformer learned externally, not trainable
- **Bowling Flexibility**: < 5 bowlers allowed if needed
- **Consecutive Overs**: Constraint applies within inning only

### To Be Decided During Training
- **Run normalization**: Test min-max / ÷100 / ÷200 / standardization
- **RL reward structure**: Win/loss only vs intermediate rewards
- **RL training methodology**: Algorithm, exploration, opponent generation

## 12. Quick Reference

### Normalization Cheat Sheet
```
over                        → over / 20
total_balls                 → total_balls / 10
legal_ball                  → sin(2π × legal_ball/6), cos(2π × legal_ball/6)
balls_remaining             → balls_remaining / 120
wickets_before              → wickets_before / 10
balls_since_boundary        → balls_since_boundary / 120
season                      → (season - 2007) / 20
runs (score/target)         → score / 200, target / 200
last_over_runs              → last_over_runs / 36
current_run_rate            → current_run_rate / 36
required_run_rate           → required_run_rate / 36 (clipped at 2)
prev_batsman_runs           → prev_batsman_runs / 6
prev_total_runs             → prev_total_runs / 6
embeddings                  → None (already normalized)
phase                       → One-hot encoding (phase_pp, phase_middle, phase_death)
inning                      → None (0 or 1)
percentage_target_achieved  → 0.0 (1st inning) or current/target (2nd inning)
```

### Model Flow
```
For each ball index:
    1. Predict Wide (ML Model)
        If Wide == 1:
            → add 1 run
            → set prev_isWide = 1
            → repeat ball index

    2. Predict No-Ball (ML Model)
        If No-Ball == 1:
            → add 1 run
            → set prev_isNoBall = 1
            → set free_hit_state = True
            → repeat ball index

    3. Get Core Predictions (If Legal or Legal Free Hit)
        If free_hit_state == True:
            → wicket = 0
            → Run LSTM Runs model
            → set free_hit_state = False (reset for next delivery)
        Else:
            → Predict Wicket (ML Model)
            If Wicket == 1:
                → runs = 0
                → new batter takes strike
            Else:
                → Run LSTM Runs model

    4. State Update
        → Update scoreboard, balls remaining, sequence history (`prev_*` features)
        → Process strike rotation logic
        If over complete:
            → RL Agent selects next bowler

    *Assumption: Wicket deliveries yield 0 runs in simulation.*
```

### Constraints Validation
```
✓ Max 4 overs per bowler per inning
✓ No overs N and N+1 by same bowler (same inning)
✓ Bowler must be in playing XI
✓ < 5 bowlers permitted (flexibility)
```