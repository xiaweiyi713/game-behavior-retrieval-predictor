# Game Behavior Retrieval Predictor

A retrieval-based game behavior prediction system for tactical shooter games (Delta Force). Given a 20-second structured game log, the system predicts the target player's future 5-second actions and generates natural language behavior descriptions in Chinese.

This project was originally developed for the 2026 Tencent Game Security Technology Competition (Game Security AI Track - Final Round).

## Task Definition

- **Input**: 20-second structured game log containing multiple players' states and events
- **Output**: Natural language description of the target player's future 5-second behavior in Chinese
- **Challenge**: Not pure text generation, but structured future behavior prediction converted to controlled Chinese output

## Architecture

The system uses a layered approach instead of end-to-end generation:

### Layer 1: Log Parsing & Sample Reconstruction
Unified parsing of training and test logs, identifying:
- Target player identity, decision type
- Historical and future event windows
- Key events: movement, aiming, shooting, skills, hits, grenades, looting

### Layer 2: Situation Feature Engineering
Two types of features per sample:
- **Text summary features**: Compressed representation of high-information events, targeting, aiming, action chains
- **Numeric statistics**: Enemy-teammate distances, visible enemy count, recent 3s action density, recent 5s skill usage, nearby teammate status

### Layer 3: TF-IDF + Numeric Similarity Retrieval with Future Action Slots
- Stable retrieval framework combining TF-IDF text similarity with numeric cosine similarity
- **Future action slot re-ranking**: Instead of just comparing text similarity, compares "which candidate has the most similar future action chain"
- Three future action slots: `future_action_1`, `future_action_2`, `future_action_3`

### Layer 4: Action/Grenade Conditional Router
Per-sample routing for high-variance action types:
- For `Action/Grenade` types: generates both `base` and `tuned` results
- Trains a lightweight routing model to decide per-sample which result to use
- Stable types (`BeingResuce/Fire/Looting/SkillStart`) use baseline retrieval

### Layer 5: Local Structure Enhancers (Experimental)
- Action tail-sentence classifier
- Grenade second-third sentence combination predictor
- Currently preserved in code but not enabled by default

### Layer 6: Controlled Chinese Text Output
Generates structured Chinese behavior descriptions based on retrieved matches.

## Directory Structure

```
.
├── gameai2026/
│   ├── __init__.py
│   ├── parser.py                    # Log parsing, sample building, target player identification
│   ├── features.py                  # Text summaries and numeric situation features
│   ├── label_builder.py             # Future 5s label construction and sentence splitting
│   ├── retrieval_baseline.py        # Retrieval, scoring, action slot re-ranking framework
│   ├── action_grenade_router.py     # Action/Grenade per-sample conditional router
│   ├── local_structure_enhancer.py  # Local structure enhancer (action tail + grenade pair)
│   └── text_generator.py           # Controlled Chinese text output
├── scripts/
│   ├── build_training_artifacts.py             # Build training records from raw data
│   ├── augment_train_records_with_future_actions.py  # Augment records with future action fields
│   ├── predict_test_set.py                     # Basic version test set prediction
│   ├── predict_test_set_action_grenade_router.py     # Official version test set prediction
│   ├── evaluate_holdout.py                     # Holdout evaluation
│   ├── evaluate_action_grenade_router.py       # Conditional router evaluation
│   ├── merge_prediction_variants.py            # Merge prediction variants
│   └── qa_final_submission.py                  # Answer sheet QA validation
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- NumPy, Pandas, SciPy, scikit-learn, openpyxl

## Usage

### 1. Build training records

```bash
python scripts/build_training_artifacts.py
python scripts/augment_train_records_with_future_actions.py
```

### 2. Evaluate on holdout

```bash
python scripts/evaluate_action_grenade_router.py \
  --train-records outputs/train_records_full_with_future.csv \
  --eval-csv outputs/holdout_eval.csv \
  --router-oof-csv outputs/router_oof.csv \
  --holdout-per-type 100 \
  --router-oof-folds 3 \
  --seed 42
```

### 3. Generate test set predictions

```bash
python scripts/predict_test_set_action_grenade_router.py \
  --train-records outputs/train_records_full_with_future.csv \
  --pred-csv outputs/predictions.csv \
  --answer-xlsx outputs/final_answer.xlsx \
  --router-oof-csv outputs/router_oof_full.csv \
  --router-oof-folds 3 \
  --seed 42
```

## Key Design Decisions

- **Retrieval over generation**: Avoids free-form LLM generation instability by grounding predictions in retrieved similar scenarios
- **Future action slots**: Transforms "text continuation" into "action chain prediction" for more accurate matching
- **Conditional routing**: Different prediction strategies for different action types rather than one-size-fits-all
- **Controlled output**: Structured Chinese generation with template constraints for stable answer format
- **Iterative refinement**: Evolved from stable baseline -> future action augmentation -> typed fusion -> per-sample routing

## Results

| Version | seed=42 | seed=43 |
|---------|---------|---------|
| Base stable baseline | 0.6517 | 0.6600 |
| Typed mix (previous) | 0.6683 | 0.6733 |
| Router mix (current) | 0.6717 | 0.6750 |

## License

MIT
