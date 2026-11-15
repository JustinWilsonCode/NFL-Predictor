# NFL Game Outcome Prediction

Predicts whether the home team wins an NFL game from pre-game information, using
[NFLverse](https://github.com/nflverse/nfldata) data (1999-present), and compares
against the Vegas point spread. Course project for *Perspectives on Data Science*.

## Results (test set, 2023+ seasons, preliminary)

| Model | Accuracy |
|---|---|
| Always predict home | 53.7% |
| Logistic regression | 62.9% |
| Gradient boosting | 64.4% |
| Random forest | 65.0% |
| Vegas spread | 67.6% |

## Method
- Chronological split: train 1999-2020, validation 2021-22, test 2023+.
- Leakage-free features (all computed before kickoff): rolling 5-game and season win %,
  points for/against, Elo rating difference, rest-day difference, neutral-site flag.
- Models refit on train+validation before the single test evaluation; exact McNemar test vs Vegas.

## Reproduce
```bash
pip install -r requirements.txt
mkdir data && curl -L -o data/games.csv https://github.com/nflverse/nfldata/raw/master/data/games.csv
python baseline.py      # trains/evaluates, writes results.json and figures/
```

## Files
- `baseline.py` - features, models, evaluation, figures
- `results.json`, `figures/` - generated outputs
- `Wilson_Project_Proposal (1).pdf` - original proposal
