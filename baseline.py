"""NFL home-win predictor (NFLverse games.csv).

Features use ONLY information available before kickoff (rolling stats shifted by
one game, Elo computed before each game) so there is no data leakage.
Split is chronological, never random:
    train 1999-2020 | validation 2021-2022 | test 2023+
Models are compared on validation; for the final test numbers they are refit
on train+validation.

Run:  python baseline.py      (writes results.json and figures/*.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOW = 5          # "recent performance" = last N games
ELO_K, ELO_HFA, ELO_REVERT = 20, 48, 1 / 3
os.makedirs("figures", exist_ok=True)

g = pd.read_csv("data/games.csv", parse_dates=["gameday"])
g = g.dropna(subset=["home_score", "away_score"]).sort_values("gameday").reset_index(drop=True)
g["home_win"] = (g.home_score > g.away_score).astype(int)  # ties count as 0 (rare)

# ---- rolling team features (long format: one row per team per game) -------
home = g[["game_id", "season", "gameday", "home_team", "home_score", "away_score"]].copy()
home.columns = ["game_id", "season", "gameday", "team", "pf", "pa"]
away = g[["game_id", "season", "gameday", "away_team", "away_score", "home_score"]].copy()
away.columns = ["game_id", "season", "gameday", "team", "pf", "pa"]
t = pd.concat([home, away]).sort_values(["team", "gameday"]).reset_index(drop=True)
t["win"] = (t.pf > t.pa).astype(float)

grp = t.groupby("team")
t["win_pct_roll"] = grp["win"].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
t["pf_roll"] = grp["pf"].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
t["pa_roll"] = grp["pa"].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
t["win_pct_season"] = t.groupby(["team", "season"])["win"].transform(
    lambda s: s.shift(1).expanding().mean())
feat = ["win_pct_roll", "pf_roll", "pa_roll", "win_pct_season"]

h = t.rename(columns={c: f"home_{c}" for c in feat})
a = t.rename(columns={c: f"away_{c}" for c in feat})
df = g.merge(h[["game_id", "team"] + [f"home_{c}" for c in feat]],
             left_on=["game_id", "home_team"], right_on=["game_id", "team"]).drop(columns="team")
df = df.merge(a[["game_id", "team"] + [f"away_{c}" for c in feat]],
              left_on=["game_id", "away_team"], right_on=["game_id", "team"]).drop(columns="team")
df = df.sort_values("gameday").reset_index(drop=True)

# ---- Elo (pre-game rating difference) -------------------------------------
elo, last_season, elo_diff = {}, None, []
for r in df.itertuples():
    if r.season != last_season:                      # regress toward mean each season
        elo = {k: 1500 + (v - 1500) * (1 - ELO_REVERT) for k, v in elo.items()}
        last_season = r.season
    eh, ea = elo.get(r.home_team, 1500), elo.get(r.away_team, 1500)
    hfa = 0 if r.location == "Neutral" else ELO_HFA
    elo_diff.append(eh + hfa - ea)
    exp_h = 1 / (1 + 10 ** (-(eh + hfa - ea) / 400))
    margin = abs(r.home_score - r.away_score)
    mult = np.log(margin + 1) * 2.2 / ((abs(eh + hfa - ea)) * 0.001 + 2.2)  # margin-of-victory
    delta = ELO_K * mult * ((1 if r.home_win else 0) - exp_h)
    elo[r.home_team], elo[r.away_team] = eh + delta, ea - delta
df["elo_diff"] = elo_diff

df["is_neutral"] = (df.location == "Neutral").astype(int)
df["rest_diff"] = df.home_rest - df.away_rest
for c in feat:
    df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]
X_cols = [f"diff_{c}" for c in feat] + ["elo_diff", "rest_diff", "is_neutral"]
df = df.dropna(subset=X_cols)

# ---- chronological split --------------------------------------------------
train = df[df.season <= 2020]
val = df[df.season.between(2021, 2022)]
trainval = df[df.season <= 2022]
test = df[df.season >= 2023]
print(f"train {len(train)}  val {len(val)}  test {len(test)} games")

def make_models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=0),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                        learning_rate=0.05, random_state=0),
    }

def metrics(y, p):
    pred = p > 0.5
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "auc": roc_auc_score(y, p)}

results = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
           "validation": {}, "test": {}}
results["validation"]["Always home"] = {"accuracy": val.home_win.mean()}
results["test"]["Always home"] = {"accuracy": test.home_win.mean()}

# validation: fit on train only
for name, m in make_models().items():
    m.fit(train[X_cols], train.home_win)
    results["validation"][name] = metrics(val.home_win, m.predict_proba(val[X_cols])[:, 1])

# test: refit on train+validation
fitted, test_probs = {}, {}
for name, m in make_models().items():
    m.fit(trainval[X_cols], trainval.home_win)
    fitted[name] = m
    test_probs[name] = m.predict_proba(test[X_cols])[:, 1]
    results["test"][name] = metrics(test.home_win, test_probs[name])

# ---- Vegas benchmark (spread_line > 0 => home favored) ---------------------
def vegas(split):
    s = split.dropna(subset=["spread_line"])
    return s[s.spread_line != 0]
for label, split in [("validation", val), ("test", test)]:
    s = vegas(split)
    results[label]["Vegas spread"] = {"accuracy": accuracy_score(s.home_win, s.spread_line > 0),
                                      "n": len(s)}

# ---- significance: each model vs Vegas on the same test games --------------
sv = vegas(test)
vegas_correct = ((sv.spread_line > 0).astype(int) == sv.home_win).values
results["mcnemar_vs_vegas"] = {}
for name in fitted:
    p = pd.Series(test_probs[name], index=test.index).loc[sv.index].values
    mod_correct = ((p > 0.5).astype(int) == sv.home_win.values)
    b = int(((mod_correct) & (~vegas_correct)).sum())   # model right, Vegas wrong
    c = int(((~mod_correct) & (vegas_correct)).sum())   # Vegas right, model wrong
    pval = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0  # exact McNemar
    results["mcnemar_vs_vegas"][name] = {"model_only_right": b, "vegas_only_right": c, "p_value": pval}

# ---- figures -----------------------------------------------------------------
best = max(fitted, key=lambda n: results["validation"][n]["accuracy"])  # chosen on validation, not test
results["best_model"] = best

# 1. accuracy comparison
names = ["Always home"] + list(fitted) + ["Vegas spread"]
accs = [results["test"][n]["accuracy"] for n in names]
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(names, accs, color=["#9aa0a6"] + ["#4c78a8"] * len(fitted) + ["#e45756"])
ax.set_ylim(0.45, 0.75); ax.set_ylabel("Test accuracy (2023+)")
for b_, v in zip(bars, accs):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig("figures/accuracy_comparison.png", dpi=200); plt.close()

# 2. feature importance (random forest)
rf = fitted["Random forest"]
imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.8))
imp.plot.barh(ax=ax, color="#4c78a8"); ax.set_xlabel("Random forest importance")
plt.tight_layout(); plt.savefig("figures/feature_importance.png", dpi=200); plt.close()

# 3. calibration of best model
frac, mean_p = calibration_curve(test.home_win, test_probs[best], n_bins=8)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], "--", color="#9aa0a6"); ax.plot(mean_p, frac, "o-", color="#4c78a8")
ax.set_xlabel("Predicted P(home win)"); ax.set_ylabel("Observed frequency"); ax.set_title(best)
plt.tight_layout(); plt.savefig("figures/calibration.png", dpi=200); plt.close()

# 4. confusion matrix of best model
cm = confusion_matrix(test.home_win, test_probs[best] > 0.5)
results["confusion_matrix_best"] = cm.tolist()
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, v, ha="center", va="center")
ax.set_xticks([0, 1], ["Away win", "Home win"]); ax.set_yticks([0, 1], ["Away win", "Home win"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(best)
plt.tight_layout(); plt.savefig("figures/confusion_matrix.png", dpi=200); plt.close()

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
