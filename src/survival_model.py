"""
Career survival model: the probability a rotation player is still a
rotation player next season.

This is built like an insurer's mortality table, but for NBA careers.
"Rotation player" means 1,000+ minutes in a season. For every rotation
player-season, we record what happened the following year, then fit a
logistic regression to predict it.

Tested and kept (each improved predictions on seasons the model never saw):
  - age, with an extra decline term after 30
  - quality (BPM)
  - role (minutes per game)
  - availability this season and last season

Tested and dropped (no meaningful out-of-sample gain):
  - height and build (BMI), and how they interact with age

Fit on 2002-2017 seasons, evaluated on 2018-2025.

Usage, from the project root:
    python src/survival_model.py
Writes data/processed/survival_table.csv (raw outcomes by age).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ROTATION_MINUTES = 1000
TRAIN_LAST_SEASON = 2017
LAST_COMPLETE_SEASON = 2026  # 2026-27 is still being played
FEATURES = ["age_c", "age_c2", "age_over_30", "bpm", "mpg", "availability", "avail_prev"]


def build_transitions() -> pd.DataFrame:
    p = pd.read_csv(PROCESSED / "player_seasons.csv")
    p = p[p["season"] <= LAST_COMPLETE_SEASON].sort_values(["player_id", "season"])
    p["avail_prev"] = p.groupby("player_id")["availability"].shift(1)

    cur = p[(p["minutes"] >= ROTATION_MINUTES)
            & (p["season"].between(2002, LAST_COMPLETE_SEASON - 1))].copy()

    nxt = p[["player_id", "season", "minutes", "games"]].rename(
        columns={"minutes": "min_next", "games": "games_next"})
    nxt["season"] -= 1
    cur = cur.merge(nxt, on=["player_id", "season"], how="left")

    def outcome(row):
        if pd.isna(row["min_next"]):
            return "out_of_league"
        if row["games_next"] == 0:
            return "missed_season"
        if row["min_next"] >= ROTATION_MINUTES:
            return "rotation"
        return "bench"

    cur["next_state"] = cur.apply(outcome, axis=1)
    cur["stay"] = (cur["next_state"] == "rotation").astype(int)

    # A first-year player has no previous season; use this season instead.
    cur["avail_prev"] = cur["avail_prev"].fillna(cur["availability"])
    cur["age_c"] = cur["age"] - 27
    cur["age_c2"] = cur["age_c"] ** 2
    cur["age_over_30"] = np.clip(cur["age"] - 30, 0, None)
    cur["mpg"] = cur["minutes"] / cur["games"]
    return cur.dropna(subset=["bpm"])


def survival_table(cur: pd.DataFrame) -> pd.DataFrame:
    """Raw outcomes by age, like a mortality table."""
    table = pd.crosstab(cur["age"].astype(int), cur["next_state"], normalize="index")
    table["players"] = cur.groupby(cur["age"].astype(int)).size()
    cols = ["players", "rotation", "bench", "missed_season", "out_of_league"]
    return table[table["players"] >= 30][cols]


def main():
    cur = build_transitions()
    table = survival_table(cur)
    table.to_csv(PROCESSED / "survival_table.csv")
    print("Next-season outcome for rotation players, by age:")
    print(table.round(3).to_string())

    train = cur[cur["season"] <= TRAIN_LAST_SEASON]
    test = cur[cur["season"] > TRAIN_LAST_SEASON]
    model = LogisticRegression(C=1e6, max_iter=2000).fit(train[FEATURES], train["stay"])
    pred = model.predict_proba(test[FEATURES])[:, 1]

    base = np.full(len(test), train["stay"].mean())
    print(f"\nTest seasons {TRAIN_LAST_SEASON + 1}-{LAST_COMPLETE_SEASON - 1}: "
          f"{len(test):,} player-seasons")
    print(f"Log loss, same guess for everyone: {log_loss(test['stay'], base):.4f}")
    print(f"Log loss, survival model:          {log_loss(test['stay'], pred):.4f}")

    # Calibration: when the model says 70%, does it happen about 70% of the time?
    check = test.assign(pred=pred)
    check["bucket"] = pd.qcut(check["pred"], 5)
    calib = check.groupby("bucket", observed=True).agg(
        predicted=("pred", "mean"), actual=("stay", "mean"), players=("stay", "size"))
    print("\nCalibration (predicted vs actual rate of staying in the rotation):")
    print(calib.round(3).to_string())

    print("\nCoefficients:")
    for name, coef in zip(FEATURES, model.coef_[0]):
        print(f"  {name:12s} {coef:+.3f}")


if __name__ == "__main__":
    main()
