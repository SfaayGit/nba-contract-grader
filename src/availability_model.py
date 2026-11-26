"""
Availability (severity) model: how many games a rotation player will miss
next season, given he stays in the rotation.

Together with survival_model.py (will he stay in the rotation at all?),
this gives expected games played for any player.

THE MODEL
---------
Games missed is a count, and misses clump (one injury costs many games in
a row), so its variance is far larger than its mean. A Poisson model can't
capture that; a negative binomial can. It's the standard model insurers use
for claim counts, for the same reason: some policyholders are much riskier
than others.

    games missed ~ NegativeBinomial(mean = m, dispersion = r)
    log(m) = b0 + b1*missed_this_season + b2*missed_last_season
             + b3*missed_two_seasons_ago + b4*age

A player's own attendance history pulls his expected misses away from the
league average, and the fitted coefficients decide how much each past
season counts: track record vs. the average player, the same credibility
idea as the impact model.

The mean is fit with Poisson regression (which gets the mean right even
when the data is overdispersed), then the dispersion r is fit by maximum
likelihood given those means.

WALK-FORWARD REFITTING
----------------------
Missed games nearly doubled from the 2000s (~9 per rotation player) to the
2020s (~16) as load management spread. A model fit on old seasons badly
misjudges new ones. So, like an insurer updating rates with recent
experience, the model is refit every season on the 5 seasons before it.
It is evaluated the same way: each test season is predicted only from
seasons that came before it.

Population: rotation players (1,000+ minutes) who are still in the rotation
the next season. Players who drop out are handled by survival_model.py.
Games are scaled to an 82-game season so shortened seasons compare fairly.

Usage, from the project root:
    python src/availability_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ROTATION_MINUTES = 1000
SEASON_GAMES = 82
WINDOW_SEASONS = 5
FIRST_TEST_SEASON = 2017       # predicts 2017-18 onward
LAST_COMPLETE_SEASON = 2026    # 2026-27 is still being played
FEATURES = ["missed_0", "missed_1", "missed_2", "age_c"]
BINS = [0, 3, 6, 11, 21, 41, 83]
BIN_LABELS = ["0-2", "3-5", "6-10", "11-20", "21-40", "41-82"]


def missed(availability: pd.Series) -> pd.Series:
    """Games missed on an 82-game basis."""
    return ((1 - availability) * SEASON_GAMES).round()


def build_dataset() -> pd.DataFrame:
    """One row per rotation player-season, with next season's games missed
    as the target. 'season' is the season the prediction is made from."""
    p = pd.read_csv(PROCESSED / "player_seasons.csv")
    p = p[p["season"] <= LAST_COMPLETE_SEASON].sort_values(["player_id", "season"])
    p["missed_0"] = missed(p["availability"])

    # Attendance in the two seasons before. A gap (e.g. a season overseas)
    # means no record for that year.
    by_player = p.groupby("player_id")
    for lag in (1, 2):
        prev = by_player["missed_0"].shift(lag)
        gap = by_player["season"].shift(lag) != p["season"] - lag
        p[f"missed_{lag}"] = prev.mask(gap)

    cur = p[(p["minutes"] >= ROTATION_MINUTES)
            & (p["season"].between(2002, LAST_COMPLETE_SEASON))].copy()
    nxt = p[["player_id", "season", "minutes", "missed_0"]].rename(
        columns={"minutes": "min_next", "missed_0": "target"})
    nxt["season"] -= 1
    cur = cur.merge(nxt, on=["player_id", "season"], how="left")
    # Rows for the newest season have no "next season" yet: kept for prediction.
    has_next = cur["min_next"].notna()
    cur = cur[~has_next | (cur["min_next"] >= ROTATION_MINUTES)]
    cur["age_c"] = cur["age"] - 27
    return cur


def fill_history(df: pd.DataFrame, typical: float) -> pd.DataFrame:
    """No history (rookies, gaps): use a typical miss count from the fit window."""
    df = df.copy()
    for col in ("missed_1", "missed_2"):
        df[col] = df[col].fillna(typical)
    return df


def fit_dispersion(y: np.ndarray, mean: np.ndarray) -> float:
    """Maximum-likelihood negative binomial dispersion r, given fitted means."""
    def neg_loglik(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(y, r, r / (r + mean)).sum()
    return float(np.exp(minimize_scalar(neg_loglik, bounds=(-5, 5), method="bounded").x))


def fit_window(data: pd.DataFrame, predict_from: int):
    """Fits on the WINDOW_SEASONS seasons before `predict_from`.
    Returns (model, r, typical) for predicting from that season."""
    train = data[(data["season"] < predict_from)
                 & (data["season"] >= predict_from - WINDOW_SEASONS)
                 & data["target"].notna()]
    typical = train["missed_0"].median()
    train = fill_history(train, typical)
    model = PoissonRegressor(alpha=0, max_iter=1000).fit(train[FEATURES], train["target"])
    r = fit_dispersion(train["target"].to_numpy(), model.predict(train[FEATURES]))
    return model, r, typical, train["target"].mean()


def bin_probabilities(mean: np.ndarray, r: float) -> np.ndarray:
    """Average predicted share of players in each games-missed bin.
    Probability beyond 82 is folded into the last bin."""
    ks = np.arange(SEASON_GAMES + 1)
    pmf = stats.nbinom.pmf(ks[None, :], r, (r / (r + mean))[:, None])
    pmf[:, -1] += 1 - pmf.sum(axis=1)
    edges = np.array(BINS)
    return np.array([pmf[:, lo:hi].sum(axis=1).mean()
                     for lo, hi in zip(edges[:-1], np.minimum(edges[1:], SEASON_GAMES + 1))])


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """Predicts each test season using only seasons before it."""
    results = []
    for season in range(FIRST_TEST_SEASON, LAST_COMPLETE_SEASON):
        model, r, typical, recent_avg = fit_window(data, season)
        test = fill_history(data[(data["season"] == season) & data["target"].notna()], typical)
        results.append(test.assign(pred=model.predict(test[FEATURES]), r=r, recent_avg=recent_avg))
    return pd.concat(results)


def main():
    data = build_dataset()
    res = walk_forward(data)
    y = res["target"].to_numpy()

    print(f"Walk-forward test: {len(res):,} player-seasons, predicting "
          f"{FIRST_TEST_SEASON + 1}-{LAST_COMPLETE_SEASON}, refit each season "
          f"on the previous {WINDOW_SEASONS}")
    print("\nAverage error in predicted games missed:")
    print(f"  Recent league average:  {np.abs(y - res['recent_avg']).mean():.2f} games")
    print(f"  Same as this season:    {np.abs(y - res['missed_0']).mean():.2f} games")
    print(f"  Model:                  {np.abs(y - res['pred']).mean():.2f} games")

    ll = stats.nbinom.logpmf(y, res["r"], res["r"] / (res["r"] + res["pred"])).mean()
    ll_pois = stats.poisson.logpmf(y, res["pred"]).mean()
    print("\nAverage log-likelihood per player (higher is better):")
    print(f"  Poisson:           {ll_pois:.3f}")
    print(f"  Negative binomial: {ll:.3f}")

    predicted = sum(bin_probabilities(g["pred"].to_numpy(), r) * len(g)
                    for r, g in res.groupby("r")) / len(res)
    actual = pd.cut(res["target"], BINS, right=False, labels=BIN_LABELS).value_counts(
        normalize=True).reindex(BIN_LABELS)
    print("\nShare of players by games missed:")
    print(pd.DataFrame({"actual": actual.to_numpy(), "model": predicted},
                       index=BIN_LABELS).round(3).to_string())

    # The model actually used for next season: fit on the most recent window.
    model, r, _, _ = fit_window(data, LAST_COMPLETE_SEASON)
    print(f"\nCurrent model (fit on {LAST_COMPLETE_SEASON - WINDOW_SEASONS}-"
          f"{LAST_COMPLETE_SEASON - 1}), dispersion r = {r:.2f}")
    print(f"  intercept   {model.intercept_:+.4f}")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  {name:10s}  {coef:+.4f}")


if __name__ == "__main__":
    main()
