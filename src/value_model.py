"""
Value model: how many wins each player-season added, what those wins were
worth on the open market, and whether the player was underpaid or overpaid.

THE CHAIN
---------
1. Impact per 100 possessions (credibility blend):
       impact = Z * RAPM + (1 - Z) * prior,   Z = minutes / (minutes + K)
   RAPM is adjusted plus-minus from real scoring margins. The prior is BPM
   rescaled to predict next season's RAPM. K = 5,400 minutes was chosen by
   testing which value best predicts next season (fit on 2010-2017, checked
   on 2018-2022). RAPM is only available through 2022-23; later seasons use
   the prior alone.

2. Team calibration. The blend is built for forecasting, so it shrinks
   everyone toward average and understates real contributions. We keep the
   blend for HOW credit is split among teammates, but make each team's
   players add up to the team's actual net rating: rescale by a single
   factor fit across all teams, then split any leftover by minutes played.

3. Wins above replacement:
       wins = (impact - replacement) * possessions / 100 * phi(0) / 12
   phi(0) / 12 comes from P(win) = Phi(margin / 12), the margin-to-wins link
   tested on real games: about 30 points of margin per win. Replacement
   level is the average of players outside the top 390 in minutes each
   season (13 per team), the talent any team can sign for the minimum.

4. Price per win: total salary above the minimum divided by total wins,
   among veterans (5+ seasons) on non-minimum deals, so rookie-scale and
   minimum contracts don't distort the market rate. Computed per season,
   as a share of the salary cap.

5. Fair salary = minimum + wins * price per win (never below the minimum).
   Surplus = fair salary - actual salary. Positive = underpaid.

Max contracts cap star salaries at 25-35% of the cap, so true superstars
will always show as underpaid. That's a feature of the league's rules, and
the output labels max-level contracts so it isn't mistaken for a finding.

Output: data/processed/player_values.csv

Usage, from the project root:
    python src/value_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from build_dataset import normalize_name

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

K_MINUTES = 5400              # credibility half-trust point for RAPM
PRIOR_FIT_LAST_SEASON = 2017  # prior is fit on seasons through 2016-17
FIRST_SEASON = 2010           # first season with RAPM
ROTATION_SLOTS = 390          # 13 roster spots x 30 teams
WINS_PER_POINT = norm.pdf(0) / 12   # wins per point of season margin, per game
MIN_SALARY_PCT = 0.014        # typical minimum salary, share of cap
MARKET_MIN_PCT = 0.025        # below this counts as a minimum-type deal
MAX_LEVEL_PCT = 0.25          # at or above this counts as max-level
ROOKIE_SCALE_YEARS = 4


def load() -> pd.DataFrame:
    p = pd.read_csv(PROCESSED / "player_seasons.csv")
    rapm = pd.read_csv(RAW / "rapm.csv")
    career = pd.read_csv(RAW / "career_info.csv")

    rapm["season"] = rapm["season"].str[:4].astype(int) + 1
    rapm["name_key"] = rapm["playerName"].map(normalize_name)
    p["name_key"] = p["player"].map(normalize_name)
    p = p.merge(rapm[["name_key", "season", "RAPM"]], on=["name_key", "season"], how="left")
    p = p.merge(career[["player_id", "from"]], on="player_id", how="left")
    p["years_in_league"] = p["season"] - p["from"] + 1
    return p.drop(columns="name_key")


def fit_prior(p: pd.DataFrame) -> tuple[float, float]:
    """BPM rescaled to predict next season's RAPM, fit on early seasons only."""
    nxt = p[["player_id", "season", "RAPM", "minutes"]].rename(
        columns={"RAPM": "rapm_next", "minutes": "min_next"})
    nxt["season"] -= 1
    pairs = p.merge(nxt, on=["player_id", "season"]).dropna(subset=["bpm", "rapm_next"])
    pairs = pairs[(pairs["min_next"] >= 500) & (pairs["season"] <= PRIOR_FIT_LAST_SEASON)]
    slope, intercept = np.polyfit(pairs["bpm"], pairs["rapm_next"], 1, w=pairs["min_next"])
    return intercept, slope


def blended_impact(p: pd.DataFrame) -> pd.Series:
    intercept, slope = fit_prior(p)
    prior = intercept + slope * p["bpm"]
    z = p["minutes"] / (p["minutes"] + K_MINUTES)
    return pd.Series(np.where(p["RAPM"].notna(), z * p["RAPM"] + (1 - z) * prior, prior),
                     index=p.index)


def team_calibrated_impact(p: pd.DataFrame) -> pd.Series:
    """Rescales the blend so each team's players add up to its real net rating."""
    adv = pd.read_csv(RAW / "advanced.csv")
    adv = adv[(adv["lg"] == "NBA") & (adv["season"] >= FIRST_SEASON)
              & ~adv["team"].str.fullmatch(r"\dTM|TOT").fillna(False)]
    stints = adv[["player_id", "season", "team", "mp"]].merge(
        p[["player_id", "season", "impact"]], on=["player_id", "season"]).dropna(subset=["impact"])

    teams = pd.read_csv(RAW / "team_summaries.csv")
    teams = teams[(teams["lg"] == "NBA") & (teams["team"] != "League Average")]
    teams = teams[["season", "abbreviation", "n_rtg"]].rename(columns={"abbreviation": "team"})

    def lineup_sum(df, col):
        return df.groupby(["season", "team"]).apply(
            lambda d: 5 * np.average(d[col], weights=d["mp"]), include_groups=False)

    summed = lineup_sum(stints, "impact").rename("summed").reset_index().merge(
        teams, on=["season", "team"])
    scale, offset = np.polyfit(summed["summed"], summed["n_rtg"], 1)
    stints["cal"] = scale * stints["impact"] + offset / 5

    resid = lineup_sum(stints, "cal").rename("summed").reset_index().merge(
        teams, on=["season", "team"])
    resid["adjust"] = (resid["n_rtg"] - resid["summed"]) / 5
    stints = stints.merge(resid[["season", "team", "adjust"]], on=["season", "team"])
    stints["team_impact"] = stints["cal"] + stints["adjust"]

    per_player = stints.groupby(["player_id", "season"]).apply(
        lambda d: np.average(d["team_impact"], weights=d["mp"]), include_groups=False)
    print(f"Team calibration: real net rating = {scale:.2f} x summed blend {offset:+.2f}")
    return p.set_index(["player_id", "season"]).index.map(per_player)


def replacement_level(p: pd.DataFrame) -> float:
    """Minutes-weighted impact of players outside the top 390 in minutes."""
    pool = []
    for _, season in p[p["team_impact"].notna()].groupby("season"):
        rest = season.sort_values("minutes", ascending=False).iloc[ROTATION_SLOTS:]
        pool.append(rest[rest["minutes"] >= 100])
    pool = pd.concat(pool)
    return float(np.average(pool["team_impact"], weights=pool["minutes"]))


def price_per_win(p: pd.DataFrame) -> pd.Series:
    """Share of the cap paid per win, among veterans on non-minimum deals.

    Uses complete seasons only (a season in progress has few wins so far,
    which would make each win look absurdly expensive). Seasons without
    salary data (2024-25) are filled from their neighbors, and the season
    in progress uses the latest complete season's price.
    """
    market = p[(p["years_in_league"] > ROOKIE_SCALE_YEARS)
               & (p["salary_pct_cap"] >= MARKET_MIN_PCT) & p["wins"].notna()
               & p["season_complete"]]
    price = market.groupby("season").apply(
        lambda d: (d["salary_pct_cap"] - MIN_SALARY_PCT).sum() / d["wins"].sum(),
        include_groups=False)
    all_seasons = range(int(p["season"].min()), int(p["season"].max()) + 1)
    return price.reindex(all_seasons).interpolate().ffill()


def main():
    p = load()
    p = p[p["season"] >= FIRST_SEASON].copy()
    p["impact"] = blended_impact(p)
    p["team_impact"] = team_calibrated_impact(p)

    pace = pd.read_csv(RAW / "team_summaries.csv").query(
        "team == 'League Average' and lg == 'NBA'").set_index("season")["pace"]
    p["possessions"] = p["minutes"] / 48 * p["season"].map(pace)

    replacement = replacement_level(p)
    p["wins"] = (p["team_impact"] - replacement) * p["possessions"] / 100 * WINS_PER_POINT
    p.loc[p["games"] == 0, "wins"] = 0.0
    print(f"Replacement level: {replacement:+.2f} per 100 possessions "
          f"(a team of these wins about {82 * norm.cdf(5 * replacement / 12):.0f} games)")

    price = price_per_win(p)
    p["price_per_win"] = p["season"].map(price)
    p["fair_pct_cap"] = MIN_SALARY_PCT + p["wins"].clip(lower=0) * p["price_per_win"]
    p["surplus_pct_cap"] = p["fair_pct_cap"] - p["salary_pct_cap"]
    p["surplus_millions"] = p["surplus_pct_cap"] * p["cap"] / 1e6
    p["contract_type"] = np.select(
        [p["years_in_league"] <= ROOKIE_SCALE_YEARS, p["salary_pct_cap"] >= MAX_LEVEL_PCT],
        ["rookie-scale", "max-level"], "market")

    cols = ["player_id", "player", "season", "age", "team", "games", "minutes",
            "impact", "team_impact", "wins", "salary", "salary_pct_cap", "cap",
            "price_per_win", "fair_pct_cap", "surplus_pct_cap", "surplus_millions",
            "contract_type", "season_complete"]
    p[cols].to_csv(PROCESSED / "player_values.csv", index=False)
    print(f"Wrote {len(p):,} player-seasons to {PROCESSED / 'player_values.csv'}")

    print("\nPrice per win, share of cap:")
    print(price.map("{:.1%}".format).to_string())

    graded = p[p["salary"].notna() & p["wins"].notna() & p["season_complete"]]
    show = ["season", "player", "games", "wins", "salary_pct_cap", "fair_pct_cap",
            "surplus_millions", "contract_type"]
    fmt = {"wins": "{:.1f}".format, "salary_pct_cap": "{:.0%}".format,
           "fair_pct_cap": "{:.0%}".format, "surplus_millions": "{:+.1f}".format}
    print("\nMost underpaid seasons:")
    print(graded.nlargest(10, "surplus_pct_cap")[show].to_string(index=False, formatters=fmt))
    print("\nMost overpaid seasons:")
    print(graded.nsmallest(10, "surplus_pct_cap")[show].to_string(index=False, formatters=fmt))
    print("\nShare of seasons underpaid, by contract type:")
    print(graded.groupby("contract_type")["surplus_pct_cap"].apply(
        lambda s: f"{(s > 0).mean():.0%}").to_string())


if __name__ == "__main__":
    main()
