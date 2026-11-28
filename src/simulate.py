"""
Monte Carlo contract simulator: simulates each player's remaining contract
thousands of times to get the full range of outcomes, not just one guess.

ONE SIMULATED SEASON
--------------------
1. Survival: does he stay a rotation player (1,000+ minutes)? A logistic
   model on age, impact, minutes per game, and games played this season
   and last. If he falls out, he's assumed to add nothing for the rest of
   the deal, but his salary is still owed. (Simplification: in reality some
   players bounce back.)
2. Impact: next season = 0.61 x this season + 0.21 x last season
   (+ aging terms), plus a random shock the size of real year-to-year
   surprises. Coefficients are refit from the data on every run.
3. Games missed: a draw from the negative binomial availability model,
   using his (simulated) attendance history.
4. Minutes per game drift with age, from historical medians.
5. Wins -> dollars at the market price per win, minus that season's salary.

All money is computed as a share of each future season's salary cap
(projected caps from the data), then converted to dollars.

Base season: the last complete season (2025-26). Contracts: the salary
list for 2026-27 onward. Player and team options are treated as
guaranteed (simplification).

Usage, from the project root:
    python src/simulate.py                    # rank every current contract
    python src/simulate.py "Nikola Jokić"     # one player in detail
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from availability_model import build_dataset as build_availability_data, fit_window
from build_dataset import normalize_name
from value_model import MIN_SALARY_PCT, WINS_PER_POINT, replacement_level

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

N_SIMS = 2000
ROTATION_MINUTES = 1000
SEASON_GAMES = 82
PRICE_WINDOW = 5      # average the last 5 seasons' price per win
SEED = 7
SURVIVAL_FEATURES = ["age_c", "age_c2", "age_over_30", "team_impact", "mpg",
                     "availability", "avail_prev"]


# ---------- fitting the pieces from history ----------

def load_values() -> pd.DataFrame:
    v = pd.read_csv(PROCESSED / "player_values.csv")
    v = v.sort_values(["player_id", "season"])
    v["mpg"] = v["minutes"] / v["games"].replace(0, np.nan)
    v["availability"] = pd.read_csv(PROCESSED / "player_seasons.csv").set_index(
        ["player_id", "season"])["availability"].reindex(
        pd.MultiIndex.from_frame(v[["player_id", "season"]])).to_numpy()
    by_player = v.groupby("player_id")
    gap = by_player["season"].shift(1) != v["season"] - 1
    v["impact_prev"] = by_player["team_impact"].shift(1).mask(gap)
    v["avail_prev"] = by_player["availability"].shift(1).mask(gap)
    return v


def impact_features(impact, impact_prev, age):
    has_prev = ~np.isnan(impact_prev)
    return np.column_stack([
        np.ones_like(impact), impact, np.where(has_prev, impact_prev, 0.0),
        has_prev.astype(float), np.clip(24 - age, 0, None), np.clip(age - 30, 0, None)])


def fit_impact_transition(v: pd.DataFrame):
    """Next season's impact from this season, last season, and age."""
    nxt = v[["player_id", "season", "team_impact", "minutes"]].rename(
        columns={"team_impact": "impact_next", "minutes": "min_next"})
    nxt["season"] -= 1
    d = v[v["season_complete"] & (v["minutes"] >= ROTATION_MINUTES)].merge(
        nxt, on=["player_id", "season"])
    d = d[d["min_next"] >= ROTATION_MINUTES]
    x = impact_features(d["team_impact"].to_numpy(), d["impact_prev"].to_numpy(),
                        d["age"].to_numpy())
    w = np.sqrt(d["min_next"].to_numpy())
    coefs, *_ = np.linalg.lstsq(x * w[:, None], d["impact_next"].to_numpy() * w, rcond=None)
    resid = d["impact_next"].to_numpy() - x @ coefs
    shock_sd = float(np.sqrt(np.average(resid ** 2, weights=d["min_next"])))
    return coefs, shock_sd


def fit_minutes_aging(v: pd.DataFrame):
    """Median year-over-year ratio of minutes per game, by age band."""
    nxt = v[["player_id", "season", "mpg", "minutes"]].rename(
        columns={"mpg": "mpg_next", "minutes": "min_next"})
    nxt["season"] -= 1
    d = v[v["minutes"] >= ROTATION_MINUTES].merge(nxt, on=["player_id", "season"])
    d = d[d["min_next"] >= ROTATION_MINUTES]
    bands = [17, 22, 25, 28, 31, 34, 45]
    ratio = d.groupby(pd.cut(d["age"], bands), observed=True).apply(
        lambda g: (g["mpg_next"] / g["mpg"]).median(), include_groups=False)
    return bands, ratio.to_numpy()


def survival_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age_c"] = df["age"] - 27
    out["age_c2"] = out["age_c"] ** 2
    out["age_over_30"] = np.clip(df["age"] - 30, 0, None)
    out["team_impact"] = df["team_impact"]
    out["mpg"] = df["mpg"]
    out["availability"] = df["availability"]
    out["avail_prev"] = df["avail_prev"].fillna(df["availability"])
    return out


def fit_survival(v: pd.DataFrame) -> LogisticRegression:
    nxt = v[["player_id", "season", "minutes"]].rename(columns={"minutes": "min_next"})
    nxt["season"] -= 1
    d = v[v["season_complete"] & (v["minutes"] >= ROTATION_MINUTES)].merge(
        nxt, on=["player_id", "season"], how="left")
    d = d[d["season"] < v.loc[v["season_complete"], "season"].max()]
    d = d.dropna(subset=["team_impact", "mpg", "availability"])
    stay = (d["min_next"] >= ROTATION_MINUTES).astype(int)
    return LogisticRegression(C=1e6, max_iter=3000).fit(survival_frame(d)[SURVIVAL_FEATURES], stay)


# ---------- current contracts ----------

def load_contracts(base_season: int) -> pd.DataFrame:
    """Remaining salary by season for every player on the current salary list."""
    c = pd.read_csv(RAW / "contracts_current.csv")
    season_cols = [col for col in c.columns if col[:4].isdigit() and "-" in col]
    future = {col: int(col[:4]) + 1 for col in season_cols if int(col[:4]) + 1 > base_season}
    long = c.melt(id_vars=["Player"], value_vars=list(future), var_name="label", value_name="salary")
    long["season"] = long["label"].map(future)
    long = long[long["salary"] > 0]
    long["name_key"] = long["Player"].map(normalize_name)
    return long[["name_key", "season", "salary"]]


# ---------- the simulator ----------

class Simulator:
    def __init__(self):
        self.v = load_values()
        self.base_season = int(self.v.loc[self.v["season_complete"], "season"].max())
        self.impact_coefs, self.shock_sd = fit_impact_transition(self.v)
        self.mpg_bands, self.mpg_ratio = fit_minutes_aging(self.v)
        self.survival = fit_survival(self.v)
        avail_data = build_availability_data()
        self.avail_model, self.avail_r, self.avail_typical, _ = fit_window(
            avail_data, self.base_season)
        self.replacement = replacement_level(self.v)
        prices = self.v.groupby("season")["price_per_win"].first()
        self.price = float(prices.loc[self.base_season - PRICE_WINDOW + 1:self.base_season].mean())
        pace = pd.read_csv(RAW / "team_summaries.csv").query(
            "team == 'League Average' and lg == 'NBA'").set_index("season")["pace"]
        self.pace = float(pace.loc[self.base_season])
        cap = pd.read_csv(RAW / "cap.csv").set_index("year")["Salary Cap"]
        self.cap = cap
        self.contracts = load_contracts(self.base_season)

    def base_state(self, player_id: str) -> pd.Series:
        rows = self.v[(self.v["player_id"] == player_id) & (self.v["season"] <= self.base_season)]
        last = rows.iloc[-1]
        hist = rows.set_index("season")["availability"]

        def missed_in(season):
            a = hist.get(season, np.nan)
            return self.avail_typical if pd.isna(a) else round((1 - a) * SEASON_GAMES)

        state = last.copy()
        state["missed"] = [missed_in(self.base_season - k) for k in (0, 1, 2)]
        return state

    def mpg_factor(self, age):
        idx = np.clip(np.searchsorted(self.mpg_bands, age, side="left") - 1,
                      0, len(self.mpg_ratio) - 1)
        return self.mpg_ratio[idx]

    def simulate(self, player_id: str, n: int = N_SIMS, seed: int = SEED) -> pd.DataFrame:
        """Returns one row per simulation per contract season."""
        state = self.base_state(player_id)
        key = normalize_name(state["player"])
        deal = self.contracts[self.contracts["name_key"] == key].sort_values("season")
        if deal.empty:
            raise ValueError(f"No current contract found for {state['player']}")
        rng = np.random.default_rng(seed)

        impact = np.full(n, state["team_impact"], dtype=float)
        impact_prev = np.full(n, state["impact_prev"], dtype=float)
        mpg = np.full(n, state["mpg"], dtype=float)
        avail = np.full(n, state["availability"], dtype=float)
        avail_prev = np.full(n, state["avail_prev"] if pd.notna(state["avail_prev"])
                             else state["availability"], dtype=float)
        missed = np.tile(np.array(state["missed"], dtype=float), (n, 1))
        active = np.ones(n, dtype=bool)
        age = float(state["age"])

        rows = []
        for season, salary in zip(deal["season"], deal["salary"]):
            age += 1
            # 1. Survival, from last season's state (age at that time).
            frame = pd.DataFrame({
                "age": age - 1, "team_impact": impact, "mpg": mpg,
                "availability": avail, "avail_prev": avail_prev})
            p_stay = self.survival.predict_proba(survival_frame(frame)[SURVIVAL_FEATURES])[:, 1]
            active &= rng.random(n) < p_stay

            # 2. Impact.
            x = impact_features(impact, impact_prev, np.full(n, age - 1))
            new_impact = x @ self.impact_coefs + rng.normal(0, self.shock_sd, n)

            # 3. Games missed (negative binomial).
            feats = pd.DataFrame({"missed_0": missed[:, 0], "missed_1": missed[:, 1],
                                  "missed_2": missed[:, 2], "age_c": np.full(n, age - 1 - 27)})
            mean_missed = self.avail_model.predict(feats)
            r = self.avail_r
            games_missed = np.minimum(rng.negative_binomial(r, r / (r + mean_missed)), SEASON_GAMES)
            games = SEASON_GAMES - games_missed

            # 4. Minutes, 5. wins and money.
            new_mpg = mpg * self.mpg_factor(age - 1)
            minutes = games * new_mpg
            wins = (new_impact - self.replacement) * minutes / 48 * self.pace / 100 * WINS_PER_POINT
            wins = np.where(active, wins, 0.0)
            cap = float(self.cap.loc[season])
            value_pct = MIN_SALARY_PCT + np.clip(wins, 0, None) * self.price
            salary_pct = salary / cap
            rows.append(pd.DataFrame({
                "sim": np.arange(n), "season": season, "age": age, "active": active,
                "games": np.where(active, games, 0), "wins": wins,
                "value_m": value_pct * cap / 1e6, "salary_m": salary / 1e6,
                "surplus_m": (value_pct - salary_pct) * cap / 1e6}))

            impact_prev, impact = impact, new_impact
            mpg = new_mpg
            avail_prev, avail = avail, games / SEASON_GAMES
            missed = np.column_stack([games_missed, missed[:, 0], missed[:, 1]])
        return pd.concat(rows, ignore_index=True)

    def summarize(self, sims: pd.DataFrame) -> dict:
        total = sims.groupby("sim")["surplus_m"].sum()
        return {
            "seasons": sims["season"].nunique(),
            "salary_m": sims.groupby("sim")["salary_m"].sum().iloc[0],
            "expected_value_m": sims.groupby("sim")["value_m"].sum().mean(),
            "expected_surplus_m": total.mean(),
            "prob_good_deal": (total > 0).mean(),
            "worst_5pct_m": total.quantile(0.05),
            "best_5pct_m": total.quantile(0.95),
        }

    def player_id(self, name: str) -> str:
        key = normalize_name(name)
        match = self.v[self.v["player"].map(normalize_name) == key]
        if match.empty:
            raise ValueError(f"Player not found: {name}")
        return match["player_id"].iloc[-1]


def print_player(sim: Simulator, name: str):
    pid = sim.player_id(name)
    sims = sim.simulate(pid)
    s = sim.summarize(sims)
    state = sim.base_state(pid)
    print(f"\n{state['player']}, age {int(state['age'])} in {sim.base_season - 1}-"
          f"{str(sim.base_season)[2:]}: impact {state['team_impact']:+.1f}, "
          f"{int(state['games'])} games")
    print(f"Remaining contract: {s['seasons']} seasons, ${s['salary_m']:.0f}M")
    print(f"Expected on-court value: ${s['expected_value_m']:.0f}M")
    print(f"Expected surplus: ${s['expected_surplus_m']:+.0f}M   "
          f"chance it's a good deal: {s['prob_good_deal']:.0%}")
    print(f"Range of outcomes: worst 5% ${s['worst_5pct_m']:+.0f}M, "
          f"best 5% ${s['best_5pct_m']:+.0f}M")
    by_season = sims.groupby("season").agg(
        age=("age", "first"), still_rotation=("active", "mean"),
        avg_games=("games", "mean"), avg_wins=("wins", "mean"),
        salary_m=("salary_m", "first"), avg_surplus_m=("surplus_m", "mean"))
    print(by_season.round(2).to_string())


def rank_all(sim: Simulator):
    current = sim.v[(sim.v["season"] == sim.base_season) & (sim.v["minutes"] >= ROTATION_MINUTES)]
    keys = set(sim.contracts["name_key"])
    current = current[current["player"].map(normalize_name).isin(keys)]
    results = []
    for pid, name in zip(current["player_id"], current["player"]):
        s = sim.summarize(sim.simulate(pid, n=500))
        results.append({"player": name, **s})
    out = pd.DataFrame(results).sort_values("expected_surplus_m", ascending=False)
    out.to_csv(PROCESSED / "contract_projections.csv", index=False)
    print(f"Projected {len(out)} current contracts -> {PROCESSED / 'contract_projections.csv'}")
    cols = ["player", "seasons", "salary_m", "expected_surplus_m", "prob_good_deal", "worst_5pct_m"]
    fmt = {"salary_m": "{:.0f}".format, "expected_surplus_m": "{:+.0f}".format,
           "prob_good_deal": "{:.0%}".format, "worst_5pct_m": "{:+.0f}".format}
    print("\nBest contracts (most expected surplus, $M):")
    print(out.head(10)[cols].to_string(index=False, formatters=fmt))
    print("\nWorst contracts:")
    print(out.tail(10)[cols].to_string(index=False, formatters=fmt))


def main():
    sim = Simulator()
    print(f"Base season {sim.base_season - 1}-{str(sim.base_season)[2:]} | "
          f"impact shock SD {sim.shock_sd:.2f} | price per win {sim.price:.1%} of cap | "
          f"replacement {sim.replacement:+.2f}")
    if len(sys.argv) > 1:
        print_player(sim, " ".join(sys.argv[1:]))
    else:
        rank_all(sim)


if __name__ == "__main__":
    main()
