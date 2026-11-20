"""
Builds one clean table with a row per player per season: production,
availability, and salary.

Output: data/processed/player_seasons.csv

Columns:
  player_id, player, season, age, team, games, team_games, availability,
  minutes, bpm, vorp, ws, salary, cap, salary_pct_cap

Notes on conventions:
  - "season" is the year the season ends (2019 = the 2018-19 season),
    the same convention both sources use.
  - Players traded mid-season appear on several teams; we keep their
    combined season row (team "2TM", "3TM", ...).
  - Salaries are expressed as a share of that season's salary cap, so a
    2015 contract and a 2024 contract can be compared fairly.
  - Salaries and stats come from different sources, so they are joined on
    a normalized player name plus season.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

FIRST_SEASON = 2001
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Salary-file name -> Basketball-Reference name, for players listed under
# a nickname or former name in one source.
ALIASES = {
    "Dennis Schroeder": "Dennis Schröder",
    "Enes Kanter": "Enes Freedom",
    "Jose Juan Barea": "J.J. Barea",
    "Sviatoslav Mykhailiuk": "Svi Mykhailiuk",
    "Cameron Whitmore": "Cam Whitmore",
    "Santiago Aldama": "Santi Aldama",
    "Wesley Iwundu": "Wes Iwundu",
    "Timothe Luwawu": "Timothé Luwawu-Cabarrot",
    "Moe Harkless": "Maurice Harkless",
}


def normalize_name(name: str) -> str:
    """'Luka Dončić' -> 'luka doncic', 'Gary Trent Jr.' -> 'gary trent'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z ]", "", text.lower().replace("-", " "))
    words = [w for w in text.split() if w not in SUFFIXES]
    return " ".join(words)


def load_stats() -> pd.DataFrame:
    adv = pd.read_csv(RAW / "advanced.csv")
    adv = adv[(adv["lg"] == "NBA") & (adv["season"] >= FIRST_SEASON)].copy()

    # Traded players have one row per team plus a combined row ("2TM", "3TM", ...).
    # Keep the combined row when it exists, otherwise the single team row.
    adv["is_combined"] = adv["team"].str.fullmatch(r"\dTM|TOT").fillna(False)
    adv = (adv.sort_values("is_combined", ascending=False)
              .drop_duplicates(["player_id", "season"], keep="first"))

    # The teams each player suited up for that season, for the games denominator.
    per_team = pd.read_csv(RAW / "advanced.csv")
    per_team = per_team[(per_team["lg"] == "NBA") & (per_team["season"] >= FIRST_SEASON)]
    per_team = per_team[~per_team["team"].str.fullmatch(r"\dTM|TOT").fillna(False)]

    teams = pd.read_csv(RAW / "team_summaries.csv")
    teams = teams[teams["lg"] == "NBA"].copy()
    teams["team_games"] = teams["w"] + teams["l"]
    team_games = per_team.merge(
        teams[["season", "abbreviation", "team_games"]],
        left_on=["season", "team"], right_on=["season", "abbreviation"], how="left",
    ).groupby(["player_id", "season"])["team_games"].max()

    adv = adv.merge(team_games.rename("team_games"), on=["player_id", "season"], how="left")
    adv = adv.rename(columns={"g": "games", "mp": "minutes"})
    adv["availability"] = (adv["games"] / adv["team_games"]).clip(upper=1.0)
    adv["name_key"] = adv["player"].map(normalize_name)

    cols = ["player_id", "player", "name_key", "season", "age", "team", "games",
            "team_games", "availability", "minutes", "bpm", "vorp", "ws"]
    return adv[cols]


def load_salaries() -> pd.DataFrame:
    hist = pd.read_csv(RAW / "salaries.csv")
    hist = hist.rename(columns={"year": "season"})[["player", "season", "salary"]]

    # Current contracts file has one column per future season ("2025-26" -> 2026).
    cur = pd.read_csv(RAW / "contracts_current.csv")
    season_cols = [c for c in cur.columns if re.fullmatch(r"\d{4}-\d{2}", c)]
    cur = cur.melt(id_vars=["Player"], value_vars=season_cols,
                   var_name="season_label", value_name="salary")
    cur["season"] = cur["season_label"].str[:4].astype(int) + 1
    cur = cur[cur["salary"] > 0].rename(columns={"Player": "player"})

    sal = pd.concat([hist, cur[["player", "season", "salary"]]], ignore_index=True)
    sal["player"] = sal["player"].replace(ALIASES)
    sal["name_key"] = sal["player"].map(normalize_name)
    # A player can appear twice in a season (traded, or listed in both files).
    # The largest figure is his salary; summing would double count.
    sal = sal.groupby(["name_key", "season"], as_index=False)["salary"].max()

    cap = pd.read_csv(RAW / "cap.csv")[["year", "Salary Cap"]]
    cap = cap.rename(columns={"year": "season", "Salary Cap": "cap"})
    sal = sal.merge(cap, on="season", how="left")
    sal["salary_pct_cap"] = sal["salary"] / sal["cap"]
    return sal


def add_missed_seasons(stats: pd.DataFrame, sal: pd.DataFrame) -> pd.DataFrame:
    """Adds zero-game rows for seasons a player was paid but never played.

    A player who misses a whole season (e.g. Kawhi Leonard in 2021-22) has
    no row in the stats data at all. Without these rows the injury model
    would never see the most severe injuries. We add a row when a player
    was paid that season and played in BOTH the season before and after it.

    Requiring both sides filters out waived or retired veterans still being
    paid (who look like "missed" seasons but aren't injuries) and draft
    picks not yet in the league. The cost: career-ending injuries, where the
    player never returns, are not counted.
    """
    played = stats[["player_id", "player", "name_key", "season", "age"]]
    paid = sal[["name_key", "season"]]
    have_name = set(zip(stats["name_key"], stats["season"]))
    have_id = set(zip(stats["player_id"], stats["season"]))

    missing = paid[[(k, s) not in have_name for k, s in zip(paid["name_key"], paid["season"])]]
    missing = missing[missing["season"] < stats["season"].max()]

    rows = []
    for key, season in zip(missing["name_key"], missing["season"]):
        before = played[(played["name_key"] == key) & (played["season"] == season - 1)]
        after = played[(played["name_key"] == key) & (played["season"] == season + 1)]
        ids = set(before["player_id"]) | set(after["player_id"])
        if len(before) != 1 or len(after) != 1 or len(ids) != 1:
            continue  # didn't play on both sides, or ambiguous name: skip
        ref = before.iloc[0]
        if (ref["player_id"], season) in have_id:
            continue  # he did play that season, under a different name

        rows.append({
            "player_id": ref["player_id"], "player": ref["player"], "name_key": key,
            "season": season, "age": ref["age"] + (season - ref["season"]),
            "team": None, "games": 0, "team_games": 82, "availability": 0.0,
            "minutes": 0, "bpm": None, "vorp": 0.0, "ws": 0.0,
        })
    return pd.concat([stats, pd.DataFrame(rows)], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = load_stats()
    sal = load_salaries()
    n_before = len(stats)
    stats = add_missed_seasons(stats, sal)
    print(f"Added {len(stats) - n_before} fully missed seasons (paid, zero games)")

    table = stats.merge(sal, on=["name_key", "season"], how="left")
    # The newest season is still being played, so its games and stats are partial.
    table["season_complete"] = table["season"] < table["season"].max()
    table = table.drop(columns="name_key").sort_values(["season", "player"])
    table.to_csv(OUT / "player_seasons.csv", index=False)

    # Coverage report: how many real rotation players got a salary attached.
    rotation = table[(table["minutes"] >= 500) & (table["season"].between(FIRST_SEASON, 2024))]
    print(f"Wrote {len(table):,} player-seasons to {OUT / 'player_seasons.csv'}")
    print(f"Salary match rate, rotation players {FIRST_SEASON}-2024: "
          f"{rotation['salary'].notna().mean():.1%}")
    by_season = rotation.groupby("season")["salary"].apply(lambda s: s.notna().mean())
    print("Match rate by season:")
    print(by_season.map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
