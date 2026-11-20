"""
Downloads every raw dataset the project uses into data/raw/.

Sources (all free, public CSVs on GitHub):
  - Basketball-Reference stats, draft history, team records:
    github.com/sumitrodatta/bball-reference-datasets
  - Historical salaries and salary cap history:
    github.com/gabriel1200/site_Data

Run once from the project root:
    python src/fetch_data.py
"""

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

BREF = "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data/"
SALARY = "https://raw.githubusercontent.com/gabriel1200/site_Data/master/"

FILES = {
    "advanced.csv": BREF + quote("Advanced.csv"),
    "totals.csv": BREF + quote("Player Totals.csv"),
    "draft.csv": BREF + quote("Draft Pick History.csv"),
    "team_summaries.csv": BREF + quote("Team Summaries.csv"),
    "salaries.csv": SALARY + "nba_salaries_master.csv",
    "contracts_current.csv": SALARY + "salary.csv",
    "cap.csv": SALARY + "cap.csv",
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print(f"Downloading {name}...")
        urlretrieve(url, RAW_DIR / name)
    print(f"Done. Files saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
