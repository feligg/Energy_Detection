"""Download the three BDG2 files needed by the project.

IMPORTANT: the big CSV files in the BDG2 repo are stored with Git LFS.
`raw.githubusercontent.com` only returns a tiny text *pointer* (~130 bytes),
so we download from `media.githubusercontent.com`, which resolves LFS files.
"""
import sys
import urllib.request
from pathlib import Path

BASE = ("https://media.githubusercontent.com/media/buds-lab/"
        "building-data-genome-project-2/master/data/")
FILES = {
    "metadata.csv": "metadata/metadata.csv",
    "weather.csv": "weather/weather.csv",
    "electricity_cleaned.csv": "meters/cleaned/electricity_cleaned.csv",
}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, rel in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[skip] {name} already present")
            continue
        print(f"[download] {name} ...")
        urllib.request.urlretrieve(BASE + rel, dest)
        size = dest.stat().st_size
        if size < 10_000:
            sys.exit(f"{name} is only {size} bytes - looks like a Git-LFS pointer")
        print(f"           {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()