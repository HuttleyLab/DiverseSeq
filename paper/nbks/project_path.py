import pathlib

PAPER_DIR = pathlib.Path(__file__).parent.parent

FIG_DIR = PAPER_DIR / "figs"
TABLE_DIR = PAPER_DIR / "tables"

FIG_DIR.mkdir(exist_ok=True)
