import pathlib
import time
from plotly.io import write_image

PAPER_DIR = pathlib.Path(__file__).parent.parent

DATA_DIR = PAPER_DIR / "data"
RESULT_DIR = PAPER_DIR / "results"

FIG_DIR = PAPER_DIR / "figs"
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR.mkdir(exist_ok=True)


class pdf_writer:
    """class that handles super annoying mathjax warning box in plotly pdf's"""

    def __init__(self) -> None:
        self._done_once = False

    def __call__(self, fig, path):
        # the sleep, plus successive write, is ESSENTIAL to avoid the super annoying
        # "[MathJax]/extensions/MathMenu.js" text box error
        # but we only need to do this once
        if not self._done_once:
            write_image(fig, path)
            time.sleep(2)
            self._done_once = True

        write_image(fig, path)
