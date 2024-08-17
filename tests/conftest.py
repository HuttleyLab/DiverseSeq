import pathlib

import pytest


@pytest.fixture()
def DATA_DIR():
    return pathlib.Path(__file__).parent / "data"
