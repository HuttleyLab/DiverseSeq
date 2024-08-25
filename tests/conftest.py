import pathlib

import pytest


@pytest.fixture
def DATA_DIR():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dvgt")
