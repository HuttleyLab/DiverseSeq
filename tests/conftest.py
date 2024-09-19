import pathlib

import pytest
from cogent3 import load_unaligned_seqs


@pytest.fixture(scope="session")
def DATA_DIR():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def seq_path(DATA_DIR):
    return DATA_DIR / "brca1.fasta"


@pytest.fixture
def unaligned_seqs(seq_path):
    return load_unaligned_seqs(seq_path)


@pytest.fixture(scope="function")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dvgt")
