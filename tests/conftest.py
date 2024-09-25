from pathlib import Path

import pytest
from cogent3 import SequenceCollection, load_unaligned_seqs


@pytest.fixture(scope="session")
def DATA_DIR() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def seq_path(DATA_DIR: Path) -> Path:
    return DATA_DIR / "brca1.fasta"


@pytest.fixture
def unaligned_seqs(seq_path: Path) -> SequenceCollection:
    return load_unaligned_seqs(seq_path)


@pytest.fixture
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dvgt")
