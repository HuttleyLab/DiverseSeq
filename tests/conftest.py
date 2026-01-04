from pathlib import Path

import pytest
from cogent3 import load_unaligned_seqs
from cogent3.core.alignment import SequenceCollection


@pytest.fixture(scope="session")
def DATA_DIR() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def processed_seq_path(tmp_path):
    import numpy as np
    from cogent3 import get_dataset

    from diverse_seq import _dvs as dvs

    outpath = tmp_path / "brca1.dvseqsz"
    dstore = dvs.make_zarr_store(str(outpath), mode="w")
    seqcoll = get_dataset("brca1").degap()
    for seq in seqcoll.seqs:
        seqarr = np.array(seq)
        metadata = {"source": f"brca1-dataset:{seq.name}"}
        dstore.write(seq.name, seqarr.tobytes(), metadata)

    return outpath


@pytest.fixture(scope="session")
def seq_path(DATA_DIR: Path) -> Path:
    return DATA_DIR / "brca1.fasta"


@pytest.fixture
def unaligned_seqs(seq_path: Path) -> SequenceCollection:
    return load_unaligned_seqs(seq_path, moltype="dna")


@pytest.fixture
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dvgt")
