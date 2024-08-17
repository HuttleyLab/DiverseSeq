import pathlib

import pytest
from cogent3 import load_unaligned_seqs
from numpy.testing import assert_array_equal

from divergent import data_store
from divergent.io import dvgt_load_seqs, dvgt_write_prepped_seqs, dvgt_write_seq_store
from divergent.util import str2arr

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_path(tmpdir_factory):
    return tmpdir_factory.mktemp("dvgt")


@pytest.fixture(scope="session")
def brca1_seqs():
    return load_unaligned_seqs(DATADIR / "brca1.fasta", moltype="dna")


@pytest.fixture(scope="session")
def brca1_dstore(tmp_path):
    dstore_maker = dvgt_write_seq_store(tmp_path / "brca1_dstore")
    return dstore_maker(DATADIR / "brca1.fasta")


@pytest.fixture(scope="function")
def hdf5_dstore(tmp_path):
    return data_store.HDF5DataStore(tmp_path / "hdf5_dstore")


def test_dvgt_write_seq_store(brca1_dstore, brca1_seqs):
    # directory contains the same number of sequences as the input file
    assert brca1_seqs.num_seqs == len(brca1_dstore.completed)


@pytest.mark.parametrize("parallel", (True, False))
def test_prep_pipeline(brca1_seqs, brca1_dstore, hdf5_dstore, parallel):
    # initialise and apply pipeline
    prep = dvgt_load_seqs(moltype="dna") + dvgt_write_prepped_seqs(hdf5_dstore.source)
    result = prep.apply_to(brca1_dstore, parallel=parallel)

    # output datastore contains same number of records as seqs in orig file
    assert brca1_seqs.num_seqs == len(result.completed)

    # check the sequence data matches
    seq_data = result.read("Cat")
    str_to_array = str2arr(moltype="dna")
    orig_seq_data = str_to_array(brca1_seqs.get_seq("Cat")._seq.seq)
    assert_array_equal(seq_data, orig_seq_data)


def test_get_seqids(DATA_DIR):
    fasta_path = DATA_DIR / "brca1.fasta"
    expect = set(load_unaligned_seqs(fasta_path).names)
    store_path = DATA_DIR / "brca1.dvgtseqs"

    got = data_store.get_seqids_from_store(store_path)

    assert set(got) == expect
