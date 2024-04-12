import pathlib

import pytest

from cogent3 import load_unaligned_seqs, open_data_store
from numpy.testing import assert_array_equal

from divergent.data_store import HDF5DataStore
from divergent.loader import dvgt_load_seqs, dvgt_seq_file_to_data_store
from divergent.util import str2arr
from divergent.writer import dvgt_write_prepped_seqs


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_path(tmpdir_factory):
    return tmpdir_factory.mktemp("dvgt")


@pytest.fixture(scope="session")
def brca1_seqs():
    return load_unaligned_seqs(DATADIR / "brca1.fasta", moltype="dna")


@pytest.fixture(scope="session")
def brca1_dstore(tmp_path):
    dstore_maker = dvgt_seq_file_to_data_store(tmp_path / "brca1_dstore")
    return dstore_maker(DATADIR / "brca1.fasta")


@pytest.fixture(scope="function")
def hdf5_dstore(tmp_path):
    return HDF5DataStore(tmp_path / "hdf5_dstore")


def test_dvgt_seq_file_to_data_store(brca1_dstore, brca1_seqs):
    # directory contains the same number of sequences as the input file
    assert brca1_seqs.num_seqs == len(brca1_dstore.completed)


@pytest.mark.parametrize("parallel", (True, False))
def test_prep_pipeline(brca1_seqs, brca1_dstore, hdf5_dstore, parallel):
    # initialise and apply pipeline
    prep = dvgt_load_seqs(moltype="dna") + dvgt_write_prepped_seqs(hdf5_dstore)
    result = prep.apply_to(brca1_dstore, parallel=parallel)

    # output datastore contains same number of records as seqs in orig file
    assert brca1_seqs.num_seqs == len(result.completed)

    # check the sequence data matches
    seq_data = result.read("Cat")
    str_to_array = str2arr(moltype="dna")
    orig_seq_data = str_to_array(brca1_seqs.get_seq("Cat")._seq.seq)
    assert_array_equal(seq_data, orig_seq_data)
