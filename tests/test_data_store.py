import pathlib

import pytest
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from numpy.testing import assert_array_equal

from divergent import data_store
from divergent.io import dvgt_file_to_dir, dvgt_load_seqs, dvgt_write_seqs
from divergent.util import str2arr

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def brca1_seqs():
    return load_unaligned_seqs(DATADIR / "brca1.fasta", moltype="dna").degap()


@pytest.fixture(scope="function")
def brca1_5(brca1_seqs):
    seqs = brca1_seqs.take_seqs(["Cat", "Dog", "Wombat", "Horse", "Rat"])
    return make_unaligned_seqs(
        data={s.name: str(s[:20]) for s in seqs.seqs},
        moltype="dna",
    )


@pytest.fixture(scope="function")
def brca1_dstore(tmp_path):
    dstore_maker = dvgt_file_to_dir(tmp_path / "brca1_dstore")
    return dstore_maker(DATADIR / "brca1.fasta")


@pytest.fixture(scope="function")
def brca1_5_dstore(tmp_path, brca1_5):
    fasta_path = tmp_path / "brca1_5.fasta"
    brca1_5.write(fasta_path)
    dstore_maker = dvgt_file_to_dir(tmp_path / "brca1_5_dstore")
    return dstore_maker(fasta_path)


@pytest.fixture(scope="function")
def hdf5_dstore_path(tmp_path):
    return tmp_path / "hdf5_dstore"


def test_dvgt_file_to_dir(brca1_5_dstore, brca1_5):
    # directory contains the same number of sequences as the input file
    assert brca1_5.num_seqs == len(brca1_5_dstore.completed)


@pytest.mark.parametrize("parallel", (False, True))
def test_prep_pipeline(brca1_5, brca1_5_dstore, hdf5_dstore_path, parallel):
    # initialise and apply pipeline
    dstore = data_store.HDF5DataStore(source=hdf5_dstore_path, mode="w")
    prep = dvgt_load_seqs(moltype="dna") + dvgt_write_seqs(data_store=dstore)
    result = prep.apply_to(brca1_5_dstore, parallel=parallel)

    # output datastore contains same number of records as seqs in orig file
    assert brca1_5.num_seqs == len(result.completed)

    # check the sequence data matches
    seq_data = result.read("Cat")
    str_to_array = str2arr(moltype="dna")
    orig_seq_data = str_to_array(str(brca1_5.get_seq("Cat")))
    assert_array_equal(seq_data, orig_seq_data)


def test_get_seqids(DATA_DIR):
    fasta_path = DATA_DIR / "brca1.fasta"
    expect = set(load_unaligned_seqs(fasta_path).names)
    store_path = DATA_DIR / "brca1.dvgtseqs"

    got = data_store.get_seqids_from_store(store_path)

    assert set(got) == expect
