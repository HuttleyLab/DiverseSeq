import pathlib

import pytest
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from numpy.testing import assert_array_equal

from diverse_seq import _dvs as dvs
from diverse_seq import io as dvs_io
from diverse_seq.util import str2arr

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def brca1_seqs():
    return load_unaligned_seqs(DATADIR / "brca1.fasta", moltype="dna").degap()


@pytest.fixture(scope="function")
def brca1_5(brca1_seqs):
    seqs = brca1_seqs.take_seqs(["Cat", "Dog", "Wombat", "Horse", "Rat"])
    return make_unaligned_seqs(
        {s.name: str(s[:20]) for s in seqs.seqs},
        moltype="dna",
    )


@pytest.fixture(scope="function")
def brca1_dstore(tmp_path):
    dstore_maker = dvs_io.dvs_file_to_dir(tmp_path / "brca1_dstore")
    return dstore_maker(DATADIR / "brca1.fasta")  # pylint: disable=not-callable


@pytest.fixture(scope="function")
def brca1_5_dstore(tmp_path, brca1_5):
    fasta_path = tmp_path / "brca1_5.fasta"
    brca1_5.write(fasta_path)
    dstore_maker = dvs_io.dvs_file_to_dir(tmp_path / "brca1_5_dstore")
    return dstore_maker(fasta_path)  # pylint: disable=not-callable


@pytest.fixture(scope="function")
def zarrstore_path(tmp_path):
    return tmp_path / "zarrstore.dvseqsz"


def test_dvs_file_to_dir(brca1_5_dstore, brca1_5):
    # directory contains the same number of sequences as the input file
    assert brca1_5.num_seqs == len(brca1_5_dstore.completed)


@pytest.mark.parametrize("parallel", (False, True))
def test_prep_pipeline(brca1_5, brca1_5_dstore, zarrstore_path, parallel):
    # initialise and apply pipeline
    dstore = dvs.make_zarr_store(str(zarrstore_path), mode="w")
    prep = dvs_io.dvs_load_seqs(moltype="dna") + dvs_io.dvs_write_seqs(
        data_store=dstore,
    )
    result = prep.apply_to(
        brca1_5_dstore, parallel=parallel, id_from_source=dvs_io.get_unique_id
    )

    # output datastore contains same number of records as seqs in orig file
    assert brca1_5.num_seqs == len(result.get_seqids())

    # check the sequence data matches
    seq_data = result.read("Cat")
    str_to_array = str2arr(moltype="dna")
    orig_seq_data = str_to_array(str(brca1_5.get_seq("Cat")))  # pylint: disable=not-callable
    assert_array_equal(seq_data, orig_seq_data.tobytes())


def test_get_seqids(DATA_DIR, processed_seq_path):
    fasta_path = DATA_DIR / "brca1.fasta"
    expect = set(load_unaligned_seqs(fasta_path, moltype="dna").names)

    got = dvs.get_seqids_from_store(str(processed_seq_path))

    assert set(got) == expect
