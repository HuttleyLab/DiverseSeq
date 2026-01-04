import cogent3 as c3
import numpy as np
import pytest

from diverse_seq import _dvs as dvs


def test_can_access_rust_func(tmp_path):
    result = dvs.make_zarr_store(str(tmp_path / "test.zarr"))
    assert not len(result)


@pytest.fixture
def empty_zarr_store(tmp_path):
    return dvs.make_zarr_store(str(tmp_path / "empty.zarr"))


@pytest.fixture
def inmem_zarr_store():
    return dvs.make_zarr_store()


@pytest.fixture(params=["empty_zarr_store", "inmem_zarr_store"])
def zstore(request):
    return request.getfixturevalue(request.param)


def test_add_seq(zstore):
    result = zstore
    assert not len(result)
    seq = np.array([0, 3, 2, 0], dtype=np.uint8)
    result.write("s3", seq.tobytes())
    assert "s3" in result
    retrieved = result.read("s3")
    np.testing.assert_array_equal(np.frombuffer(retrieved, dtype=np.uint8), seq)


def test_num_unique(zstore):
    result = zstore
    assert not len(result)
    seq = np.array([0, 3, 2, 0], dtype=np.uint8)
    result.write("s3", seq.tobytes())
    result.write("s3", seq.tobytes())
    assert result.num_unique() == 1
    assert len(result) == 1
    result.write("s4", seq.tobytes())
    assert result.num_unique() == 1
    assert len(result) == 2
    assert result.unique_seqids == ["s4"]


def test_write_many(zstore):
    result = zstore
    seqcoll = c3.get_dataset("brca1").degap()
    for seq in seqcoll.seqs:
        arr = np.array(seq)
        result.write(
            seq.name, arr.tobytes(), {"source": f"{seqcoll.source}:{seq.name}"}
        )
    assert len(result) == seqcoll.num_seqs
    orig = np.array(seqcoll.seqs["Human"]).tobytes()
    got = result.read("Human")
    assert orig == got
    metadata = result.read_metadata("Human")
    assert metadata["source"] == f"{seqcoll.source}:Human"


def test_pickle(empty_zarr_store):
    import pickle

    result = empty_zarr_store
    seq = np.array([0, 3, 2, 0], dtype=np.uint8)
    result.write("s3", seq.tobytes())
    dumped = pickle.dumps(result)
    loaded = pickle.loads(dumped)
    assert "s3" in loaded
    retrieved = loaded.read("s3")
    np.testing.assert_array_equal(np.frombuffer(retrieved, dtype=np.uint8), seq)


def test_pickle_memstore(inmem_zarr_store):
    import pickle  # noqa: PLC0415

    result = inmem_zarr_store
    seq = np.array([0, 3, 2, 0], dtype=np.uint8)
    result.write("s3", seq.tobytes())
    with pytest.raises(TypeError):
        _ = pickle.dumps(result)
