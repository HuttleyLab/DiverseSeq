import numpy
import pytest
from cogent3 import (
    get_app,
    get_dataset,
    load_aligned_seqs,
    load_unaligned_seqs,
    make_aligned_seqs,
    make_seq,
    make_unaligned_seqs,
    open_data_store,
)
from cogent3.app.composable import NotCompleted
from cogent3.maths.measure import jsd
from numpy.testing import assert_allclose

from diverse_seq import _dvs as dvs
from diverse_seq import load_sample_data
from diverse_seq import records as dvs_records
from diverse_seq.util import populate_inmem_zstore


@pytest.fixture
def seqcoll():
    data = {"a": "AAAA", "b": "AAAA", "c": "TTTT", "d": "ACGT"}
    return make_unaligned_seqs(data, moltype="dna")


@pytest.fixture
def zstore(seqcoll):
    return populate_inmem_zstore(seqcoll)


@pytest.mark.parametrize("k", [1, 2])
def test_total_jsd(zstore, k):
    lzseqs = [zstore.get_lazyseq(n, num_states=4) for n in zstore.unique_seqids]
    sr = dvs.get_delta_jsd_calculator(
        [(s.seqid, s.get_seq()) for s in lzseqs], k=k, num_states=4
    ).get_result()
    freqs = {s.seqid: numpy.array(s.get_kfreqs(k)) for s in lzseqs}
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_mean_delta_jsd(zstore):
    k = 1
    lzseqs = [zstore.get_lazyseq(n, num_states=4) for n in zstore.get_seqids()]
    wout_a = [lz for lz in lzseqs if lz.seqid != "a"]
    assert len(wout_a) == len(lzseqs) - 1
    sr_with_a = dvs.get_delta_jsd_calculator(
        [(s.seqid, s.get_seq()) for s in lzseqs], k=k, num_states=4
    ).get_result()
    sr_without_a = dvs.get_delta_jsd_calculator(
        [(s.seqid, s.get_seq()) for s in wout_a], k=k, num_states=4
    ).get_result()
    assert sr_without_a.mean_delta_jsd > sr_with_a.mean_delta_jsd


def test_max_divergent(seqcoll):
    zstore = populate_inmem_zstore(seqcoll)
    k = 1
    got = dvs.max_divergent(zstore, min_size=2, max_size=2, k=k)
    assert got.size == 2


def test_max_divergent_min_to_big(seqcoll):
    zstore = populate_inmem_zstore(seqcoll)
    k = 1
    with pytest.raises(ValueError):
        dvs.max_divergent(zstore, min_size=30, max_size=2, k=k)


def test_most_divergent(zstore):
    k = 1
    got = dvs.nmost_divergent(zstore, n=3, k=k)
    assert got.size == 3
    got_seqids = set(got.record_names)
    seqids = set(zstore.unique_seqids)
    assert got_seqids == seqids


def test_most_divergent_n_to_big(zstore):
    k = 1
    with pytest.raises(ValueError):
        dvs.nmost_divergent(zstore, n=30, k=k)


def test_most_divergent_pickle(zstore):
    # can we pickle a nmost_divergent result and unpickle it
    import pickle  # noqa: PLC0415

    k = 1
    orig = dvs.nmost_divergent(zstore, n=3, k=k)
    assert orig.size == 3
    dumped = pickle.dumps(orig)
    loaded = pickle.loads(dumped)  # noqa: S301
    assert loaded.size == orig.size


@pytest.fixture(scope="session")
def brca1_coll(DATA_DIR):
    return load_unaligned_seqs(DATA_DIR / "brca1.fasta", moltype="dna").degap()


@pytest.fixture(scope="session")
def brca1_zstore(DATA_DIR):
    seqcoll = load_unaligned_seqs(DATA_DIR / "brca1.fasta", moltype="dna").degap()
    store = dvs.make_zarr_store()
    for seq in seqcoll.seqs:
        arr = numpy.array(seq)
        store.write(seq.name, arr.tobytes())
    return store


@pytest.fixture
def brca1_zstore_path(DATA_DIR, tmp_path):
    seqcoll = get_dataset("brca1").degap()
    outpath = tmp_path / "brca1.dvseqsz"
    store = dvs.make_zarr_store(str(outpath), mode="w")
    for seq in seqcoll.seqs:
        arr = numpy.array(seq)
        store.write(seq.name, arr.tobytes())
    return outpath


@pytest.fixture(scope="session")
def brca1_alignment(DATA_DIR):
    return load_aligned_seqs(DATA_DIR / "brca1.fasta", moltype="dna")


def test_merge_summed_records(brca1_zstore_path):
    path = brca1_zstore_path
    zstore = dvs.make_zarr_store(str(path))
    names = zstore.unique_seqids
    app = dvs_records.select_nmost(seq_store=path, n=5, k=1)
    sr1 = app(names[:10])  # pylint: disable=not-callable
    sr2 = app(names[10:20])  # pylint: disable=not-callable
    rnames1 = sr1.record_names
    rnames2 = sr2.record_names
    assert set(rnames1) != set(rnames2)
    got = dvs_records.select_final_nmost(n=5)([sr1, sr2])
    assert len(got.record_names) == 5


def test_select_final_nmost_n_to_big(brca1_zstore_path):
    path = brca1_zstore_path
    zstore = dvs.make_zarr_store(str(path))
    names = zstore.unique_seqids
    app = dvs_records.select_nmost(seq_store=path, n=5, k=1)
    sr1 = app(names[:10])  # pylint: disable=not-callable
    sr2 = app(names[10:20])  # pylint: disable=not-callable
    rnames1 = sr1.record_names
    rnames2 = sr2.record_names
    assert set(rnames1) != set(rnames2)
    got = dvs_records.select_final_nmost(n=500)([sr1, sr2])
    assert isinstance(got, NotCompleted)


def test_select_final_max_min_to_big(brca1_zstore_path):
    path = brca1_zstore_path
    zstore = dvs.make_zarr_store(str(path))
    names = zstore.unique_seqids
    app = dvs_records.select_max(seq_store=path, min_size=4, max_size=5, k=1)
    sr1 = app(names[:10])  # pylint: disable=not-callable
    app = dvs_records.select_final_max(min_size=10, max_size=20, stat="stdev")
    got = app([sr1])
    assert isinstance(got, NotCompleted)


def test_dvs_select_max(brca1_coll):
    app = dvs_records.dvs_max(k=1, min_size=2, max_size=5)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert 2 <= got.num_seqs <= 5
    app = dvs_records.dvs_max(k=1, min_size=2, max_size=5, seed=123)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert 2 <= got.num_seqs <= 5


@pytest.mark.parametrize("include", ("Human", ["Human"], ["Human", "Mouse"]))
def test_dvs_select_max_include(brca1_coll, include):
    app = dvs_records.dvs_max(k=1, min_size=2, max_size=5, include=include)
    got = app(brca1_coll)  # pylint: disable=not-callable
    include = {include} if isinstance(include, str) else set(include)
    assert include <= set(got.names)


def test_dvs_select_nmost(brca1_coll):
    app = dvs_records.dvs_nmost(k=1, n=5)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert got.num_seqs == 5
    # try setting a seed
    app = dvs_records.dvs_nmost(k=1, n=5, seed=123)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert got.num_seqs == 5


@pytest.mark.parametrize("include", ("Human", ["Human"], ["Human", "Mouse"]))
def test_dvs_select_nmost_keep(brca1_coll, include):
    app = dvs_records.dvs_nmost(k=1, n=5, include=include)
    got = app(brca1_coll)  # pylint: disable=not-callable
    include = {include} if isinstance(include, str) else set(include)
    assert include <= set(got.names)


@pytest.mark.parametrize("app_name", ("dvs_nmost", "dvs_max"))
def test_serialisable_nmost(brca1_coll, tmp_path, app_name):
    brca1_coll.info.source = "blah.fa"
    outstore = open_data_store(tmp_path / "data.sqlitedb", mode="w")
    select = get_app(app_name, k=2)
    writer = get_app("write_db", data_store=outstore)
    app = select + writer
    _ = app(brca1_coll)  # pylint: disable=not-callable
    assert len(outstore.completed) == 1


@pytest.mark.parametrize("app_name", ("dvs_nmost", "dvs_max"))
@pytest.mark.parametrize("aligned", (False, True))
def test_select_return_type(brca1_alignment, app_name, aligned):
    coll = brca1_alignment if aligned else brca1_alignment.degap()
    select = get_app(app_name, k=2)
    got = select(coll)  # pylint: disable=not-callable
    assert isinstance(got, coll.__class__)


def test_dvs_delta_jsd_zero_length_in_ref():
    data = {"s1": "ACGT-A", "s2": "------"}
    seqs = make_aligned_seqs(data, moltype="dna")
    with pytest.raises(ValueError):
        dvs_records.dvs_delta_jsd(seqs=seqs, k=1)


def test_dvs_delta_jsd_zero_length_query():
    data = {"s1": "ACGTA", "s2": "ACGTA"}
    seqs = make_unaligned_seqs(data, moltype="dna")
    app = dvs_records.dvs_delta_jsd(seqs=seqs, k=1, moltype="dna")
    query = make_seq("", name="s3", moltype="dna")
    name, delta = app(query)  # pylint: disable=not-callable
    assert name == "s3"
    assert numpy.isnan(delta)


def test_dvs_delta_jsd_moltype():
    data = {"s1": "ACGTA", "s2": "ACGTA"}
    seqs = make_unaligned_seqs(data, moltype="dna")
    app = dvs_records.dvs_delta_jsd(seqs=seqs, k=1, moltype="dna")
    query = make_seq("ACAAA", name="s3", moltype="dna")
    _, expect = app(query)  # pylint: disable=not-callable
    query = make_seq("ACAAA", name="s3", moltype="text")
    _, got = app(query)  # pylint: disable=not-callable
    assert expect == got


@pytest.fixture
def ref_query_seqs():
    seqcoll = load_sample_data()
    sliced = [seq[1500:1650] for seq in seqcoll.seqs]
    newcoll = make_unaligned_seqs(sliced, moltype="dna")

    num = 4
    ref_seqs = newcoll.take_seqs(seqcoll.names[:num])
    query_seqs = newcoll.take_seqs(seqcoll.names[:num], negate=True)
    return ref_seqs, query_seqs


def test_jsd_calc(ref_query_seqs):
    ref_seqs, _ = ref_query_seqs
    calc = dvs.get_delta_jsd_calculator(
        [(s.name, s.to_array().tobytes()) for s in ref_seqs.seqs], k=3, num_states=4
    )
    sr = calc.get_result()
    assert sr.total_jsd > 0.0


def test_jsd_calc_exists(ref_query_seqs):
    ref_seqs, _ = ref_query_seqs
    seq_records = [(s.name, s.to_array().tobytes()) for s in ref_seqs.seqs]
    calc = dvs.get_delta_jsd_calculator(seq_records, k=3, num_states=4)
    got = calc.delta_jsd(*seq_records[0])
    assert numpy.allclose(got, 0.0)


def test_jsd_calc_empty_seq(ref_query_seqs):
    ref_seqs, _ = ref_query_seqs
    calc = dvs.get_delta_jsd_calculator(
        [(s.name, s.to_array().tobytes()) for s in ref_seqs.seqs], k=3, num_states=4
    )
    with pytest.raises(ValueError):
        calc.delta_jsd("blah", b"")
