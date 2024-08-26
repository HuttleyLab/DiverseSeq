import pathlib
import sys

import h5py
import pytest
from click.testing import CliRunner
from cogent3 import load_table, load_unaligned_seqs

from diverse_seq.cli import max as dvs_max
from diverse_seq.cli import nmost as dvs_nmost
from diverse_seq.cli import prep as dvs_prep
from diverse_seq.data_store import HDF5DataStore

__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dvgt")


@pytest.fixture(scope="function")
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("seqdata")


@pytest.fixture(scope="session")
def runner():
    """exportrc works correctly."""
    return CliRunner()


@pytest.fixture(scope="session")
def seq_path():
    return DATADIR / "brca1.fasta"


@pytest.fixture(scope="function")
def rna_seq_path(tmp_dir, seq_path):
    seqs = load_unaligned_seqs(seq_path, moltype="rna")
    path = tmp_dir / "test_rna.fasta"
    seqs.write(path)
    return path


@pytest.fixture(scope="function")
def seq_dir(tmp_path, seq_path):
    seqs = load_unaligned_seqs(seq_path, moltype="dna")
    for s in seqs.iter_seqs():
        fn = tmp_path / f"{s.name}.fasta"
        fn.write_text(s.to_fasta(), encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="session")
def processed_seq_path():
    return DATADIR / "brca1.dvseqs"


def _checked_output(path, eval_rows=None, eval_header=None):
    table = load_table(str(path))
    assert table.shape[1] == 2, table.shape[1]
    if eval_rows:
        assert eval_rows(table.shape[0])

    if eval_header:
        assert eval_header(table.header)


def _checked_h5_dstore(path, source=None):
    # TODO: how to more thoroughly interrogate output
    dstore = HDF5DataStore(path)
    assert len(dstore.completed) > 0
    assert len(dstore.not_completed) == 0


def test_defaults(runner, tmp_dir, processed_seq_path):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {processed_seq_path} -o {outpath}".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(str(outpath))


@pytest.mark.parametrize("min_size", (2, 5))
def test_min_size(runner, tmp_dir, processed_seq_path, min_size):
    outpath = tmp_dir / "test_min_size.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -z {min_size}".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x >= min_size)


@pytest.mark.parametrize("max_size", (5, 7))
def test_max_size(runner, tmp_dir, processed_seq_path, max_size):
    outpath = tmp_dir / "test_max_size.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -z 3 -zp {max_size}".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x <= max_size)


def test_max_include(runner, tmp_dir, processed_seq_path):
    outpath = tmp_dir / "test_include.tsv"
    args = (
        f"-s {processed_seq_path} -o {outpath} -z 3 -zp 5 -L 10 -k 1 -i Human".split()
    )
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    got = load_table(outpath)
    assert "Human" in got.columns["names"]


@pytest.mark.parametrize("size", (5, 7))
def test_min_eq_max(runner, tmp_dir, processed_seq_path, size):
    outpath = tmp_dir / "test_min_eq_max.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -z {size} -zp {size}".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x == size)


def test_min_gt_max_fail(runner, tmp_dir, processed_seq_path):
    outpath = tmp_dir / "test_min_gt_max_fail.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -zp 2".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code != 0, r.output


@pytest.mark.parametrize("stat", ("stdev", "cov"))
def test_stat(runner, tmp_dir, processed_seq_path, stat):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -st {stat}".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath)


@pytest.mark.parametrize("k", (1, 3, 7))
def test_k(runner, tmp_dir, processed_seq_path, k):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -k {k}".split()
    r = runner.invoke(dvs_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath)


def test_nmost(runner, tmp_dir, processed_seq_path):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -k 1 -n 5".split()
    r = runner.invoke(dvs_nmost, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x == 5)


def test_nmost_include(runner, tmp_dir, processed_seq_path):
    outpath = tmp_dir / "test_include.tsv"
    args = f"-s {processed_seq_path} -o {outpath} -k 1 -n 5 -L 10 -i Human".split()
    r = runner.invoke(dvs_nmost, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    got = load_table(outpath)
    assert "Human" in got.columns["names"]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Test skipped on Windows")
def test_prep_seq_file(runner, tmp_dir, seq_path):
    outpath = tmp_dir / "test_prep_seq_file.dvseqs"
    args = f"-s {seq_path} -o {outpath} -sf fasta".split()
    r = runner.invoke(dvs_prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_h5_dstore(str(outpath))


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Test skipped on Windows")
def test_prep_outpath_without_suffix(runner, tmp_dir, seq_path):
    outpath = tmp_dir / "test_prep_outpath_without_suffix"
    args = f"-s {seq_path} -o {outpath}".split()
    r = runner.invoke(dvs_prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_h5_dstore(str(outpath) + ".dvseqs")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Test skipped on Windows")
def test_prep_force_override(runner, tmp_dir, seq_path):
    outpath = tmp_dir / "test_prep_force_override.dvseqs"
    args = f"-s {seq_path} -o {outpath}".split()

    # Run prep once, it should succeed
    r = runner.invoke(dvs_prep, args)
    assert r.exit_code == 0, r.output
    _checked_h5_dstore(str(outpath))

    # with the force write flag, it should succeed
    args += ["-F"]
    r = runner.invoke(dvs_prep, args)
    assert r.exit_code == 0, r.output
    _checked_h5_dstore(str(outpath))


def test_prep_max_rna(runner, tmp_dir, rna_seq_path):
    outpath = tmp_dir / "test_prep_max_rna.dvseqs"
    args = f"-s {rna_seq_path} -o {outpath} -m rna".split()
    r = runner.invoke(dvs_prep, args)
    assert r.exit_code == 0, r.output
    _checked_h5_dstore(str(outpath))

    max_outpath = tmp_dir / "test_prep_rna.tsv"
    max_args = f"-s {outpath} -o {max_outpath}".split()
    r = runner.invoke(dvs_max, max_args)
    assert r.exit_code == 0, r.output
    _checked_output(str(max_outpath))


def test_prep_source_from_file(runner, tmp_dir, seq_path):
    outpath = tmp_dir / "test_prep_source_from_file.dvseqs"
    args = f"-s {seq_path} -o {outpath} -sf fasta".split()
    r = runner.invoke(dvs_prep, args)
    assert r.exit_code == 0


def test_prep_source_from_directory(runner, tmp_dir, seq_dir):
    outpath = tmp_dir / "test_prep_source_from_directory.dvseqs"
    args = f"-s {seq_dir} -o {outpath} -sf fasta".split()
    r = runner.invoke(dvs_prep, args)
    assert r.exit_code == 0, r.output
    with h5py.File(outpath, mode="r") as f:
        for name, dset in f.items():
            if name == "md5":
                continue
            assert dset.attrs["source"] == str(seq_dir)
