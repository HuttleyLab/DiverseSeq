import pathlib

import h5py
import pytest

from click.testing import CliRunner
from cogent3 import load_table

from divergent.cli import max as dvgt_max
from divergent.cli import prep as dvgt_prep


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("dvgt")


@pytest.fixture(scope="session")
def runner():
    """exportrc works correctly."""
    return CliRunner()


@pytest.fixture(scope="session")
def fasta_seq_path():
    return DATADIR / "brca1.fasta"


@pytest.fixture(scope="session")
def h5_seq_path():
    return DATADIR / "brca1.h5"


def _checked_output(path, eval_rows=None, eval_header=None):
    table = load_table(str(path))
    assert table.shape[1] == 2, table.shape[1]
    if eval_rows:
        assert eval_rows(table.shape[0])

    if eval_header:
        assert eval_header(table.header)


def _checked_h5_output(path, source=None):
    with h5py.File(path, mode="r") as f:
        assert len(f.keys()) > 0
        assert isinstance(f.attrs["moltype"], str)

        if source:
            assert f.attrs["source"] == source

        for _, dataset in f.items():
            assert isinstance(dataset, h5py.Dataset)
            assert dataset.dtype == "u1"


def test_defaults(runner, tmp_dir, h5_seq_path):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {h5_seq_path} -o {outpath}".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(str(outpath))


@pytest.mark.parametrize("min_size", (2, 5))
def test_min_size(runner, tmp_dir, h5_seq_path, min_size):
    outpath = tmp_dir / "test_min_size.tsv"
    args = f"-s {h5_seq_path} -o {outpath} -z {min_size}".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x >= min_size)


@pytest.mark.parametrize("max_size", (5, 7))
def test_max_size(runner, tmp_dir, h5_seq_path, max_size):
    outpath = tmp_dir / "test_max_size.tsv"
    args = f"-s {h5_seq_path} -o {outpath} -z 3 -zp {max_size}".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x <= max_size)


@pytest.mark.parametrize("size", (5, 7))
def test_min_eq_max(runner, tmp_dir, h5_seq_path, size):
    outpath = tmp_dir / "test_min_eq_max.tsv"
    args = f"-s {h5_seq_path} -o {outpath} -z {size} -zp {size}".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath, eval_rows=lambda x: x == size)


def test_min_gt_max_fail(runner, tmp_dir, h5_seq_path):
    outpath = tmp_dir / "test_min_gt_max_fail.tsv"
    args = f"-s {h5_seq_path} -o {outpath} -zp 2".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code != 0, r.output


@pytest.mark.parametrize("stat", ("total_jsd", "mean_jsd", "mean_delta_jsd"))
def test_stat(runner, tmp_dir, h5_seq_path, stat):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {h5_seq_path} -o {outpath} -st {stat}".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    _checked_output(outpath, eval_header=lambda x: stat in x)


@pytest.mark.parametrize("k", (1, 3, 7))
def test_k(runner, tmp_dir, h5_seq_path, k):
    outpath = tmp_dir / "test_defaults.tsv"
    args = f"-s {h5_seq_path} -o {outpath} -k {k}".split()
    r = runner.invoke(dvgt_max, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_output(outpath)


def test_prep_defaults(runner, tmp_dir, fasta_seq_path):
    outpath = tmp_dir / "test_prep_defaults.h5"
    args = f"-s {fasta_seq_path} -o {outpath}".split()
    r = runner.invoke(dvgt_prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_h5_output(str(outpath))


def test_prep_outpath_without_suffix(runner, tmp_dir, fasta_seq_path):
    outpath = tmp_dir / "test_prep_outpath_without_suffix"
    args = f"-s {fasta_seq_path} -o {outpath}".split()
    r = runner.invoke(dvgt_prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    _checked_h5_output(str(outpath) + ".h5")


def test_prep_force_override(runner, tmp_dir, fasta_seq_path):
    outpath = tmp_dir / "test_prep_force_override.h5"
    args = f"-s {fasta_seq_path} -o {outpath}".split()

    # Run prep once, it should succeed
    r = runner.invoke(dvgt_prep, args)
    assert r.exit_code == 0, r.output
    _checked_h5_output(str(outpath))

    # Run prep again without force overwrite, it should fail with a FileExistsError
    r = runner.invoke(dvgt_prep, args)
    assert r.exit_code != 0, r.output
    assert "FileExistsError" in r.output
    _checked_h5_output(str(outpath))

    # with the force write flag, it should succeed
    args += ["-F"]
    r = runner.invoke(dvgt_prep, args)
    assert r.exit_code == 0, r.output
    _checked_h5_output(str(outpath))


@pytest.mark.parametrize("moltype", ["dna", "rna"])
def test_prep_max_moltype(runner, tmp_dir, fasta_seq_path, moltype):
    outpath = tmp_dir / f"test_prep_{moltype}.h5"
    args = f"-s {fasta_seq_path} -o {outpath} -m {moltype}".split()
    r = runner.invoke(dvgt_prep, args)
    assert r.exit_code == 0, r.output
    _checked_h5_output(str(outpath))

    max_outpath = tmp_dir / f"test_prep_{moltype}.tsv"
    max_args = f"-s {outpath} -o {max_outpath}".split()
    r = runner.invoke(dvgt_max, max_args)
    assert r.exit_code == 0, r.output  
    _checked_output(str(max_outpath))
