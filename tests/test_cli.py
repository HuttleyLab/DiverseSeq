import pathlib

import pytest

from click.testing import CliRunner
from cogent3 import load_table

from divergent.cli import max as dvgt_max


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
