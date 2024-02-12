import pathlib

import nox


_py_versions = range(10, 13)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install(".[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
    )


@nox.session(python=["3.10"])
def testcov(session):
    session.install(".[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "--cov-report",
        "html",
        "--cov",
        "divergent",
    )
