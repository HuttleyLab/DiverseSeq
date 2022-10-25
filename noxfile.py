import pathlib

import nox


_py_versions = range(8, 11)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    py_version = session.python.replace(".", "")
    session.install(".[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        "--junitxml",
        f"junit-{py_version}.xml",
        "--cov-report",
        "xml",
        "--cov",
        "divergent",
    )


@nox.session(python=["3.10"])
def testcov(session):
    session.install(*dependencies)
    session.install(".")
    session.chdir("tests")
    session.run(
        "pytest",
        "--cov-report",
        "html",
        "--cov",
        "divergent",
    )
