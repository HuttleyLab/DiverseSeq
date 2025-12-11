import nox

_py_versions = range(11, 15)

nox.options.sessions = ["test", "testcov"]


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install("-e.[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        *session.posargs,  # propagates sys.argv to pytest
    )


@nox.session(python=["3.13"])
def testcov(session):
    session.install(".[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "--cov-report",
        "html",
        "--cov",
        "diverse_seq",
    )


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test_coveralls(session):
    session.install("-e.[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        *session.posargs,  # propagates sys.argv to pytest
    )
