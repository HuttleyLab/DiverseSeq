import os
import subprocess
from pathlib import Path

import nox

PROJ_ROOT = Path(__file__).parent

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


def cleanup_profraw_files(suffix):
    for profraw in PROJ_ROOT.glob(f"*.{suffix}"):
        profraw.unlink()
    for profraw in (PROJ_ROOT / "tests").glob(f"*.{suffix}"):
        profraw.unlink()


def find_llvm_bin(sysroot):
    # Recursively search for llvm-profdata
    for llvm_profdata in sysroot.rglob("llvm-profdata"):
        if llvm_profdata.is_file():
            return llvm_profdata.parent

    msg = f"llvm-profdata not found in {sysroot}"
    raise OSError(msg)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def rustcov(session):
    """Generate coverage for Rust code triggered by Python tests."""
    session.install("-e.[test]")
    output_dir = Path("coverage-report")

    # Determine LLVM bin path using rustc sysroot
    sysroot = Path(
        subprocess.run(
            ["rustc", "--print", "sysroot"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    llvm_bin = find_llvm_bin(sysroot)

    # Build the extension with coverage instrumentation
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C instrument-coverage"
    # Unset conda/venv conflicts
    env.pop("CONDA_PREFIX", None)
    session.run("maturin", "develop", env=env)

    # Clean up old profraw files
    cleanup_profraw_files("profdata")
    cleanup_profraw_files("profraw")

    # Run pytest with profiling enabled
    env["LLVM_PROFILE_FILE"] = str(PROJ_ROOT / "coverage-%p-%m.profraw")
    session.chdir("tests")
    session.run(
        "pytest",
        "--cov-report",
        "lcov:python.lcov",
        "--cov",
        "diverse_seq",
        env=env,
        external=True,
    )
    session.chdir("..")

    # Find the built extension
    extension_path = None
    for so_file in PROJ_ROOT.rglob("_dvs*.so"):
        extension_path = so_file
        break

    if not extension_path:
        session.error("Could not find built extension _dvs*.so")

    if profraw_files := list(PROJ_ROOT.glob("coverage-*.profraw")):
        session.run(
            str(llvm_bin / "llvm-profdata"),
            "merge",
            "-sparse",
            *[str(f) for f in profraw_files],
            "-o",
            "coverage.profdata",
            external=True,
        )
    else:
        session.error("No coverage-*.profraw files found")

    rep_args = [
        str(llvm_bin / "llvm-cov"),
        "show",
        "-instr-profile=coverage.profdata",
        str(extension_path),
        f"-output-dir={output_dir}",
        "-format=html",
        "-ignore-filename-regex=\\.cargo|.rustup|registry",
        "-show-instantiations=false",
        "src",
    ]

    # Generate html coverage report
    session.run(*rep_args, external=True)

    # Generate lcov coverage report
    lcov_cmd = (
        f"{llvm_bin / 'llvm-cov'} export "
        f"-instr-profile=coverage.profdata "
        f"{extension_path} "
        f"-format=lcov "
        f"-ignore-filename-regex='\\.cargo|.rustup|registry' "
        f"src > {output_dir / 'rust.lcov'}"
    )
    session.run("bash", "-c", lcov_cmd, external=True)

    # Clean up profraw files
    cleanup_profraw_files("profdata")
    cleanup_profraw_files("profraw")

    # Merge Python and Rust coverage files
    python_lcov = PROJ_ROOT / "tests" / "python.lcov"
    rust_lcov = output_dir / "rust.lcov"

    if python_lcov.exists() and rust_lcov.exists():
        session.run(
            "lcov",
            "--ignore-errors",
            "inconsistent",
            "--add-tracefile",
            str(python_lcov),
            "--add-tracefile",
            str(rust_lcov),
            "-o",
            "coverage.lcov",
            external=True,
        )
        session.log("Merged Python and Rust coverage into coverage.lcov")

        html_report_dir = "coverage-html"
        session.run(
            "genhtml",
            "--ignore-errors",
            "category",
            "coverage.lcov",
            "-o",
            str(html_report_dir),
            "--branch-coverage",
            external=True,
        )
        session.log(f"Merged HTML coverage report generated in {html_report_dir}/")
