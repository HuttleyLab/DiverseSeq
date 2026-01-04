import os
import platform
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


@nox.session(python=[f"3.{v}" for v in _py_versions])
def rustcov(session):
    """Generate coverage for Rust code triggered by Python tests."""
    output_dir = Path("coverage-report")
    # Determine LLVM bin path based on platform
    toolchain = (
        subprocess.run(
            ["rustup", "show", "active-toolchain"],
            check=False,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split()[0]
    )

    if platform.system() == "Darwin":
        machine = (
            "aarch64-apple-darwin"
            if platform.machine() == "arm64"
            else "x86_64-apple-darwin"
        )
    else:
        machine = platform.machine()

    llvm_bin = (
        Path.home()
        / ".rustup"
        / "toolchains"
        / toolchain
        / "lib"
        / "rustlib"
        / machine
        / "bin"
    )

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
    session.run("pytest", env=env)
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
    session.run(*rep_args)

    # Generate lcov coverage report
    lcov_cmd = (
        f"{llvm_bin / 'llvm-cov'} export "
        f"-instr-profile=coverage.profdata "
        f"{extension_path} "
        f"-format=lcov "
        f"-ignore-filename-regex='\\.cargo|.rustup|registry' "
        f"src > {output_dir / 'rust.lcov'}"
    )
    session.run("bash", "-c", lcov_cmd)

    # Clean up profraw files
    cleanup_profraw_files("profdata")
    cleanup_profraw_files("profraw")

    session.log("Coverage report generated in coverage-report-rust/")
