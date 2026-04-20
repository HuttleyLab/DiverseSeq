"""diverse_seq: a tool for sampling diverse biological sequences"""

import typing
import warnings

import scinexus

warnings.filterwarnings("ignore", message='.+Cannot register "UNREGISTER".+')

scinexus.set_parallel_backend("multiprocess")

if typing.TYPE_CHECKING:
    from cogent3.core.alignment import SequenceCollection

# change version in Cargo.toml as well
__version__ = "2026.4.20"


def load_sample_data() -> "SequenceCollection":
    """load sample data"""
    from cogent3 import load_aligned_seqs  # noqa: PLC0415

    from .util import get_sample_data_path  # noqa: PLC0415

    path = get_sample_data_path()
    return load_aligned_seqs(path, moltype="dna").degap()
