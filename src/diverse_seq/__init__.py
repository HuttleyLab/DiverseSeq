"""diverse_seq: a tool for sampling diverse biological sequences"""

import typing

# need to import hdf5plugin here to make sure the plugin path can be
# found by h5py
import hdf5plugin  # noqa: F401

if typing.TYPE_CHECKING:
    from cogent3.core.alignment import SequenceCollection

__version__ = "2025.12.17"


def load_sample_data() -> "SequenceCollection":
    """load sample data"""
    from cogent3 import load_aligned_seqs  # noqa: PLC0415

    from .util import get_sample_data_path  # noqa: PLC0415

    path = get_sample_data_path()
    return load_aligned_seqs(path, moltype="dna").degap()
