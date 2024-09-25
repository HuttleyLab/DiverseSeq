import functools
import inspect
import io
import pathlib
from collections.abc import Sequence

import h5py
import hdf5plugin
import numpy
import numpy.typing
from cogent3.app.data_store import DataMember, DataStoreABC, Mode, StrOrBytes
from scitrack import get_text_hexdigest

_NOT_COMPLETED_TABLE = "not_completed"
_LOG_TABLE = "logs"
_MD5_TABLE = "md5"
_H5_STRING_DTYPE = h5py.string_dtype()


_HDF5_BLOSC2_KWARGS = hdf5plugin.Blosc2(
    cname="blosclz",
    clevel=3,
    filters=hdf5plugin.Blosc2.BITSHUFFLE,
)


class HF5FileWrapper:
    """virtualizes a HDF5 file so that it can be either on disk or in memory
    behaving consistently as a context manager"""

    def __init__(
        self,
        *,
        source: str | pathlib.Path,
        mode: Mode = "r",
        in_memory: bool = False,
    ):
        self.in_memory = in_memory
        mode = "w" if in_memory else mode
        self.mode = Mode(mode)
        self.source = pathlib.Path(source)
        if self.mode == Mode.r and not self.source.exists():
            raise OSError(f"{self.source!s} not found")
        if in_memory:
            mode = "w"
            in_memory = io.BytesIO()

        self._file = None
        self._init_file(in_memory)

    @functools.singledispatchmethod
    def _init_file(self, in_memory: io.BytesIO) -> None:
        self._file = h5py.File(in_memory, mode=self.mode.name)
        self._file.create_group(_NOT_COMPLETED_TABLE)
        self._file.create_group(_LOG_TABLE)
        self._file.create_group(_MD5_TABLE)
        self._file.attrs["source"] = str(self.source)

    @_init_file.register
    def _(self, in_memory: bool) -> None:
        if self.mode == Mode.w:
            with h5py.File(self.source, mode="w") as f:
                f.create_group(_NOT_COMPLETED_TABLE)
                f.create_group(_LOG_TABLE)
                f.create_group(_MD5_TABLE)
                f.attrs["source"] = str(self.source)
            self.mode = Mode.a

    def __enter__(self):
        if not self.in_memory:
            self._file = h5py.File(self.source, mode=self.mode.name)
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.in_memory:
            self._file.close()
        return False

    def close(self):
        if self.in_memory:
            self._file.close()


class HDF5DataStore(DataStoreABC):
    """Stores array data in HDF5 data sets. Associated information is
    stored as attributed on data sets.
    """

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        init_sig = inspect.signature(cls.__init__)
        bargs = init_sig.bind_partial(cls, *args, **kwargs)
        bargs.apply_defaults()
        init_vals = bargs.arguments
        init_vals.pop("self", None)
        obj._init_vals = init_vals
        return obj

    def __init__(
        self,
        source: str | pathlib.Path,
        mode: Mode = "r",
        limit: int = None,
        in_memory: bool = False,
    ) -> None:
        self._hf5_file = HF5FileWrapper(source=source, mode=mode, in_memory=in_memory)
        self._source = pathlib.Path(source)
        self._mode = self._hf5_file.mode
        self._limit = limit
        self._completed = []

    def __getstate__(self):
        init_vals = self._init_vals.copy()
        init_vals["mode"] = "w" if init_vals["in_memory"] else "r"
        return init_vals

    def __setstate__(self, state):
        obj = self.__class__(**state)
        self.__dict__.update(obj.__dict__)

    @property
    def source(self) -> pathlib.Path:
        """string that references connecting to data store"""
        return self._hf5_file.source

    @property
    def mode(self) -> Mode:
        """string that references datastore mode, override in subclass constructor"""
        return self._mode

    @property
    def limit(self):
        return self._limit

    def read(self, unique_id: str) -> numpy.ndarray:
        """reads and return array data corresponding to identifier"""
        with self._hf5_file as f:
            data = f[unique_id]
            out = numpy.empty(len(data), dtype=numpy.uint8)
            data.read_direct(out)
            return out

    def get_attrs(self, unique_id: str) -> dict:
        """return all data set attributes connected to an identifier"""
        with self._hf5_file as f:
            data = f[unique_id]

        return dict(data.attrs.items())

    def _write(
        self,
        *,
        subdir: str,
        unique_id: str,
        data: numpy.ndarray,
        **kwargs,
    ) -> DataMember:
        if subdir == _LOG_TABLE:
            return None

        with self._hf5_file as f:
            path = f"{subdir}/{unique_id}" if subdir else unique_id
            dset = f.create_dataset(path, data=data, dtype="u1", **_HDF5_BLOSC2_KWARGS)

            if subdir == _NOT_COMPLETED_TABLE:
                member = DataMember(
                    data_store=self,
                    unique_id=pathlib.Path(_NOT_COMPLETED_TABLE) / unique_id,
                )

            elif not subdir:
                member = DataMember(data_store=self, unique_id=unique_id)
                for key, value in kwargs.items():
                    dset.attrs[key] = value

            md5 = get_text_hexdigest(data.tobytes())
            f.create_dataset(
                f"{_MD5_TABLE}/{unique_id}",
                data=md5,
                dtype=_H5_STRING_DTYPE,
            )
            del data

        return member

    def write(self, *, unique_id: str, data: numpy.ndarray, **kwargs) -> DataMember:
        """Writes a completed record to a dataset in the HDF5 file.

        Parameters
        ----------
        unique_id
            The unique identifier for the record. This will be used
            as the name of the dataset in the HDF5 file.
        data
            The data to be stored in the record.
        kwargs
            Any additional keyword arguments will be stored as attributes on the
            dataset. Each key-value pair in kwargs corresponds to an attribute
            name and its value.

        Returns
        -------
        DataMember
            A DataMember object representing the new record.

        Notes
        -----
        Drops any not-completed member corresponding to this identifier
        """
        member = self._write(subdir="", unique_id=unique_id, data=data, **kwargs)
        self.drop_not_completed(unique_id=unique_id)
        if member is not None:
            self._completed.append(member)
        return member

    def write_not_completed(self, *, unique_id: str, data: StrOrBytes) -> None: ...

    def write_log(self, *, unique_id: str, data: StrOrBytes) -> None: ...

    def drop_not_completed(self, *, unique_id: str | None = None) -> None: ...

    def md5(self, unique_id: str) -> str | None:
        with self._hf5_file as f:
            if f"{_MD5_TABLE}/{unique_id}" in f:
                dset = f[f"{_MD5_TABLE}/{unique_id}"]
                return dset[()].decode("utf-8")
        return None

    @property
    def completed(self) -> list[DataMember]:
        if not self._completed:
            r = []
            with self._hf5_file as f:
                for name in f.keys():
                    if name in (_LOG_TABLE, _NOT_COMPLETED_TABLE, _MD5_TABLE):
                        continue

                    m = DataMember(data_store=self, unique_id=name)
                    r.append(m)
            self._completed = r
        return self._completed

    @property
    def logs(self) -> list[DataMember]:
        return []

    @property
    def not_completed(self) -> list[DataMember]:
        return []

    def __del__(self):
        self.close()

    def close(self):
        """closes the hdf5 file"""
        # hdf5 dumps content to stdout if resource already closed, so
        # we trap that here, and capture expected exceptions raised in the
        # process


def get_seqids_from_store(
    seq_store: str | pathlib.Path,
) -> list[str]:
    """return the list of seqids in a sequence store"""
    dstore = HDF5DataStore(seq_store, mode="r")
    return [m.unique_id for m in dstore.completed]


def get_ordered_records(
    seq_store: HDF5DataStore,
    seq_names: Sequence[str],
) -> list[DataMember]:
    """Returns ordered data store records given a seqeunce
    of sequence names.

    Parameters
    ----------
    seq_store
        The data store to load from.
    seq_names
        The ordered sequence names.

    Returns
    -------
    list[DataMember]
        Ordered data member records.
    """
    records = {m.unique_id: m for m in seq_store.completed if m.unique_id in seq_names}
    return [records[name] for name in seq_names]
