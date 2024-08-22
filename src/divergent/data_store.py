import contextlib
import inspect
import os
import pathlib

import h5py
import hdf5plugin
import numpy
import numpy.typing
from cogent3.app.data_store import DataMember, DataStoreABC, Mode, StrOrBytes
from scitrack import get_text_hexdigest

_NOT_COMPLETED_TABLE = "not_completed"
_LOG_TABLE = "logs"
_MD5_TABLE = "md5"


_HDF5_BLOSC2_KWARGS = hdf5plugin.Blosc2(
    cname="blosclz",
    clevel=3,
    filters=hdf5plugin.Blosc2.BITSHUFFLE,
)


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
        if in_memory:
            h5_kwargs = dict(
                driver="core",
                backing_store=False,
            )
            source = "memory"
            mode = "w"
        else:
            h5_kwargs = {}

        self._source = pathlib.Path(source)

        self._mode = Mode(mode)
        if self._mode == Mode.r and not self._source.exists():
            raise OSError(f"{self._source!s} not found")
        self._limit = limit
        self._h5_kwargs = h5_kwargs
        self._file = h5py.File(source, mode=self.mode.name, **self._h5_kwargs)
        self._is_open = True
        self._completed = []

    def __getstate__(self):
        init_vals = self._init_vals.copy()
        init_vals["mode"] = "w" if init_vals["in_memory"] else "r"
        return init_vals

    def __setstate__(self, state):
        obj = self.__class__(**state)
        self.__dict__.update(obj.__dict__)
        # because we have a __del__ method, and self attributes point to
        # attributes on obj, we need to modify obj state so that garbage
        # collection does not screw up self
        obj._is_open = False
        obj._file = None

    @property
    def source(self) -> pathlib.Path:
        """string that references connecting to data store"""
        return self._source

    @property
    def mode(self) -> Mode:
        """string that references datastore mode, override in subclass constructor"""
        return self._mode

    @property
    def limit(self):
        return self._limit

    def read(self, unique_id: str) -> numpy.ndarray:
        """reads and return array data corresponding to identifier"""
        data = self._file[unique_id]
        out = numpy.empty(len(data), dtype=numpy.uint8)
        data.read_direct(out)
        return out

    def get_attrs(self, unique_id: str) -> dict:
        """return all data set attributes connected to an identifier"""
        data = self._file[unique_id]
        return dict(data.attrs.items())

    def _write(
        self,
        *,
        subdir: str,
        unique_id: str,
        data: numpy.ndarray,
        **kwargs,
    ) -> DataMember:
        f = self._file
        path = f"{subdir}/{unique_id}" if subdir else unique_id
        dset = f.create_dataset(path, data=data, dtype="u1", **_HDF5_BLOSC2_KWARGS)

        if subdir == _LOG_TABLE:
            return None

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
        md5_dtype = h5py.string_dtype()
        f.create_dataset(f"{_MD5_TABLE}/{unique_id}", data=md5, dtype=md5_dtype)
        f.flush()
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
        f = self._file
        if f"md5/{unique_id}" in f:
            dset = f[f"md5/{unique_id}"]
            return dset[()].decode("utf-8")
        return None

    @property
    def completed(self) -> list[DataMember]:
        if not self._completed:
            self._completed = [
                DataMember(data_store=self, unique_id=name)
                for name in self._file.keys()
                if name not in (_LOG_TABLE, _NOT_COMPLETED_TABLE, _MD5_TABLE)
            ]
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
        try:
            open
        except NameError:
            # builtin open() already garbage collected, so nothing to do
            return
        with open(os.devnull, "w") as devnull:
            with (
                contextlib.redirect_stderr(devnull),
                contextlib.redirect_stdout(devnull),
            ):
                with contextlib.suppress(
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    PermissionError,
                ):
                    if self._is_open:
                        self._file.flush()

                with contextlib.suppress(AttributeError):
                    self._file.close()


def get_seqids_from_store(
    seq_store: str | pathlib.Path,
) -> list[str]:
    """return the list of seqids in a sequence store"""
    dstore = HDF5DataStore(seq_store, mode="r")
    return [m.unique_id for m in dstore.completed]
