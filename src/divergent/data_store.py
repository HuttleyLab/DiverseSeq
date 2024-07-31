from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin
from cogent3.app.data_store import (DataMember, DataStoreABC, Mode, NoneType,
                                    StrOrBytes)
from numpy import empty, ndarray, uint8
from scitrack import get_text_hexdigest

_NOT_COMPLETED_TABLE = "not_completed"
_LOG_TABLE = "logs"
_MD5_TABLE = "md5"


class HDF5DataStore(DataStoreABC):
    """Stores array data in HDF5 data sets. Associated information is
    stored as attributed on data sets.
    """

    def __init__(self, source: str | Path, mode: Mode = "w", limit: int = None) -> None:
        self._source = Path(source)
        self._mode = Mode(mode)
        self._limit = limit

    @property
    def source(self) -> Path:
        """string that references connecting to data store"""
        return self._source

    @property
    def mode(self) -> Mode:
        """string that references datastore mode, override in subclass constructor"""
        return self._mode

    @property
    def limit(self):
        return self._limit

    def read(self, unique_id: str) -> ndarray:
        """reads and return array data corresponding to identifier"""
        with h5py.File(self._source, mode="r") as f:
            # TODO: this will fail if the unique_id is not in the file
            data = f[unique_id]
            out = empty(len(data), dtype=uint8)
            data.read_direct(out)
        return out

    def get_attrs(self, unique_id: str) -> dict:
        """return all data set attributes connected to an identifier"""
        with h5py.File(self._source, mode="r") as f:
            data = f[unique_id]
            attrs = {key: value for key, value in data.attrs.items()}
        return attrs

    def _write(
        self,
        *,
        subdir: str,
        unique_id: str,
        data: ndarray,
        **kwargs,
    ) -> DataMember:
        with h5py.File(self._source, mode="a") as f:
            path = f"{subdir}/{unique_id}" if subdir else unique_id
            dset = f.create_dataset(path, data=data, dtype="u1", **hdf5plugin.Blosc2())

            if subdir == _LOG_TABLE:
                return None

            if subdir == _NOT_COMPLETED_TABLE:
                member = DataMember(
                    data_store=self,
                    unique_id=Path(_NOT_COMPLETED_TABLE) / unique_id,
                )

            elif not subdir:
                member = DataMember(data_store=self, unique_id=unique_id)
                for key, value in kwargs.items():
                    dset.attrs[key] = value

            md5 = get_text_hexdigest(data.tobytes())
            md5_dtype = h5py.string_dtype()
            f.create_dataset(f"{_MD5_TABLE}/{unique_id}", data=md5, dtype=md5_dtype)
        return member

    def write(self, *, unique_id: str, data: ndarray, **kwargs) -> DataMember:
        """Writes a completed record to a dataset in the HDF5 file.

        Parameters
        ----------
        unique_id
            The unique identifier for the record. This will be used
            as the name of the dataset in the HDF5 file.
        data
            The data to be stored in the record.
        kwargs
            Any additional keyword arguments will be stored as attributes on the dataset.
            Each key-value pair in kwargs corresponds to an attribute name and its value.

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

    def drop_not_completed(self, *, unique_id: Optional[str] = None) -> None: ...

    def md5(self, unique_id: str) -> Union[str, NoneType]:
        with h5py.File(self._source, mode="a") as f:
            if f"md5/{unique_id}" in f:
                dset = f[f"md5/{unique_id}"]
                md5 = dset[()].decode("utf-8")
            else:
                md5 = None
        return md5

    @property
    def completed(self) -> list[DataMember]:
        if not self._completed:
            self._completed = []
            with h5py.File(self._source, mode="a") as f:
                for name, item in f.items():
                    if isinstance(item, h5py.Dataset):
                        self._completed.append(
                            DataMember(data_store=self, unique_id=name),
                        )

        return self._completed

    @property
    def logs(self) -> list[DataMember]:
        return []

    @property
    def not_completed(self) -> list[DataMember]:
        return []
