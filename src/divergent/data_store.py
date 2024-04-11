from pathlib import Path
from typing import Optional, Union

import h5py
import hdf5plugin

from cogent3.app import typing as c3_types
from cogent3.app.composable import define_app
from cogent3.app.data_store import (
    OVERWRITE,
    DataMember,
    DataStoreABC,
    DataStoreDirectory,
    Mode,
    NoneType,
    StrOrBytes,
)
from cogent3.format.fasta import alignment_to_fasta
from numpy import empty, ndarray, uint8
from scitrack import get_text_hexdigest

from divergent.loader import _label_func, faster_load_fasta


_NOT_COMPLETED_TABLE = "not_completed"
_LOG_TABLE = "logs"
_MD5_TABLE = "md5"


@define_app
class dvgt_seq_file_to_data_store:
    def __init__(
        self,
        dest: Optional[c3_types.IdentifierType] = None,
        limit: Optional[int] = None,
        mode: Union[str, Mode] = OVERWRITE,
    ):
        self.dest = dest
        self.limit = limit
        self.mode = mode
        self.loader = faster_load_fasta(label_func=_label_func)

    def main(self, fasta_path: c3_types.IdentifierType) -> DataStoreABC:
        outpath = Path(self.dest) if self.dest else Path(fasta_path).with_suffix("")
        outpath.mkdir(parents=True, exist_ok=True)
        out_dstore = DataStoreDirectory(source=outpath, mode=self.mode, suffix=".fa")

        seqs = self.loader(fasta_path)

        for seq_id, seq_data in seqs.items():
            fasta_seq_data = alignment_to_fasta({seq_id: seq_data}, block_size=80)
            out_dstore.write(unique_id=seq_id, data=fasta_seq_data)

        return out_dstore


class HDF5DataStore(DataStoreABC):
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
        """reads data corresponding to identifier"""
        with h5py.File(self._source, mode="r") as f:
            # todo: this will fail if the unique_id is not in the file
            data = f[unique_id]
            out = empty(len(data), dtype=uint8)
            data.read_direct(out)
            return out

    def _write(
        self, *, subdir: str, unique_id: str, data: ndarray, moltype: str, source: str
    ) -> DataMember:
        with h5py.File(self._source, mode="a") as f:
            path = f"{subdir}/{unique_id}" if subdir else unique_id
            dset = f.create_dataset(path, data=data, dtype="u1", **hdf5plugin.Blosc2())

            if subdir == _LOG_TABLE:
                return None

            if subdir == _NOT_COMPLETED_TABLE:
                member = DataMember(
                    data_store=self, unique_id=Path(_NOT_COMPLETED_TABLE) / unique_id
                )

            elif not subdir:
                member = DataMember(data_store=self, unique_id=unique_id)
                dset.attrs["moltype"] = moltype
                dset.attrs["source"] = source

            md5 = get_text_hexdigest(data.tobytes())
            md5_dtype = h5py.string_dtype()
            f.create_dataset(f"{_MD5_TABLE}/{unique_id}", data=md5, dtype=md5_dtype)
            return member

    def write(
        self, *, unique_id: str, data: ndarray, moltype: str, source: str
    ) -> DataMember:
        """writes a completed record

        Parameters
        ----------
        unique_id
            unique identifier
        data
            seq data

        Returns
        -------
        a member for this record

        Notes
        -----
        Drops any not-completed member corresponding to this identifier
        """
        member = self._write(
            subdir="", unique_id=unique_id, data=data, moltype=moltype, source=source
        )
        self.drop_not_completed(unique_id=unique_id)
        if member is not None:
            self._completed.append(member)
        return member

    def write_not_completed(self, *, unique_id: str, data: StrOrBytes) -> None:
        ...

    def write_log(self, *, unique_id: str, data: StrOrBytes) -> None:
        ...

    def drop_not_completed(self, *, unique_id: Optional[str] = None) -> None:
        ...

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
                            DataMember(data_store=self, unique_id=name)
                        )

        return self._completed

    @property
    def logs(self) -> list[DataMember]:
        return []

    @property
    def not_completed(self) -> list[DataMember]:
        return []
