from typing import Optional

from cogent3.app import typing as c3_types
from cogent3.app.composable import WRITER, define_app
from cogent3.app.data_store import get_unique_id

from divergent.data_store import HDF5DataStore
from divergent.record import SeqArray, SeqRecord


@define_app(app_type=WRITER)
class dvgt_write_prepped_seqs:
    def __init__(
        self,
        data_store: HDF5DataStore,
        id_from_source: callable = get_unique_id,
    ):
        self.data_store = data_store
        self.id_from_source = id_from_source

    def main(
        self, data: SeqArray, identifier: Optional[str] = None
    ) -> c3_types.IdentifierType:
        unique_id = identifier or self.id_from_source(data.unique_id)
        return self.data_store.write(
            unique_id=unique_id,
            data=data.data,
            moltype=data.moltype,
            source=str(data.source),
        )


@define_app(app_type=WRITER)
class dvgt_write_record:
    def __init__(
        self,
        data_store: HDF5DataStore,
    ):
        self.data_store = data_store

    def main(
        self, data: SeqRecord, identifier: Optional[str] = None
    ) -> c3_types.IdentifierType:
        unique_id = identifier or data.name
        return self.data_store.write(
            unique_id=unique_id,
            data=data.kcounts.data,
            length=data.length,
        )
