from typing import Optional

from cogent3.app import typing as c3_types
from cogent3.app.composable import WRITER, define_app
from cogent3.app.data_store import get_unique_id

from divergent.data_store import HDF5DataStore
from divergent.record import SeqArray


@define_app(app_type=WRITER)
class dvgt_write_prepped_seqs:
    """Write preprocessed seqs to a dvgtseq datastore"""
    def __init__(
        self,
        dest: c3_types.IdentifierType,
        limit: int = None,
        id_from_source: callable = get_unique_id,
    ):
        self.dest = dest
        self.data_store = HDF5DataStore(self.dest, limit=limit)
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
