from typing import Optional

from cogent3.app import typing as c3_types
from cogent3.app.composable import WRITER, define_app
from cogent3.app.data_store import DataStoreABC, get_unique_id

from divergent.record import SeqArray


@define_app(app_type=WRITER)
class dvgt_write_prepped_seqs:
    def __init__(
        self,
        data_store: DataStoreABC,
        id_from_source: callable = get_unique_id,
        moltype: str = None,
    ):
        self.data_store = data_store
        self.id_from_source = id_from_source
        self.moltype = moltype

    def main(
        self, data: SeqArray, identifier: Optional[str] = None
    ) -> c3_types.IdentifierType:
        return self.data_store.write(unique_id=identifier, data=data.data)
