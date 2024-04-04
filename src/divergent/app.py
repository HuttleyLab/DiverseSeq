from pathlib import Path
from typing import Optional, Union

from cogent3.app import typing as c3_types
from cogent3.app.composable import define_app
from cogent3.app.data_store import (
    READONLY,
    DataStoreABC,
    DataStoreDirectory,
    Mode,
)
from cogent3.format.fasta import alignment_to_fasta

from divergent.loader import _label_func, faster_load_fasta


@define_app
class dvgt_seq_file_to_data_store:
    def __init__(
        self,
        dest: Optional[c3_types.IdentifierType] = None,
        limit: Optional[int] = None,
        mode: Union[str, Mode] = READONLY,  # should we even give the option?
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
