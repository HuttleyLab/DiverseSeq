use crate::record::{KmerSeq, SeqRecord};
use crate::records::SummedRecords;
use crate::zarr_io::{Storage, ZarrStore};
use pyo3::Python;
use pyo3::prelude::{Bound, PyErr, PyResult, pyclass, pymethods};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyTuple};
use rustc_hash::FxHashMap;
use std::path::PathBuf;

#[pyclass(module = "diverse_seq._dvs")]
pub struct SummedRecordsResult {
    #[pyo3(get)]
    pub total_jsd: f64,
    #[pyo3(get)]
    pub record_deltas: Vec<(String, f64)>,
    #[pyo3(get)]
    pub mean_delta_jsd: f64,
    #[pyo3(get)]
    pub std_delta_jsd: f64,
    #[pyo3(get)]
    pub cov_delta_jsd: f64,
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub k: usize,
    #[pyo3(get)]
    pub num_states: usize,
}

#[pymethods]
impl SummedRecordsResult {
    #[getter]
    pub fn record_names(&self) -> Vec<String> {
        self.record_deltas.iter().map(|r| r.0.to_string()).collect()
    }
}

#[pyclass(module = "diverse_seq._dvs", unsendable)]
pub struct SummedRecordsWrapper {
    pub summed_records: SummedRecords,
    num_states: usize,
    k: usize,
}

#[pymethods]
impl SummedRecordsWrapper {
    #[new]
    #[pyo3(signature = (records, k, num_states=4))]
    pub fn new(records: Vec<(String, Vec<u8>)>, k: usize, num_states: usize) -> Self {
        let mut kseq_recs: Vec<KmerSeq> = Vec::new();

        for (seqid, seq) in records.iter() {
            let seqrec = SeqRecord::new(seqid, seq, num_states);
            let kseq = seqrec.to_kmerseq(k);
            if kseq.is_ok() {
                kseq_recs.push(kseq.unwrap());
            }
        }

        Self {
            summed_records: SummedRecords::new(kseq_recs),
            num_states: num_states,
            k: k,
        }
    }

    pub fn delta_jsd(&self, seqid: &str, seq: Vec<u8>) -> PyResult<f64> {
        let seqrec = SeqRecord::new(seqid, &seq, self.num_states as usize);
        let kseq = seqrec.to_kmerseq(self.k);
        if kseq.is_ok() {
            let kseq = kseq.unwrap();
            Ok(self.summed_records.delta_jsd(&kseq))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create add {}",
                seqid
            )))
        }
    }
}
