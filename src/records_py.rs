use crate::record::SeqRecord;
use crate::records::{SummedRecords, make_summed_records};
use pyo3::prelude::{PyErr, PyResult, pyclass, pymethods};

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
    k: usize,
    num_states: usize,
}

#[pymethods]
impl SummedRecordsWrapper {
    #[new]
    #[pyo3(signature = (records, k, num_states=4))]
    pub fn new(records: Vec<(String, Vec<u8>)>, k: usize, num_states: usize) -> Self {
        let summed_records = make_summed_records(records, k, num_states);

        Self {
            summed_records: summed_records,
            k: k,
            num_states: num_states,
        }
    }

    pub fn delta_jsd(&self, seqid: &str, seq: Vec<u8>) -> PyResult<f64> {
        let seqrec = SeqRecord::new(seqid, &seq, self.num_states as usize);
        let kseq = seqrec.to_kmerseq(self.k).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "delta_jsd('{}') failed: {}",
                seqid, e
            ))
        })?;
        Ok(self.summed_records.delta_jsd(&kseq))
    }

    pub fn get_result(&self) -> PyResult<SummedRecordsResult> {
        Ok(self.summed_records.get_result())
    }
}
