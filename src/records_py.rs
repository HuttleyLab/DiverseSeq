use crate::record::SeqRecord;
use crate::records::{SummedRecords, make_summed_records};
use pyo3::Python;
use pyo3::prelude::{Bound, PyErr, PyResult, pyclass, pymethods};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};

#[pyclass(module = "diverse_seq._dvs")]
#[derive(Clone)]
pub struct SummedRecordsResult {
    #[pyo3(get)]
    pub total_jsd: f64,
    #[pyo3(get)]
    pub records: Vec<(String, Vec<f64>, f64)>,
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
    #[new]
    pub fn new() -> Self {
        Self {
            total_jsd: 0.0,
            records: Vec::new(),
            mean_delta_jsd: 0.0,
            std_delta_jsd: 0.0,
            cov_delta_jsd: 0.0,
            size: 0,
            k: 0,
            num_states: 0,
        }
    }

    #[getter]
    pub fn record_names(&self) -> Vec<String> {
        self.records.iter().map(|r| r.0.to_string()).collect()
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = PyDict::new(py);
        macro_rules! set_field {
            ($key:expr, $value:expr) => {
                state.set_item($key, $value)?
            };
        }

        set_field!("total_jsd", self.total_jsd);
        set_field!("records", self.records.clone());
        set_field!("mean_delta_jsd", self.mean_delta_jsd);
        set_field!("std_delta_jsd", self.std_delta_jsd);
        set_field!("cov_delta_jsd", self.cov_delta_jsd);
        set_field!("size", self.size);
        set_field!("k", self.k);
        set_field!("num_states", self.num_states);
        Ok(state)
    }

    fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        macro_rules! get_field {
            ($key:expr) => {
                state
                    .get_item($key)?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>($key))?
                    .extract()?
            };
        }

        self.total_jsd = get_field!("total_jsd");
        self.records = get_field!("records");
        self.mean_delta_jsd = get_field!("mean_delta_jsd");
        self.std_delta_jsd = get_field!("std_delta_jsd");
        self.cov_delta_jsd = get_field!("cov_delta_jsd");
        self.size = get_field!("size");
        self.k = get_field!("k");
        self.num_states = get_field!("num_states");
        Ok(())
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
