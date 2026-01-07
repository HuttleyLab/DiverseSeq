use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
#[cfg(feature = "python")]
use pyo3::prelude::{Bound, PyModule, PyResult, pyfunction, pymodule};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;
mod record;
use record::LazySeq;
mod records;
use records::{Stat, select_max_divergent, select_nmost_divergent};
mod records_py;
use records_py::{SummedRecordsResult, SummedRecordsWrapper};
mod zarr_io;
use zarr_py::ZarrStoreWrapper;
pub mod distance;
use distance::mash_sketch;
pub mod zarr_py;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[pyfunction]
#[pyo3(signature = (path=None, mode="r"))]
fn make_zarr_store(path: Option<String>, mode: &str) -> PyResult<ZarrStoreWrapper> {
    ZarrStoreWrapper::new(path, mode)
}

#[pyfunction]
#[pyo3(signature = (path,))]
fn get_seqids_from_store(path: &str) -> PyResult<Vec<String>> {
    let store = ZarrStoreWrapper::new(Some(path.to_string()), "r")?;
    Ok(store.get_seqids().unwrap())
}

#[pyfunction]
#[pyo3(signature = (store, n, k, num_states=4, seqids=None))]
fn nmost_divergent(
    store: &ZarrStoreWrapper,
    n: usize,
    k: usize,
    num_states: usize,
    seqids: Option<Vec<String>>,
) -> PyResult<SummedRecordsResult> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        select_nmost_divergent(&store.store, n, k, num_states, seqids)
    }));

    match result {
        Ok(r) => Ok(r.get_result()),
        Err(payload) => {
            // Try to surface the original panic message if available
            let msg: String = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "select_nmost_divergent panicked (likely: n exceeds available sequences)"
                    .to_string()
            };
            Err(PyValueError::new_err(msg))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (store, min_size, max_size, k, num_states=4, seqids=None, stat="stdev".to_string()))]
fn max_divergent(
    store: &ZarrStoreWrapper,
    min_size: usize,
    max_size: usize,
    k: usize,
    num_states: usize,
    seqids: Option<Vec<String>>,
    stat: String,
) -> PyResult<SummedRecordsResult> {
    let stat = if stat == "stdev" {
        Stat::Std
    } else {
        Stat::Cov
    };

    let result = select_max_divergent(
        &store.store,
        stat,
        min_size,
        max_size,
        k,
        num_states,
        seqids,
    );
    Ok(result.get_result())
}

#[pyfunction]
#[pyo3(signature = (seqids_seqs, k, num_states=4))]
fn get_delta_jsd_calculator(
    seqids_seqs: Vec<(String, Vec<u8>)>,
    k: usize,
    num_states: usize,
) -> PyResult<SummedRecordsWrapper> {
    let result = SummedRecordsWrapper::new(seqids_seqs, k, num_states);
    Ok(result)
}

/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn _dvs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZarrStoreWrapper>()?;
    m.add_class::<SummedRecordsResult>()?;
    m.add_class::<LazySeq>()?;
    m.add_function(wrap_pyfunction!(make_zarr_store, m)?)?;
    m.add_function(wrap_pyfunction!(get_seqids_from_store, m)?)?;
    m.add_function(wrap_pyfunction!(nmost_divergent, m)?)?;
    m.add_function(wrap_pyfunction!(max_divergent, m)?)?;
    m.add_function(wrap_pyfunction!(mash_sketch, m)?)?;
    m.add_function(wrap_pyfunction!(get_delta_jsd_calculator, m)?)?;
    Ok(())
}
