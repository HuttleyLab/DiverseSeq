#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::{Bound, PyModule, PyResult, pyfunction, pymodule};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;
mod record;
use record::{KmerSeq, LazySeq, SeqRecord};
mod records;
use records::{Stat, SummedRecords, select_max_divergent, select_nmost_divergent};
mod records_py;
use records_py::{SummedRecordsResult, SummedRecordsWrapper};
mod zarr_io;
use zarr_py::ZarrStoreWrapper;
pub mod distance;
use distance::mash_sketch;
pub mod zarr_py;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (seqids, seqs, num_states=4, k=2, max_size=3))]
fn proc_seqs(
    seqids: Vec<String>,
    seqs: Vec<Vec<u8>>,
    num_states: usize,
    k: usize,
    max_size: usize,
) -> PyResult<f64> {
    let seq_records: Vec<SeqRecord<'_>> = seqids
        .iter()
        .take(max_size)
        .zip(seqs.iter().take(max_size))
        .map(|(id, seq)| SeqRecord::new(id.as_str(), seq.as_slice(), num_states))
        .collect();
    // now make them kmer seqs
    let kmer_seqs: Vec<KmerSeq> = seq_records
        .iter()
        .map(|sr| sr.to_kmerseq(k))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // and now create the SummedRecords container
    let selected = SummedRecords::new(kmer_seqs);

    let lowest: &KmerSeq = &selected.records[selected.lowest_index as usize];
    Ok(selected.total_jsd)
}

#[pyfunction]
#[pyo3(signature = (seqid, seq, num_states=4, k=2))]
fn get_entropy(seqid: &str, seq: &[u8], num_states: usize, k: usize) -> PyResult<f64> {
    let sr = SeqRecord::new(seqid, seq, num_states);
    let r = sr
        .to_kmerseq(k)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(r.entropy)
}

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
#[pyo3(signature = (store,seqid,k,num_states=4))]
fn get_kfreqs(
    store: &ZarrStoreWrapper,
    seqid: String,
    k: usize,
    num_states: usize,
) -> PyResult<Vec<f64>> {
    let data = store.read(&seqid).unwrap();
    let seq = SeqRecord::new(&seqid, &data, num_states);
    Ok(seq.to_kmerseq(k).unwrap().kfreqs)
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
    let result = select_nmost_divergent(&store.store, n, k, num_states, seqids);
    Ok(result.get_result())
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
    m.add_function(wrap_pyfunction!(get_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(proc_seqs, m)?)?;
    m.add_function(wrap_pyfunction!(make_zarr_store, m)?)?;
    m.add_function(wrap_pyfunction!(make_zarr_store, m)?)?;
    m.add_function(wrap_pyfunction!(get_seqids_from_store, m)?)?;
    m.add_function(wrap_pyfunction!(nmost_divergent, m)?)?;
    m.add_function(wrap_pyfunction!(max_divergent, m)?)?;
    m.add_function(wrap_pyfunction!(mash_sketch, m)?)?;
    m.add_function(wrap_pyfunction!(get_kfreqs, m)?)?;
    m.add_function(wrap_pyfunction!(get_delta_jsd_calculator, m)?)?;
    Ok(())
}
