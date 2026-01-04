#[cfg(feature = "python")]
use pyo3::prelude::{Bound, PyModule, PyResult, pyfunction, pymodule};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;
mod record;
use record::{KmerSeq, SeqRecord};
mod records;
use records::SummedRecords;
pub mod zarr_io;
use zarr_io::ZarrStore;

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
    let kmer_seqs: Vec<KmerSeq<'_>> = seq_records.iter().map(|sr| sr.to_kmerseq(k)).collect();
    // and now create the SummedRecords container
    let selected = SummedRecords::new(kmer_seqs);

    let lowest: &KmerSeq<'_> = &selected.records[selected.lowest_index as usize];
    println!(
        "Lowest seq {:?} has delta_jsd {:.4}",
        lowest.seqid,
        lowest.delta_jsd.get()
    );
    Ok(selected.total_jsd)
}

#[pyfunction]
#[pyo3(signature = (seqid, seq, num_states=4, k=2))]
fn play(seqid: &str, seq: &[u8], num_states: usize, k: usize) -> PyResult<Vec<f64>> {
    let sr = SeqRecord::new(seqid, seq, num_states);
    let r = sr.to_kmerseq(k);
    println!(
        "Playing with '{}', which has entropy={:.4}",
        r.seqid, r.entropy
    );
    Ok(r.kfreqs)
}

#[pyfunction]
#[pyo3(signature = (seqid, seq, num_states=4, k=2))]
fn get_entropy(seqid: &str, seq: &[u8], num_states: usize, k: usize) -> PyResult<f64> {
    let sr = SeqRecord::new(seqid, seq, num_states);
    let r = sr.to_kmerseq(k);
    println!(
        "Playing with '{}', which has entropy={:.4}",
        r.seqid, r.entropy
    );
    Ok(r.entropy)
}

/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn _dvs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(play, m)?)?;
    m.add_function(wrap_pyfunction!(proc_seqs, m)?)?;
    Ok(())
}
