use crate::records_py::SummedRecordsResult;
use crate::zarr_io::ZarrStore;
use core::panic;
use std::collections::HashSet;

use crate::record::{KmerSeq, LazySeqRecord, SeqRecord, entropy};

#[derive(Debug, Clone)]
/// Container of most divergent sequences
pub struct SummedRecords {
    pub records: Vec<KmerSeq>,
    pub size: u32,
    /* summed_kfreqs will be the sum of
    the kfreqs for all members in records aside
    from that denoted as lowest. Similarly, for
    summed_entropies. */
    pub summed_kfreqs: Vec<f64>,
    pub summed_entropies: f64,
    // total_jsd is from all records
    pub total_jsd: f64,
    // The record with the minimum delta_JSD.
    pub lowest_index: u32,
    pub seqids: HashSet<String>,
}

impl SummedRecords {
    pub fn new(records: Vec<KmerSeq>) -> Self {
        if records.is_empty() {
            panic!("records cannot be empty");
        }
        /* size is the len of records
        iterate over records
        make the summed kcounts and summed entropies
        */
        let size: u32 = records.len() as u32;
        let mut summed_entropies: f64 = 0.0;
        let num_kmers: usize = records[0].kfreqs.len();
        let mut summed_kfreqs: Vec<f64> = vec![0f64; num_kmers];
        for record in &records {
            iadd_vectors(&mut summed_kfreqs, &record.kfreqs);
            summed_entropies += record.entropy;
        }
        // calculate total jsd
        let mean_kfreqs = div_vector(&summed_kfreqs, size as f64);
        let total_jsd: f64 = entropy(&mean_kfreqs) - summed_entropies / size as f64;
        /* compute delta_jsd for each KmerSeq
        panic if there's a nan, displaying KmerSeq.seqid
        record the index of the KmerSeq with the lowest delta_jsd
        at the end subtract lowest from all for the summed_kcounts
        and summed_entropies  */
        let seqids: HashSet<String> = records.iter().map(|r| r.seqid.to_string()).collect();
        let lowest_index: u32 = get_lowest_record_index(
            &records,
            num_kmers,
            &summed_kfreqs,
            summed_entropies,
            total_jsd,
        );
        Self {
            records,
            size,
            summed_kfreqs,
            summed_entropies,
            total_jsd,
            lowest_index,
            seqids,
        }
    }

    pub fn delta_jsd(&self, rec: &KmerSeq) -> f64 {
        if self.seqids.contains(&rec.seqid) {
            return 0.0;
        }
        let lowest_rec = &self.records[self.lowest_index as usize];
        let mut mean_kfreqs = vec![0f64; lowest_rec.kfreqs.len()];
        let mean_entropy =
            (self.summed_entropies - lowest_rec.entropy + rec.entropy) / self.size as f64;
        for i in 0..mean_kfreqs.len() {
            mean_kfreqs[i] =
                (self.summed_kfreqs[i] - lowest_rec.kfreqs[i] + rec.kfreqs[i]) / self.size as f64;
        }
        let entropy_of_mean = entropy(&mean_kfreqs);
        entropy_of_mean - mean_entropy
    }

    pub fn increases_jsd(&self, rec: &KmerSeq) -> bool {
        if self.seqids.contains(&rec.seqid) {
            return false;
        }
        let jsd = self.delta_jsd(rec);
        jsd > self.total_jsd + f64::EPSILON
    }

    pub fn drop_lowest(&mut self) {
        let old_rec = self.records.remove(self.lowest_index as usize);
        // remove from hashset
        self.seqids.remove(&old_rec.seqid);

        // remove lowest from total entropies
        let num_kmers: usize = old_rec.kfreqs.len();
        self.summed_entropies -= old_rec.entropy;
        // and remove from summed_kfreqs
        for i in 0..num_kmers {
            self.summed_kfreqs[i] -= old_rec.kfreqs[i];
            if self.summed_kfreqs[i] <= f64::EPSILON {
                self.summed_kfreqs[i] = 0.0;
            }
        }
    }

    pub fn replace_lowest(&mut self, rec: KmerSeq) {
        if self.seqids.contains(&rec.seqid) {
            return;
        }
        self.drop_lowest();
        // add the new record
        self.push(rec);
    }

    pub fn push(&mut self, rec: KmerSeq) {
        if self.seqids.contains(&rec.seqid) {
            return;
        }
        let num_kmers: usize = self.records[0].kfreqs.len();
        let seqid = rec.seqid.to_string();
        self.seqids.insert(seqid);

        // add to total entropies
        self.summed_entropies += rec.entropy;
        // and summed_kfreqs
        for i in 0..num_kmers {
            self.summed_kfreqs[i] += &rec.kfreqs[i];
        }
        self.records.push(rec);
        self.size = self.records.len() as u32;
        let mean_kfreqs = div_vector(&self.summed_kfreqs, self.size as f64);
        let mean_entropy = entropy(&mean_kfreqs);
        self.total_jsd = mean_entropy - self.summed_entropies / self.size as f64;
        let lowest_index: u32 = get_lowest_record_index(
            &self.records,
            num_kmers,
            &self.summed_kfreqs,
            self.summed_entropies,
            self.total_jsd,
        );
        self.lowest_index = lowest_index;
    }

    pub fn get_by_seqid(&self, seqid: &str) -> Option<&KmerSeq> {
        self.records.iter().find(|r| r.seqid == seqid)
    }

    pub fn mean_jsd(&self) -> f64 {
        self.total_jsd / self.size as f64
    }
    pub fn mean_delta_jsd(&self) -> f64 {
        self.records.iter().map(|r| r.delta_jsd.get()).sum::<f64>() / self.size as f64
    }

    pub fn std_delta_jsd(&self) -> f64 {
        let mean = self.mean_delta_jsd();
        let mut sum = 0.0;
        for r in &self.records {
            sum += (r.delta_jsd.get() - mean).powi(2);
        }
        // unbiased estimator
        (sum / (self.size as f64 - 1.0)).sqrt()
    }

    pub fn cov_delta_jsd(&self) -> f64 {
        self.std_delta_jsd() / self.mean_delta_jsd()
    }

    pub fn record_deltas(&self) -> Vec<(String, f64)> {
        self.records
            .iter()
            .map(|r| (r.seqid.clone(), r.delta_jsd.get()))
            .collect::<Vec<(String, f64)>>()
    }

    pub fn clone(&self) -> Self {
        let records = self
            .records
            .iter()
            .map(|r| r.clone())
            .collect::<Vec<KmerSeq>>();
        Self::new(records)
    }

    pub fn for_display(&self, title: String) -> String {
        let record_deltas = self
            .record_deltas()
            .iter()
            .map(|(s, d)| format!("{}\t{}", s, d))
            .collect::<Vec<String>>()
            .join("\n");

        format!(
            "# {}\n# seqid\tdelta_jsd\n{}\ntotal_jsd={}",
            title, record_deltas, self.total_jsd
        )
    }

    pub fn get_result(&self) -> SummedRecordsResult {
        SummedRecordsResult {
            record_deltas: self.record_deltas(),
            total_jsd: self.total_jsd,
            mean_delta_jsd: self.mean_delta_jsd(),
            std_delta_jsd: self.std_delta_jsd(),
            cov_delta_jsd: self.cov_delta_jsd(),
            size: self.size as usize,
            k: self.records[0].k,
            num_states: self.records[0].num_states,
        }
    }
}

/// update delta_jsd on records and return index for lowest delta_jsd
fn get_lowest_record_index(
    records: &[KmerSeq],
    num_kmers: usize,
    summed_kfreqs: &[f64],
    summed_entropies: f64,
    total_jsd: f64,
) -> u32 {
    let div = records.len() as f64 - 1.0;
    let mut min_delta_jsd: f64 = 1e6;
    let mut lowest_index: u32 = 0;
    let mut mean_kfreqs = vec![0f64; num_kmers];
    for (i, record) in records.iter().enumerate() {
        // calculating the mean entropy of records but this one
        let mean_entropy = (summed_entropies - record.entropy) / div;

        // calculate the mean kfreqs of records but this one
        updated_mean_freqs(&mut mean_kfreqs, summed_kfreqs, &record.kfreqs, div);
        let entropy_of_mean = entropy(&mean_kfreqs);

        // JSD for all records but this one
        let jsd = entropy_of_mean - mean_entropy;
        // the JSD contribution from the current record
        record.delta_jsd.set(total_jsd - jsd);
        if record.delta_jsd.get() < min_delta_jsd {
            min_delta_jsd = record.delta_jsd.get();
            lowest_index = i as u32;
        }
    }
    lowest_index
}

/// add two f64 vectors, updating first in place
fn iadd_vectors(summed_freqs: &mut [f64], freqs: &[f64]) {
    assert_eq!(
        summed_freqs.len(),
        freqs.len(),
        "length mismatch for add_vectors"
    );
    for j in 0..freqs.len() {
        summed_freqs[j] += freqs[j];
    }
}

/// return vector<f64> divided by a scalar
fn div_vector(vector: &[f64], div: f64) -> Vec<f64> {
    if div == 0.0 {
        panic!("division by zero");
    }

    vector.iter().map(|x| *x / div).collect()
}

/// update vector
fn updated_mean_freqs(dest: &mut [f64], total_freqs: &[f64], record_kfreqs: &[f64], div: f64) {
    if dest.len() != total_freqs.len() || dest.len() != record_kfreqs.len() {
        panic!("length mismatch for mean_freqs")
    };
    for i in 0..dest.len() {
        dest[i] = (total_freqs[i] - record_kfreqs[i]) / div;
        if dest[i] <= f64::EPSILON {
            dest[i] = 0.0;
        }
    }
}

fn get_lazyrecords_and_init_summed_records(
    store: &ZarrStore,
    seqids: Vec<String>,
    n: usize,
    k: usize,
    num_states: usize,
) -> (Vec<LazySeqRecord<'_>>, SummedRecords) {
    let records: Vec<LazySeqRecord> = seqids
        .iter()
        .map(|seqid| LazySeqRecord::new(seqid, num_states, &store))
        .collect();
    let mut init: Vec<KmerSeq> = Vec::new();
    for i in 0..n {
        let kseq = records[i].to_kmerseq(k);
        if kseq.is_ok() {
            init.push(kseq.unwrap());
        }
    }
    let summed = SummedRecords::new(init);
    (records, summed)
}

/// returns the SummedRecords object containing nmost divergent sequences
pub fn select_nmost_divergent(
    store: &ZarrStore,
    n: usize,
    k: usize,
    num_states: usize,
    seqids: Option<Vec<String>>,
) -> SummedRecords {
    let seqids = match seqids {
        Some(seqids) => seqids,
        None => store.list_unique_seqids().unwrap(),
    };

    if seqids.len() < n {
        panic!("The number of sequences {} is < n {}", seqids.len(), n);
    }

    let (records, mut summed) =
        get_lazyrecords_and_init_summed_records(store, seqids, n, k, num_states);

    // now iterate over the rest
    for i in n..records.len() {
        let rec = records[i].to_kmerseq(k);
        if !rec.is_ok() {
            continue;
        }
        let rec = rec.unwrap();
        if summed.increases_jsd(&rec) {
            summed.replace_lowest(rec);
        }
    }
    summed
}

pub enum Stat {
    Cov,
    Std,
}

/// returns SummedRecords of sequences that maximise divergence
pub fn select_max_divergent(
    store: &ZarrStore,
    stat: Stat,
    min_size: usize,
    max_size: usize,
    k: usize,
    num_states: usize,
    seqids: Option<Vec<String>>,
) -> SummedRecords {
    let seqids = match seqids {
        Some(seqids) => seqids,
        None => store.list_unique_seqids().unwrap(),
    };

    if seqids.len() < min_size {
        panic!(
            "The number of sequences {} is < n {}",
            seqids.len(),
            min_size
        );
    }

    let max_size = if seqids.len() > max_size {
        max_size
    } else {
        seqids.len()
    };
    let (records, mut summed) =
        get_lazyrecords_and_init_summed_records(store, seqids, min_size, k, num_states);

    // now iterate over the rest
    for i in min_size..records.len() {
        let rec = records[i].to_kmerseq(k);
        if !rec.is_ok() {
            continue;
        }
        let rec = rec.unwrap();
        if !summed.increases_jsd(&rec) {
            continue;
        } else if summed.size == max_size as u32 {
            summed.replace_lowest(rec);
            continue;
        }

        let mut new_summed = summed.clone();
        new_summed.push(rec);
        summed = match stat {
            Stat::Cov => {
                if new_summed.cov_delta_jsd() > summed.cov_delta_jsd() {
                    new_summed
                } else {
                    summed
                }
            }
            Stat::Std => {
                if new_summed.std_delta_jsd() > summed.std_delta_jsd() {
                    new_summed
                } else {
                    summed
                }
            }
        };
    }
    summed
}

pub fn make_summed_records(
    records: Vec<(String, Vec<u8>)>,
    k: usize,
    num_states: usize,
) -> SummedRecords {
    let mut kseq_recs: Vec<KmerSeq> = Vec::new();

    for (seqid, seq) in records.iter() {
        let seqrec = SeqRecord::new(seqid, seq, num_states);
        let kseq = seqrec.to_kmerseq(k);
        if kseq.is_ok() {
            kseq_recs.push(kseq.unwrap());
        }
    }
    SummedRecords::new(kseq_recs)
}

#[cfg(test)]
mod tests {
    use crate::record::SeqRecord;
    use rstest::{fixture, rstest};

    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use tempfile::TempDir;

    #[fixture]
    fn temp_dir() -> TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    #[test]
    fn add_vectors() {
        let mut v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let expect = vec![5.0, 7.0, 9.0];
        iadd_vectors(&mut v1, &v2);
        assert_eq!(v1, expect);
        assert_eq!(v2, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn invalid_records() {
        let mut v1 = vec![1.0, 2.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let result = catch_unwind(AssertUnwindSafe(|| iadd_vectors(&mut v1, &v2)));
        assert!(result.is_err());
    }

    #[test]
    fn divide_vector() {
        let v1 = vec![1.0, 2.0, 3.0];
        let div: f64 = 2.0;
        let expect = vec![1.0 / div, 2.0 / div, 3.0 / div];

        let got: Vec<f64> = div_vector(&v1, div);
        assert_eq!(got, expect);
    }

    #[test]
    fn idivide_vector_divide_by_zero() {
        let mut v1 = vec![1.0, 2.0, 3.0];
        let result = catch_unwind(AssertUnwindSafe(|| idiv_vector(&mut v1, 0.0)));
        assert!(result.is_err());
    }

    #[test]
    fn invalid_mean_freqs() {
        let mut wrk = vec![0.0, 0.1, 0.2, 0.1];
        let tots = vec![0.0, 0.1, 0.2, 0.1];
        let rec = vec![0.0, 0.1, 0.2];

        let result = catch_unwind(AssertUnwindSafe(|| {
            updated_mean_freqs(&mut wrk, &tots, &rec, 4.0)
        }));
        assert!(result.is_err());
    }

    #[test]
    fn div_vector_by_zero() {
        let v1 = vec![1.0, 2.0, 3.0];
        let div: f64 = 0.0;
        let result = catch_unwind(|| div_vector(&v1, div));
        assert!(result.is_err());
    }

    #[fixture]
    fn summed() -> SummedRecords {
        SummedRecords::new(
            [
                SeqRecord::new("seq1", &[0, 1, 2, 3], 4),
                SeqRecord::new("seq2", &[0, 1, 2, 2, 3], 4),
                SeqRecord::new("seq3", &[3, 0, 0], 4),
            ]
            .iter()
            .map(|sr| sr.to_kmerseq(1).unwrap())
            .collect(),
        )
    }

    #[rstest]
    fn construct_summed_records(summed: SummedRecords) {
        // values from python version of diverse-seq
        assert_eq!(summed.size, 3);
        assert_eq!(summed.total_jsd, 0.31174344844038515);

        // let entropies: Vec<f64> = records.iter().map(|r| r.entropy).copied().collect();
        let entropies: Vec<f64> = summed.records.iter().map(|r| r.entropy).collect();
        assert_eq!(entropies, vec![2.0, 1.9219280948873623, 0.9182958340544896]);
        let delta_jsds: Vec<f64> = summed.records.iter().map(|r| r.delta_jsd.get()).collect();
        assert_eq!(summed.summed_entropies, 4.840223928941851);
        assert_eq!(
            delta_jsds,
            vec![
                -0.09602255461972087,
                -0.013445832597674734,
                0.2931216853661194
            ]
        );
    }
    #[test]
    fn invalid_input_summed() {
        let records: Vec<KmerSeq> = vec![];
        let result = catch_unwind(AssertUnwindSafe(|| SummedRecords::new(records)));
        assert!(result.is_err());
    }

    #[rstest]
    fn check_increases_jsd(summed: SummedRecords) {
        let better = SeqRecord::new("seq4", &[0, 1, 2, 1], 4);
        assert!(summed.increases_jsd(&better.to_kmerseq(1).unwrap()));
    }

    #[rstest]
    fn check_not_increases_jsd(summed: SummedRecords) {
        let same = &summed.records[0];
        assert!(!summed.increases_jsd(&same));
    }

    #[rstest]
    fn check_delta_jsd_same(summed: SummedRecords) {
        let same = &summed.records[0];
        assert_eq!(summed.delta_jsd(&same), 0.0);
    }

    #[rstest]
    fn check_increases_jsd_same(summed: SummedRecords) {
        assert!(!summed.increases_jsd(&summed.records[0]));
    }
    #[rstest]
    fn check_replace_lowest(mut summed: SummedRecords) {
        let better = SeqRecord::new("seq4", &[0, 1, 2, 1], 4);
        summed.replace_lowest(better.to_kmerseq(1).unwrap());
        assert!(summed.seqids.contains("seq4"));
    }

    #[rstest]
    fn check_push(mut summed: SummedRecords) {
        let better = SeqRecord::new("seq4", &[0, 1, 2, 1], 4);
        let orig_jsd = summed.total_jsd;
        let orig_size = summed.size;

        summed.push(better.to_kmerseq(1).unwrap());
        assert_eq!(summed.size, orig_size + 1);
        assert_eq!(summed.records.len() as u32, orig_size + 1);
        assert!(summed.seqids.contains("seq4"));
        assert!(summed.total_jsd != orig_jsd);
    }

    #[rstest]
    fn check_mean_jsd(summed: SummedRecords) {
        assert_eq!(summed.mean_jsd(), summed.total_jsd / summed.size as f64);
    }

    #[rstest]
    fn check_mean_delta_jsd(summed: SummedRecords) {
        // value from python version of diverse-seq
        assert_eq!(summed.mean_delta_jsd(), 0.061217766049574594);
    }

    #[rstest]
    fn check_std_delta_jsd(summed: SummedRecords) {
        assert_eq!(summed.std_delta_jsd(), 0.20503487410866827);
    }
    #[rstest]
    fn check_cov_delta_jsd(summed: SummedRecords) {
        assert_eq!(
            summed.cov_delta_jsd(),
            summed.std_delta_jsd() / summed.mean_delta_jsd()
        );
    }

    fn make_zstore(path: std::path::PathBuf, add_invalid: bool) -> ZarrStore {
        let mut store = ZarrStore::new(Some(path), "w").unwrap();
        let sequences = vec![
            vec![0, 0, 1, 1],       // seq1
            vec![1, 1, 1, 3],       // seq2
            vec![0, 0, 0, 2, 2, 2], // seq3
            vec![1, 1, 1, 1, 3],    // seq4
            vec![1, 2],             // seq5
        ];
        for (i, data) in sequences.into_iter().enumerate() {
            let seqid = format!("seq{}", i + 1);
            let _ = store.add_uint8_array(&seqid, &data, None);
        }
        if add_invalid {
            let _ = store.add_uint8_array("seq-invalid", &vec![4, 4, 4, 4], None);
        }
        store
    }

    fn summed234() -> SummedRecords {
        SummedRecords::new(
            [
                SeqRecord::new("seq3", &[0, 0, 0, 2, 2, 2], 4),
                SeqRecord::new("seq4", &[1, 1, 1, 1, 3], 4),
                SeqRecord::new("seq2", &[1, 1, 1, 3], 4),
            ]
            .iter()
            .map(|sr| sr.to_kmerseq(1).unwrap())
            .collect(),
        )
    }

    #[rstest]
    fn checked_most_divergent(temp_dir: TempDir) {
        let size = 3;
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, false);
        let rs = select_nmost_divergent(&zstore, size, 1, 4, None);
        assert_eq!(rs.size, size as u32);
        let expect = summed234();
        let gs4 = rs.get_by_seqid("seq4").unwrap();
        let es4 = expect.get_by_seqid("seq4").unwrap();
        assert_eq!(gs4.delta_jsd.get(), es4.delta_jsd.get());
        let gs3 = rs.get_by_seqid("seq3").unwrap();
        let es3 = expect.get_by_seqid("seq3").unwrap();
        assert_eq!(gs3.delta_jsd.get(), es3.delta_jsd.get());
    }

    #[rstest]
    fn checked_summed_records_add_duplicate(temp_dir: TempDir) {
        let size = 3;
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, false);
        let mut rs = select_nmost_divergent(&zstore, size, 1, 4, None);
        let one = rs.records[1].clone();
        rs.push(one);
        assert_eq!(rs.size, size as u32);
    }

    #[rstest]
    fn checked_summed_records_replace_lowest_duplicate(temp_dir: TempDir) {
        let size = 3;
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, false);
        let mut rs = select_nmost_divergent(&zstore, size, 1, 4, None);
        let one = rs.records[1].clone();
        rs.replace_lowest(one);
        assert_eq!(rs.size, size as u32);
    }

    #[rstest]
    fn checked_summed_records_record_deltas(temp_dir: TempDir) {
        let size = 3;
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, false);
        let rs = select_nmost_divergent(&zstore, size, 1, 4, None);
        let result = rs.record_deltas();
        assert_eq!(result.len(), size);
        let seqids = result
            .iter()
            .map(|r| r.0.to_string())
            .collect::<HashSet<String>>();
        assert_eq!(seqids, rs.seqids);
        assert!(result.iter().all(|v| !v.1.is_nan()));
    }

    #[rstest]
    fn checked_summed_records_with_invalid(temp_dir: TempDir) {
        let size = 3;
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, true);
        let mut rs = select_nmost_divergent(&zstore, size, 1, 4, None);
        let one = rs.records[1].clone();
        rs.replace_lowest(one);
        assert_eq!(rs.size, size as u32);
    }

    #[rstest]
    fn checked_most_divergent_invalid_n(temp_dir: TempDir) {
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, false);

        let result = catch_unwind(AssertUnwindSafe(|| {
            select_nmost_divergent(&zstore, 20, 1, 4, None)
        }));
        assert!(result.is_err());
    }

    #[rstest]
    fn checked_most_divergent_with_seqids(temp_dir: TempDir) {
        let size = 3;
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, false);
        let seqids = vec!["seq1".to_string(), "seq3".to_string(), "seq5".to_string()];
        let rs = select_nmost_divergent(&zstore, size, 1, 4, Some(seqids.clone()));
        assert_eq!(rs.size, size as u32);
        let expect = seqids.into_iter().collect::<HashSet<String>>();
        assert_eq!(rs.seqids, expect);
    }

    #[test]
    fn checked_clone() {
        let orig = summed234();
        let clone = orig.clone();
        assert_eq!(orig.size, clone.size);
        assert_eq!(orig.total_jsd, clone.total_jsd);
        assert_eq!(orig.lowest_index, clone.lowest_index);
        assert_eq!(orig.seqids, clone.seqids);
        assert_eq!(orig.summed_entropies, clone.summed_entropies);
        assert_eq!(orig.summed_kfreqs, clone.summed_kfreqs);
    }

    #[test]
    fn checked_result() {
        let orig = summed234();
        let stats = orig.get_result();
        assert_eq!(stats.size, orig.size as usize);
        assert_eq!(stats.total_jsd, orig.total_jsd);
        assert_eq!(stats.mean_delta_jsd, orig.mean_delta_jsd());
        assert_eq!(stats.std_delta_jsd, orig.std_delta_jsd());
        assert_eq!(stats.cov_delta_jsd, orig.cov_delta_jsd());
        assert_eq!(stats.record_deltas, orig.record_deltas());
    }

    #[test]
    fn exercise_display() {
        // shouldn't panic
        let sr = summed234();
        let r = sr.for_display("test".to_string());
        println!("{}", r);
    }

    #[rstest]
    #[case(Stat::Cov, Some(vec!["seq1".to_string(), "seq2".to_string(), "seq3".to_string(), "seq4".to_string(), "seq5".to_string()]))]
    #[case(Stat::Cov, None)]
    #[case(Stat::Std, None)]
    #[case(Stat::Std, Some(vec!["seq1".to_string(), "seq2".to_string(), "seq3".to_string(), "seq4".to_string(), "seq5".to_string()]))]
    fn check_max_divergent(
        temp_dir: TempDir,
        #[case] stat: Stat,
        #[case] seqids: Option<Vec<String>>,
    ) {
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, true);
        let min_size = 3;
        let max_size = 4;
        let num_states = 4;
        let k = 1;

        let sr = select_max_divergent(&zstore, stat, min_size, max_size, k, num_states, seqids);
        assert!(sr.size >= min_size as u32);
        assert!(sr.size <= max_size as u32);
    }

    #[rstest]
    fn check_max_divergent_invalid_size(temp_dir: TempDir) {
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, true);
        let min_size = 30;
        let max_size = 40;
        let num_states = 4;
        let k = 1;
        let result = catch_unwind(AssertUnwindSafe(|| {
            select_max_divergent(&zstore, Stat::Std, min_size, max_size, k, num_states, None);
        }));
        assert!(result.is_err());
    }

    #[rstest]
    fn check_max_divergent_max_size(temp_dir: TempDir) {
        // max_size is bigger than number of records
        let path = temp_dir.path().join("zarrs.dvseqz");
        let zstore = make_zstore(path, true);
        let min_size = 3;
        let max_size = 10;
        let num_states = 4;
        let k = 1;
        let sr = select_max_divergent(&zstore, Stat::Std, min_size, max_size, k, num_states, None);
        assert!(sr.size >= min_size as u32);
        assert!(sr.size <= zstore.list_unique_seqids().unwrap().len() as u32);
    }
    #[fixture]
    fn big_seqs() -> Vec<(String, Vec<u8>)> {
        vec![
            (
                "FlyingFox".to_string(),
                vec![
                    3, 2, 0, 2, 3, 0, 2, 0, 1, 0, 1, 2, 0, 0, 2, 1, 0, 3, 3, 2, 2, 1, 1, 0, 3, 2,
                    1, 2, 0, 1, 1, 1, 2, 3, 2, 3, 2, 3, 3, 3, 0, 2, 2, 2, 3, 2, 1, 2, 3, 1, 2, 1,
                    1, 2, 2, 2, 1, 1, 2, 0, 1, 2, 0, 3, 1, 2, 3, 1, 2, 2, 0, 0, 2, 2, 2, 2, 2, 1,
                    1, 1, 1, 2, 3, 2, 3, 2, 2, 1, 0, 0, 2, 0, 0, 1, 2, 0, 3, 3, 0, 0, 3, 0, 0, 1,
                    0, 3, 2, 2, 3, 2, 0, 2, 1, 0, 2, 3, 2, 2, 2, 0, 3, 2, 0, 3, 1, 2, 3, 2, 3, 3,
                    3, 1, 0, 0, 0, 2, 2, 2, 3, 2, 0, 1, 1, 2, 0, 0, 3, 2, 3, 2,
                ],
            ),
            (
                "DogFaced".to_string(),
                vec![
                    1, 2, 2, 3, 2, 0, 2, 3, 0, 2, 0, 1, 0, 1, 2, 0, 0, 2, 1, 0, 3, 3, 2, 2, 0, 1,
                    0, 3, 2, 1, 2, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 3, 2, 1, 2, 3,
                    1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 0, 3, 1, 2, 3, 2, 0, 1, 0, 3, 0, 3, 0,
                    3, 1, 2, 3, 1, 2, 2, 0, 0, 3, 2, 2, 2, 2, 1, 1, 1, 1, 2, 3, 2, 3, 2, 2, 1, 0,
                    0, 2, 0, 0, 1, 2, 0, 3, 2, 0, 0, 3, 0, 0, 0, 0, 2, 2, 2, 3, 2, 2, 2, 1, 0, 2,
                    3, 2, 2, 2, 0, 3, 2, 1, 2, 1, 2, 3, 2, 3, 2, 3, 1, 0, 0, 0,
                ],
            ),
            (
                "FreeTaile".to_string(),
                vec![
                    3, 2, 0, 2, 3, 0, 2, 0, 1, 0, 1, 2, 0, 0, 2, 1, 0, 3, 3, 2, 2, 1, 1, 0, 3, 2,
                    1, 2, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 3, 1, 0, 1,
                    1, 2, 2, 2, 0, 1, 2, 2, 0, 2, 0, 3, 1, 3, 3, 3, 0, 1, 0, 3, 0, 3, 0, 3, 1, 2,
                    3, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 1, 1, 5, 1, 2, 2, 3, 3, 2, 2, 1, 0, 0, 2, 0,
                    1, 1, 2, 0, 3, 3, 0, 0, 3, 0, 0, 1, 0, 2, 2, 1, 3, 2, 0, 2, 2, 0, 2, 3, 2, 2,
                    2, 0, 3, 2, 1, 2, 3, 2, 3, 2, 3, 3, 2, 1, 2, 0, 0, 2, 2, 3,
                ],
            ),
            (
                "LittleBro".to_string(),
                vec![
                    3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 0, 2, 3, 0, 3, 0, 1, 0, 1, 2, 0, 0, 2, 1, 0,
                    3, 3, 2, 2, 1, 1, 0, 3, 2, 1, 2, 0, 1, 1, 1, 2, 3, 3, 3, 2, 2, 3, 3, 1, 2, 2,
                    2, 2, 2, 1, 2, 3, 1, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2, 0, 3, 0, 3, 3, 3, 3, 2, 0,
                    1, 0, 3, 0, 3, 0, 3, 1, 2, 3, 1, 2, 3, 0, 0, 2, 2, 2, 2, 2, 0, 1, 1, 0, 2, 2,
                    2, 3, 2, 2, 1, 0, 0, 2, 0, 0, 1, 3, 0, 3, 3, 0, 0, 3, 0, 0, 1, 0, 2, 2, 2, 3,
                    2, 0, 2, 0, 0, 2, 3, 2, 2, 2, 0, 3, 2, 1, 2, 3, 2, 3, 2, 3,
                ],
            ),
        ]
    }

    #[rstest]
    fn check_big(big_seqs: Vec<(String, Vec<u8>)>) {
        let srw = make_summed_records(big_seqs, 3, 4);
        let rs = srw.get_result();
        assert!(!rs.total_jsd.is_nan());
    }
}
