use std::collections::HashSet;

use crate::record::{KmerSeq, entropy};

#[derive(Debug)]
/// Container of most divergent sequences
pub struct SummedRecords<'b> {
    pub records: Vec<KmerSeq<'b>>,
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

impl<'b> SummedRecords<'b> {
    pub fn new(records: Vec<KmerSeq<'b>>) -> Self {
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
            size as f64 - 1.0,
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

    pub fn increases_jsd(&self, rec: &KmerSeq<'b>) -> bool {
        if self.seqids.contains(rec.seqid) {
            return false;
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
        let jsd = entropy_of_mean - mean_entropy;
        jsd > self.total_jsd
    }

    pub fn replace_lowest(&mut self, rec: KmerSeq<'b>) {
        if self.seqids.contains(rec.seqid) {
            return;
        }

        let old_rec = self.records.remove(self.lowest_index as usize);
        self.seqids.remove(old_rec.seqid);
        let seqid = rec.seqid.to_string();
        self.seqids.insert(seqid);
        // now modify total entropies etc ...
        let num_kmers: usize = old_rec.kfreqs.len();
        self.summed_entropies -= old_rec.entropy + rec.entropy;
        for i in 0..num_kmers {
            self.summed_kfreqs[i] -= old_rec.kfreqs[i] + rec.kfreqs[i];
        }
        self.records.push(rec);
        let mean_kfreqs = div_vector(&self.summed_kfreqs, self.size as f64);
        let mean_entropy = entropy(&mean_kfreqs);
        self.total_jsd = mean_entropy - self.summed_entropies / self.size as f64;
        let lowest_index: u32 = get_lowest_record_index(
            &self.records,
            num_kmers,
            &self.summed_kfreqs,
            self.summed_entropies,
            self.total_jsd,
            self.size as f64 - 1.0,
        );
        self.lowest_index = lowest_index;
    }

    pub fn push(&mut self, rec: KmerSeq<'b>) {
        if self.seqids.contains(rec.seqid) {
            return;
        }
        let num_kmers: usize = self.records[0].kfreqs.len();
        if rec.kfreqs.is_empty() {
            panic!("kfreqs cannot be empty");
        } else if rec.kfreqs.len() != num_kmers {
            panic!(
                "kfreqs length {} does not match num_kmers {}",
                rec.kfreqs.len(),
                num_kmers
            );
        }

        let seqid = rec.seqid.to_string();
        self.seqids.insert(seqid);
        // now modify total entropies etc ...

        self.summed_entropies += rec.entropy;
        for i in 0..num_kmers {
            self.summed_kfreqs[i] += &rec.kfreqs[i];
        }
        self.records.push(rec);
        self.size += 1;
        let mean_kfreqs = div_vector(&self.summed_kfreqs, self.size as f64);
        let mean_entropy = entropy(&mean_kfreqs);
        self.total_jsd = mean_entropy - self.summed_entropies / self.size as f64;
        let lowest_index: u32 = get_lowest_record_index(
            &self.records,
            num_kmers,
            &self.summed_kfreqs,
            self.summed_entropies,
            self.total_jsd,
            self.size as f64,
        );
        self.lowest_index = lowest_index;
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
}

/// update delta_jsd on records and return index for lowest delta_jsd
fn get_lowest_record_index<'b>(
    records: &[KmerSeq<'b>],
    num_kmers: usize,
    summed_kfreqs: &[f64],
    summed_entropies: f64,
    total_jsd: f64,
    div: f64,
) -> u32 {
    let mut min_delta_jsd: f64 = 1e6;
    let mut lowest_index: u32 = 0;
    let mut mean_kfreqs = vec![0f64; num_kmers];
    for (i, record) in records.iter().enumerate() {
        // calculating the mean entropy of records but this one
        let mean_entropy = (summed_entropies - record.entropy) / div;
        // calculate the mean kfreqs of records but this one
        mean_freqs(&mut mean_kfreqs, summed_kfreqs, &record.kfreqs, div);
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

/// divide f64 vector by a scalar
fn div_vector(summed_val: &[f64], div: f64) -> Vec<f64> {
    if div == 0.0 {
        panic!("division by zero");
    }

    summed_val.iter().map(|x| *x / div).collect()
}

/// update vector
fn mean_freqs(dest: &mut [f64], total_freqs: &[f64], record_kfreqs: &[f64], div: f64) {
    if dest.len() != total_freqs.len() || dest.len() != record_kfreqs.len() {
        panic!("length mismatch for mean_freqs")
    };
    for i in 0..dest.len() {
        dest[i] = (total_freqs[i] - record_kfreqs[i]) / div;
    }
}

#[cfg(test)]
mod tests {
    use crate::record::SeqRecord;
    use rstest::{fixture, rstest};

    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

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
    fn invalid_mean_freqs() {
        let mut wrk = vec![0.0, 0.1, 0.2, 0.1];
        let tots = vec![0.0, 0.1, 0.2, 0.1];
        let rec = vec![0.0, 0.1, 0.2];

        let result = catch_unwind(AssertUnwindSafe(|| mean_freqs(&mut wrk, &tots, &rec, 4.0)));
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
    fn summed() -> SummedRecords<'static> {
        SummedRecords::new(
            [
                SeqRecord::new("seq1", &[0, 1, 2, 3], 4),
                SeqRecord::new("seq2", &[0, 1, 2, 2, 3], 4),
                SeqRecord::new("seq3", &[3, 0, 0], 4),
            ]
            .iter()
            .map(|sr| sr.to_kmerseq(1))
            .collect(),
        )
    }

    #[rstest]
    fn construct_summed_records(summed: SummedRecords<'static>) {
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
    fn check_increases_jsd(summed: SummedRecords<'static>) {
        let better = SeqRecord::new("seq4", &[0, 1, 2, 1], 4);
        assert!(summed.increases_jsd(&better.to_kmerseq(1)));
    }

    #[rstest]
    fn check_increases_jsd_same(summed: SummedRecords<'static>) {
        assert!(!summed.increases_jsd(&summed.records[0]));
    }
    #[rstest]
    fn check_replace_lowest(mut summed: SummedRecords<'static>) {
        let better = SeqRecord::new("seq4", &[0, 1, 2, 1], 4);
        summed.replace_lowest(better.to_kmerseq(1));
        assert!(summed.seqids.contains("seq4"));
    }

    #[rstest]
    fn check_push(mut summed: SummedRecords<'static>) {
        let better = SeqRecord::new("seq4", &[0, 1, 2, 1], 4);
        let orig_jsd = summed.total_jsd;
        let orig_size = summed.size;

        summed.push(better.to_kmerseq(1));
        assert_eq!(summed.size, orig_size + 1);
        assert_eq!(summed.records.len() as u32, orig_size + 1);
        assert!(summed.seqids.contains("seq4"));
        assert!(summed.total_jsd != orig_jsd);
    }

    #[rstest]
    fn check_mean_jsd(summed: SummedRecords<'static>) {
        assert_eq!(summed.mean_jsd(), summed.total_jsd / summed.size as f64);
    }

    #[rstest]
    fn check_mean_delta_jsd(summed: SummedRecords<'static>) {
        // value from python version of diverse-seq
        assert_eq!(summed.mean_delta_jsd(), 0.061217766049574594);
    }

    #[rstest]
    fn check_std_delta_jsd(summed: SummedRecords<'static>) {
        assert_eq!(summed.std_delta_jsd(), 0.20503487410866827);
    }
    #[rstest]
    fn check_cov_delta_jsd(summed: SummedRecords<'static>) {
        assert_eq!(
            summed.cov_delta_jsd(),
            summed.std_delta_jsd() / summed.mean_delta_jsd()
        );
    }
}
