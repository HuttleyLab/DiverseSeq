use crate::zarr_io::ZarrStore;
use std::cell::Cell;

/// Coefficients for multi-dimensional coordinate conversion into 1D index.
fn coord_conversion_coeffs(num_states: usize, k: usize) -> Vec<usize> {
    (1..=k)
        .rev()
        .map(|i| num_states.pow((i - 1) as u32))
        .collect()
}

// converts k-mer into single integer
fn kmer_to_index(kmer: &[u8], num_states: usize, coeffs: &[usize], max_index: usize) -> usize {
    let mut index: usize = 0;
    for (i, &byte) in kmer.iter().enumerate() {
        if byte as usize >= num_states {
            index = max_index;
            break;
        } else {
            index += coeffs[i] * (byte as usize);
        }
    }
    index
}

fn count_monomers(seq: &[u8], num_states: usize) -> Vec<usize> {
    let mut counts = vec![0usize; num_states];
    for &state in seq.iter() {
        if state < num_states as u8 {
            counts[state as usize] += 1;
        }
    }
    counts
}

fn count_kmers(seq: &[u8], num_states: usize, k: usize) -> Vec<usize> {
    let coeffs = coord_conversion_coeffs(num_states, k);
    let size: usize = num_states.pow(k as u32);
    let mut counts = vec![0usize; size];
    let mut skip_until = 0;

    for (i, state) in seq.iter().take(k).enumerate() {
        if (*state as usize) >= num_states {
            skip_until = i + 1
        }
    }

    let mut index: isize = -1;
    let nstates = num_states as u8;
    let biggest_coeff = coeffs[0] as isize;

    for (i, window) in seq.windows(k).enumerate() {
        let gained_char = seq[i + k - 1];
        if gained_char >= nstates {
            // we reset the kmer index to invalid
            // until we get a proper one
            index = -1;
            skip_until = i + k;
        }
        if i < skip_until {
            continue;
        }

        if index < 0 {
            index = kmer_to_index(window, num_states, &coeffs, size - 1) as isize;
        } else {
            let dropped_char = seq[i - 1] as isize;
            index =
                (index - dropped_char * biggest_coeff) * num_states as isize + gained_char as isize;
        }

        if index < 0 {
            continue;
        } else {
            counts[index as usize] += 1;
        }
    }
    counts
}

pub fn entropy(kfreqs: &Vec<f64>) -> f64 {
    if kfreqs.is_empty() {
        let entropy = f64::NAN;
        return entropy;
    }
    let mut entropy: f64 = 0.0;
    let mut total_freq: f64 = 0.0;
    for freq in kfreqs.iter() {
        if *freq == 0.0 {
            continue;
        }
        entropy += -*freq * freq.log2();
        total_freq += *freq;
    }
    if (total_freq - 1.0).abs() > f64::EPSILON {
        entropy = f64::NAN;
    }
    entropy
}

pub struct SeqRecord<'a> {
    pub seqid: &'a str,
    pub seq: &'a [u8],
    pub num_states: usize,
}

impl<'a> SeqRecord<'a> {
    pub fn new(seqid: &'a str, seq: &'a [u8], num_states: usize) -> Self {
        Self {
            seqid,
            seq,
            num_states,
        }
    }

    pub fn to_kmerseq(&self, k: usize) -> Result<KmerSeq, Box<dyn std::error::Error>> {
        let kcounts = match k {
            0 => panic!("k cannot be 0"),
            1 => count_monomers(self.seq, self.num_states),
            _ => count_kmers(self.seq, self.num_states, k),
        };

        let total = kcounts.iter().sum::<usize>() as f64;
        if total == 0.0 {
            return Err(format!("No valid k-mers for '{}'", self.seqid).into());
        }
        let kfreqs: Vec<f64> = kcounts.iter().map(|x| *x as f64 / total).collect();
        Ok(KmerSeq::new(self.seqid, kfreqs, self.num_states, k))
    }
}

#[derive(Debug, Clone)]
pub struct KmerSeq {
    pub seqid: String,
    pub kfreqs: Vec<f64>,
    pub entropy: f64,
    // a Cell because they are mutable, even when
    // their container is not
    pub delta_jsd: Cell<f64>,
    pub num_states: usize,
    pub k: usize,
}

impl KmerSeq {
    pub fn new(seqid: &str, kfreqs: Vec<f64>, num_states: usize, k: usize) -> Self {
        let delta_jsd: Cell<f64> = Cell::new(0.0);
        let entropy: f64 = entropy(&kfreqs);
        Self {
            seqid: seqid.to_string(),
            kfreqs,
            entropy,
            delta_jsd,
            num_states,
            k,
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            seqid: self.seqid.to_string(),
            kfreqs: self.kfreqs.clone(),
            entropy: self.entropy,
            delta_jsd: 0.0.into(),
            num_states: self.num_states,
            k: self.k,
        }
    }
}

pub struct LazySeqRecord<'a> {
    pub seqid: &'a String,
    pub num_states: usize,
    storage: &'a ZarrStore,
}

impl<'a> LazySeqRecord<'a> {
    pub fn new(seqid: &'a String, num_states: usize, storage: &'a ZarrStore) -> Self {
        Self {
            seqid,
            num_states,
            storage,
        }
    }

    pub fn to_kmerseq(&self, k: usize) -> Result<KmerSeq, Box<dyn std::error::Error>> {
        let seq = self.storage.read_uint8_array(&self.seqid).unwrap();
        let sr = SeqRecord::new(&self.seqid, &seq, self.num_states);
        sr.to_kmerseq(k)
    }
}

pub fn make_seq_records<'a>(
    seqids: &'a [String],
    seqs: &'a [Vec<u8>],
    num_states: usize,
) -> Vec<SeqRecord<'a>> {
    seqids
        .iter()
        .zip(seqs.iter())
        .map(|(id, seq)| SeqRecord::new(id.as_str(), seq.as_slice(), num_states))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn maximum_entropy() {
        // zero element added too
        let freqs = Vec::from([0.25, 0.0, 0.25, 0.25, 0.25]);
        let entropy = entropy(&freqs);
        assert_eq!(entropy, 2.0);
    }

    #[rstest]
    #[case(vec![0.0, 0.0, 0.0, 0.0])]
    #[case(vec![])]
    #[case(vec![0.9, 0.9])]
    #[case(vec![1.9, 0.0])]
    fn nan_entropy(#[case] freqs: Vec<f64>) {
        let entropy = entropy(&freqs);
        assert!(entropy.is_nan());
    }

    #[rstest]
    #[case([2, 1], 9)]
    #[case([0, 0], 0)]
    #[case([3, 3], 15)]
    #[case([4, 3], 16)]
    #[case([4, 4], 16)]
    fn kmer_to_index_valid(#[case] seq: [u8; 2], #[case] expected: usize) {
        let coeffs = coord_conversion_coeffs(4, 2);
        let index = kmer_to_index(&seq, 4, &coeffs, 16);
        assert_eq!(index, expected);
    }

    #[test]
    fn kmer_count() {
        let seq: [u8; 21] = [
            2, 5, 1, 5, 0, 0, 2, 1, 0, 0, 3, 0, 0, 3, 1, 0, 2, 1, 1, 5, 1,
        ];
        let kcounts = count_kmers(&seq, 4, 2);
        let expect = Vec::<usize>::from([3, 0, 2, 2, 2, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0]);
        assert_eq!(kcounts, expect);
    }

    #[test]
    fn seq_record_to_kmerseq_invalidk() {
        let seqid = String::from("blah");
        let seq: [u8; 4] = [0, 1, 2, 0];
        let sr = SeqRecord::new(&seqid, &seq, 4);
        let result = std::panic::catch_unwind(|| sr.to_kmerseq(0));
        assert!(result.is_err());
    }

    #[test]
    fn seq_record_to_kmerseq_keq1() {
        let seqid = String::from("blah");
        let seq: [u8; 6] = [0, 1, 2, 0, 0, 1];
        let sr = SeqRecord::new(&seqid, &seq, 4);
        let kseq = sr.to_kmerseq(1).unwrap();
        assert_eq!(kseq.seqid, "blah");
        assert_eq!(kseq.num_states, 4);
        assert_eq!(kseq.k, 1);
        let expect = Vec::<f64>::from([3.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 0.0]);
        assert_eq!(kseq.kfreqs, expect);
    }

    #[test]
    fn seq_record_to_kmerseq_keq2() {
        let seqid = String::from("blah");
        let seq: [u8; 6] = [0, 1, 2, 0, 0, 1];
        let sr = SeqRecord::new(&seqid, &seq, 4);
        let kseq = sr.to_kmerseq(2).unwrap();
        assert_eq!(kseq.seqid, "blah");
        assert_eq!(kseq.num_states, 4);
        assert_eq!(kseq.k, 2);
        let expect = Vec::<f64>::from([
            0.2, 0.4, 0., 0., 0., 0., 0.2, 0., 0.2, 0., 0., 0., 0., 0., 0., 0.,
        ]);
        assert_eq!(kseq.kfreqs, expect);
    }

    #[test]
    fn no_data_seq_record_to_kmerseq() {
        let seqid = String::from("blah");
        let num_states: u8 = 4;
        let seq: [u8; 4] = [num_states, num_states, num_states, num_states];
        let sr = SeqRecord::new(&seqid, &seq, num_states as usize);
        let result = sr.to_kmerseq(1);
        assert!(result.is_err());
    }

    #[test]
    fn check_make_records() {
        let seqids: Vec<String> = vec![
            String::from("seq1"),
            String::from("seq2"),
            String::from("seq3"),
        ];
        let seqs: Vec<Vec<u8>> = vec![vec![0, 1, 2, 1], vec![0, 1, 2, 2, 3], vec![3, 0, 0]];
        let records = make_seq_records(&seqids, &seqs, 4);
        assert_eq!(records[0].seqid, "seq1");
        assert_eq!(records[1].seqid, "seq2");
        assert_eq!(records[2].seqid, "seq3");
        assert_eq!(records[0].num_states, 4);
        assert_eq!(records[1].num_states, 4);
        assert_eq!(records[2].num_states, 4);
        assert_eq!(records[0].seq, &[0, 1, 2, 1]);
        assert_eq!(records[1].seq, &[0, 1, 2, 2, 3]);
        assert_eq!(records[2].seq, &[3, 0, 0]);
    }
}
