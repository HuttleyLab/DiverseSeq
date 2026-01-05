use pyo3::prelude::{PyResult, pyfunction};
use std::collections::{BinaryHeap, HashSet};

/// Take the reverse complement of a kmer.
///
/// Assumes cogent3 DNA/RNA sequences (numerical
/// representation for complement offset by 2
/// from original).
///
/// # Arguments
///
/// * `kmer` - The kmer to attain the reverse complement of
///
/// # Returns
///
/// The reverse complement of a kmer.
pub fn reverse_complement(kmer: &[u8]) -> Vec<u8> {
    kmer.iter().map(|&base| (base + 2) % 4).rev().collect()
}

pub fn murmurhash3_32(data: &[u8], seed: u32) -> u32 {
    const DEFAULT_SEED: u32 = 0x9747B28C;

    let seed = if seed == 0 { DEFAULT_SEED } else { seed };
    let length = data.len() as u32;
    let mut h = seed ^ length;

    for &value in data.iter() {
        let mut k = value as u32;

        // Mix the hash
        k = k.wrapping_mul(0xCC9E2D51);
        k = k.rotate_left(15); // Rotate left by 15 bits
        k = k.wrapping_mul(0x1B873593);

        h ^= k;
        h = h.rotate_left(13); // Rotate left by 13 bits
        h = h.wrapping_mul(5).wrapping_add(0xE6546B64);
    }

    // Finalization
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;

    h // Already 32-bit, no need to mask
}

/// Hash a kmer, optionally use the mash canonical representation.
///
/// # Arguments
///
/// * `kmer` - The kmer to hash.
/// * `mash_canonical` - Whether to use the mash canonical representation for a kmer.
///
/// # Returns
///
/// The hash of a kmer.
///
/// # Notes
///
/// Uses MurmurHash3 32-bit implementation.
pub fn hash_kmer(kmer: &[u8], mash_canonical: bool) -> u32 {
    let smallest = if mash_canonical {
        let reverse = reverse_complement(kmer);

        // Compare kmer with its reverse complement
        for i in 0..kmer.len() {
            if kmer[i] < reverse[i] {
                break; // kmer is smaller, use original
            }
            if kmer[i] > reverse[i] {
                // reverse is smaller, use it
                return murmurhash3_32(&reverse, 0);
            }
        }

        // If we get here, they're equal or kmer is smaller
        kmer
    } else {
        kmer
    };

    murmurhash3_32(smallest, 0)
}

/// Get the kmer hashes comprising a sequence.
///
/// # Arguments
///
/// * `seq` - A sequence of uint8
/// * `k` - kmer size.
/// * `num_states` - Number of states allowed for sequence type.
/// * `mash_canonical` - Whether to use the mash canonical representation of kmers.
///
/// # Returns
///
/// kmer hashes for the sequence.
pub fn get_kmer_hashes(seq: &[u8], k: usize, num_states: u8, mash_canonical: bool) -> Vec<u32> {
    if seq.len() < k {
        return Vec::new();
    }

    let mut kmer_hashes = Vec::with_capacity(seq.len() - k + 1);

    // Find initial skip position
    let mut skip_until = 0;
    for i in 0..k {
        if seq[i] >= num_states {
            skip_until = i + 1;
        }
    }

    // Process each kmer using windows
    for (i, kmer) in seq.windows(k).enumerate() {
        // Check if last position of current kmer is invalid
        if kmer[k - 1] >= num_states {
            skip_until = i + k;
        }

        // Skip invalid kmers
        if i < skip_until {
            continue;
        }

        // Hash valid kmer
        let hash = hash_kmer(kmer, mash_canonical);
        kmer_hashes.push(hash);
    }

    kmer_hashes
}

#[pyfunction]
#[pyo3(signature = (seq_array, k, sketch_size, num_states=4, mash_canonical=false))]
/// Find the mash sketch for a sequence array.
///
/// # Arguments
///
/// * `seq_array` - The sequence array to find the sketch for.
/// * `k` - kmer size.
/// * `sketch_size` - Size of the sketch.
/// * `num_states` - Number of possible states (e.g. GCAT gives 4 for DNA).
/// * `mash_canonical` - Whether to use the mash canonical representation of kmers.
///
/// # Returns
///
/// The bottom sketch for the given sequence array.
pub fn mash_sketch(
    seq_array: &[u8],
    k: usize,
    sketch_size: usize,
    num_states: u8,
    mash_canonical: bool,
) -> PyResult<Vec<u32>> {
    // Get unique kmer hashes
    let kmer_hashes = get_kmer_hashes(seq_array, k, num_states, mash_canonical);
    let unique_hashes: HashSet<u32> = kmer_hashes.into_iter().collect();

    // Use a max-heap to keep the smallest sketch_size elements
    // BinaryHeap is a max-heap by default, so we store values directly
    let mut heap = BinaryHeap::new();

    for kmer_hash in unique_hashes {
        if heap.len() < sketch_size {
            heap.push(kmer_hash);
        } else if let Some(&max) = heap.peek() {
            if kmer_hash < max {
                heap.pop();
                heap.push(kmer_hash);
            }
        }
    }

    // Convert heap to sorted vector
    let mut result: Vec<u32> = heap.into_iter().collect();
    result.sort_unstable();

    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_reverse_complement() {
        let kmer = vec![0, 1, 2, 3];
        let rev_comp = super::reverse_complement(&kmer);
        assert_eq!(rev_comp, vec![1, 0, 3, 2]);
    }
}
