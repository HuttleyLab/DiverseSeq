use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_64;
use zarrs::array::codec::ZstdCodec;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;

/// A wrapper around a zarr directory store that manages uint8 array members with zstd compression
pub struct ZarrStore {
    path: PathBuf,
    store: Arc<FilesystemStore>,
    seqid_to_hash: FxHashMap<String, [u8; 16]>,
    root: String,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    seqid_to_hash: Vec<(String, [u8; 16])>,
}

impl ZarrStore {
    /// Create a new ZarrStore at the specified directory path
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.into();
        std::fs::create_dir_all(&path)?;
        let store = Arc::new(FilesystemStore::new(&path)?);
        let seqid_to_hash = Self::load_metadata(&path);

        let root = "/group".to_string();
        zarrs::group::GroupBuilder::new()
            .build(store.clone(), &root)?
            .store_metadata()?;

        Ok(ZarrStore {
            path,
            store,
            seqid_to_hash,
            root,
        })
    }

    /// Get the path to the zarr directory
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Load metadata from disk
    fn load_metadata(path: &PathBuf) -> FxHashMap<String, [u8; 16]> {
        let metadata_path = path.join(".seqid_to_hash.bin");
        if let Ok(contents) = std::fs::read(&metadata_path)
            && let Ok(metadata) = postcard::from_bytes::<Metadata>(&contents)
        {
            return metadata.seqid_to_hash.into_iter().collect();
        }
        FxHashMap::default()
    }

    /// Save metadata to disk atomically
    fn save_metadata(&self) -> Result<(), Box<dyn std::error::Error>> {
        let metadata_path = self.path.join(".seqid_to_hash.bin");
        let temp_path = self.path.join(".seqid_to_hash.bin.tmp");

        // Convert HashMap to Vec for serialization
        let metadata = Metadata {
            seqid_to_hash: self
                .seqid_to_hash
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        };

        // Serialize using postcard
        let encoded = postcard::to_allocvec(&metadata)?;
        let mut file = std::fs::File::create(&temp_path)?;
        file.write_all(&encoded)?;
        file.sync_all()?; // Ensure data is written to disk

        // Atomically replace the old file
        std::fs::rename(&temp_path, &metadata_path)?;

        Ok(())
    }

    /// Add a new uint8 array member with compression
    ///
    /// # Arguments
    /// * `name` - Name of the array member
    /// * `data` - The uint8 data to store
    ///
    /// # Note
    /// No file handle is retained after this call. The zarr store opens a handle
    /// only during the write operation and closes it afterward.
    pub fn add_uint8_array(
        &mut self,
        seqid: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.contains(seqid) {
            return Ok(());
        }

        // Compute xxhash 64-bit hash of the data
        let hash = xxh3_64(data);
        let hexdigest = format!("{:016x}", hash);
        let hex_bytes: [u8; 16] = hexdigest.as_bytes().try_into()?;

        // Store the mapping from name to hash
        self.seqid_to_hash.insert(seqid.to_string(), hex_bytes);

        let array_path = format!("{}/{}", self.root, hexdigest);
        // Build array with zstd compression
        let codec = ZstdCodec::new(10, true);
        let data_len = data.len() as u64;
        let chunk_size = std::cmp::min(data_len, 1024); // Use 1024-byte chunks

        let array = ArrayBuilder::new(
            vec![data_len],
            vec![chunk_size],
            DataType::UInt8,
            FillValue::from(0u8),
        )
        .bytes_to_bytes_codecs(vec![Arc::new(codec)])
        .build(self.store.clone(), &array_path)
        .expect("Failed to build array");

        // Write the data
        array
            .store_array_subset_elements::<u8>(&array.subset_all(), data)
            .expect("Failed to write data");

        // Store array metadata to disk
        array
            .store_metadata()
            .expect("Failed to store array metadata");

        Ok(())
    }

    /// Read a uint8 array member
    ///
    /// # Arguments
    /// * `name` - Name of the array member to read
    ///
    /// # Returns
    /// A tuple of (data, shape)
    pub fn read_uint8_array(&self, seqid: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let hex_bytes = self.seqid_to_hash[seqid];
        let hexdigest = String::from_utf8(hex_bytes.to_vec())?;
        let array_path = format!("{}/{}", self.root, hexdigest);
        let array = zarrs::array::Array::open(self.store.clone(), &array_path)
            .expect("failed to build array for reading");

        let data = array.retrieve_array_subset_elements::<u8>(&array.subset_all())?;

        Ok(data)
    }

    /// Checks if an array member exists in the store
    ///
    /// # Arguments
    /// * `name` - Name of the array member to check
    ///
    /// # Returns
    /// `true` if the array exists, `false` otherwise
    pub fn contains(&self, seqid: &str) -> bool {
        self.seqid_to_hash.contains_key(seqid)
    }

    /// List all array members in the store
    pub fn list_hexdigests(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        use std::collections::HashSet;

        let unique_hashes: HashSet<String> = self
            .seqid_to_hash
            .values()
            .map(|hex_bytes| {
                String::from_utf8(hex_bytes.to_vec()).expect("hex bytes are always valid UTF-8")
            })
            .collect();

        Ok(unique_hashes.into_iter().collect())
    }

    /// List all seqid's in the store
    pub fn list_seqids(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(self.seqid_to_hash.keys().cloned().collect())
    }
}

impl Drop for ZarrStore {
    fn drop(&mut self) {
        // Save metadata when the store is dropped
        if let Err(e) = self.save_metadata() {
            eprintln!("Warning: Failed to save metadata on drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    #[fixture]
    fn temp_dir() -> TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    #[rstest]
    fn test_create_store(temp_dir: TempDir) {
        let store = ZarrStore::new(temp_dir.path());
        assert!(store.is_ok());
    }

    #[fixture]
    fn a_store(temp_dir: TempDir) -> ZarrStore {
        ZarrStore::new(temp_dir.path()).expect("failed to create store")
    }

    #[rstest]
    fn test_write_seq(mut a_store: ZarrStore) {
        let name = "s1";
        let r = a_store.add_uint8_array(&"s1", &[0, 3, 1, 0]);
        assert!(r.is_ok());
        assert!(a_store.seqid_to_hash.contains_key(name));
        assert!(a_store.contains(name));
        // it's already present, so it will still return Ok
        let r = a_store.add_uint8_array(&"s1", &[0, 3, 1, 0]);
        assert!(r.is_ok());
        let got = a_store.read_uint8_array(name);
        assert!(got.is_ok());
    }

    #[fixture]
    fn b_store(mut a_store: ZarrStore) -> PathBuf {
        let _ = a_store.add_uint8_array(&"s1", &[0, 3, 1, 0]);
        let _ = a_store.add_uint8_array(&"s2", &[0, 3, 1, 0]);
        let path = a_store.path().clone();
        drop(a_store); // Explicitly close without deleting
        path
    }

    #[rstest]
    fn test_reload_store(b_store: PathBuf) {
        let name = "s1";
        let reopened_store = ZarrStore::new(&b_store).expect("failed to reopen store");
        assert!(reopened_store.contains(name));
        let got = reopened_store.read_uint8_array(name);
        assert!(got.is_ok());
        assert_eq!(got.unwrap(), &[0, 3, 1, 0]);
    }

    #[rstest]
    fn test_reloaded_store_seqids(b_store: PathBuf) {
        let reopened_store = ZarrStore::new(&b_store).expect("failed to reopen store");
        assert_eq!(reopened_store.list_seqids().unwrap(), vec!["s1", "s2"]);
        assert_eq!(reopened_store.list_hexdigests().unwrap().len(), 1);
    }
}
