use crate::zarr_io::{Storage, ZarrStore};
use pyo3::Python;
use pyo3::prelude::{Bound, PyErr, PyResult, pyclass, pymethods};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyTuple};
use rustc_hash::FxHashMap;
use std::path::PathBuf;

#[pyclass(module = "diverse_seq._dvs")]
pub struct ZarrStoreWrapper {
    pub store: ZarrStore,
    #[pyo3(get)]
    pub source: String,
}

#[pymethods]
impl ZarrStoreWrapper {
    #[new]
    pub fn new(path: Option<String>) -> PyResult<Self> {
        let source = match path {
            Some(ref p) => p,
            None => &"".to_string(),
        };

        let path: Option<PathBuf> = match path {
            Some(ref p) => Some(PathBuf::from(p)),
            None => None,
        };
        let store = ZarrStore::new(path);
        if !store.is_ok() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create ZarrStore: {:?}",
                source
            )));
        }
        Ok(Self {
            store: store.unwrap(),
            source: source.to_string(),
        })
    }

    pub fn __repr__(&self) -> String {
        let source = if self.source.len() == 0 {
            "'in memory'"
        } else {
            &self.source
        };
        format!(
            "ZarrStoreWrapper(source={}, num members={})",
            source,
            self.__len__()
        )
    }

    pub fn __contains__(&self, key: &str) -> bool {
        self.store.contains_seqid(key)
    }

    pub fn __len__(&self) -> usize {
        self.store.list_seqids().unwrap().len()
    }

    #[getter]
    pub fn source(&self) -> String {
        self.store.path().to_string_lossy().to_string()
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if matches!(self.store.store, Storage::Memory(_)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot pickle in-memory store",
            ));
        }

        let path_str = self.store.path().to_string_lossy().to_string();

        // Save metadata to disk before pickling
        self.store.save_metadata().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to save metadata: {:?}",
                e
            ))
        })?;

        let state = PyDict::new(py);
        state.set_item("path", path_str)?;
        Ok(state)
    }

    fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        let path: String = state
            .get_item("path")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("path"))?
            .extract()?;

        let path_buf = std::path::PathBuf::from(path);
        let store = ZarrStore::new(Some(path_buf.clone())).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to unpickle: {:?}",
                e
            ))
        })?;

        self.store = store;

        Ok(())
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &[self.store.path().to_string_lossy().to_string()])
    }

    #[pyo3(signature = (seqid, seq, metadata=None))]
    pub fn write(
        &mut self,
        py: Python<'_>,
        seqid: &str,
        seq: &[u8],
        metadata: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let metadata = match metadata {
            Some(m) => m,
            None => {
                let default = PyDict::new(py);
                default.set_item("source", "unknown")?;
                default
            }
        };
        // Convert PyDict to HashMap
        let metadata_map: FxHashMap<String, String> = metadata
            .iter()
            .map(|(key, value)| {
                let k: String = key.extract()?;
                let v: String = value.extract()?;
                Ok((k, v))
            })
            .collect::<PyResult<FxHashMap<String, String>>>()?;

        let result = self.store.add_uint8_array(seqid, seq, Some(metadata_map));
        if !result.is_ok() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create add {}",
                seqid
            )));
        }
        Ok(())
    }

    #[pyo3(signature = (unique_id, data))]
    pub fn write_log(&self, unique_id: String, data: String) {}

    #[pyo3(signature = (seqid))]
    pub fn read(&self, seqid: &str) -> PyResult<Vec<u8>> {
        let result = self.store.read_uint8_array(seqid);
        if !result.is_ok() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create add {}",
                seqid
            )));
        }
        Ok(result.unwrap())
    }

    #[pyo3(signature = (seqid))]
    pub fn read_metadata<'py>(&self, py: Python<'py>, seqid: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.store.read_metadata(seqid);
        if let Err(e) = result {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to read metadata for {}: {}",
                seqid, e
            )));
        }

        let map = result.unwrap();
        let dict = PyDict::new(py);
        for (k, v) in map {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// returns the number of unique sequences in the store
    pub fn num_unique(&self) -> usize {
        self.store.list_hexdigests().unwrap().len()
    }

    #[getter]
    /// returns the names of unique sequences in the store
    pub fn unique_seqids(&self) -> PyResult<Vec<String>> {
        Ok(self.store.list_unique_seqids().unwrap())
    }

    /// returns the names of unique sequences in the store
    pub fn get_seqids(&self) -> PyResult<Vec<String>> {
        Ok(self.store.list_seqids().unwrap())
    }
}
