use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use dashmap::{DashMap, DashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

fn iter_trigram_keys(data: &[u8]) -> impl Iterator<Item = u32> + '_ {
    data.windows(3)
        .map(|w| ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32))
}

#[derive(Default, Serialize, Deserialize)]
struct ByteTrigramIndexCore {
    postings: DashMap<u32, RoaringBitmap>,
    doc_ids: DashSet<u32>,
}

impl ByteTrigramIndexCore {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, doc_id: u32, data: &[u8]) -> Result<(), String> {
        if data.len() < 3 {
            return Err("data must be at least 3 bytes".to_string());
        }

        if !self.doc_ids.insert(doc_id) {
            return Err(format!("duplicate doc_id: {}", doc_id));
        }

        let seen: DashSet<u32> = iter_trigram_keys(data).collect();
        for tg in seen {
            self.postings.entry(tg).or_default().insert(doc_id);
        }

        Ok(())
    }

    fn trigram_df(&self, tg: u32) -> u64 {
        self.postings.get(&tg).map(|bm| bm.len()).unwrap_or(0)
    }

    fn rarest_query_trigrams(&self, needle: &[u8]) -> Vec<u32> {
        let mut uniq: Vec<u32> = {
            let set: DashSet<u32> = iter_trigram_keys(needle).collect();
            set.into_iter().collect()
        };
        uniq.sort_by_key(|&tg| self.trigram_df(tg));
        uniq
    }

    fn candidate_docs(&self, needle: &[u8]) -> RoaringBitmap {
        let tgs = self.rarest_query_trigrams(needle);
        if tgs.is_empty() {
            return RoaringBitmap::new();
        }

        let first = match self.postings.get(&tgs[0]) {
            Some(bm) => bm,
            None => return RoaringBitmap::new(),
        };

        let mut result = first.clone();

        for tg in &tgs[1..] {
            match self.postings.get(tg) {
                Some(other) => {
                    result &= other.value();
                    if result.is_empty() {
                        break;
                    }
                }
                None => return RoaringBitmap::new(),
            }
        }

        result
    }

    fn search_candidate_topk(&self, needle: &[u8], k: usize) -> Vec<u32> {
        self.candidate_docs(needle).iter().take(k).collect()
    }

    fn save_to_path(&self, path: &str) -> Result<(), String> {
        let bytes = postcard::to_allocvec(self).map_err(|e| e.to_string())?;
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())
    }

    fn load_from_path(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        postcard::from_bytes(&bytes).map_err(|e| e.to_string())
    }
}

#[pyclass]
struct TrigramIndex {
    inner: ByteTrigramIndexCore,
}

#[pymethods]
impl TrigramIndex {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: ByteTrigramIndexCore::new(),
        }
    }

    fn add(&mut self, doc_id: u32, data: &[u8]) -> PyResult<()> {
        self.inner.add(doc_id, data).map_err(PyValueError::new_err)
    }

    fn search(&self, py: Python<'_>, needle: &[u8], k: usize) -> PyResult<Vec<u32>> {
        if needle.len() < 3 {
            return Err(PyValueError::new_err("needle must be at least 3 bytes"));
        }
        Ok(py.allow_threads(|| self.inner.search_candidate_topk(needle, k)))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_path(path).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = ByteTrigramIndexCore::load_from_path(path)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
}

#[pymodule]
fn roaring_trigram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TrigramIndex>()?;
    Ok(())
}