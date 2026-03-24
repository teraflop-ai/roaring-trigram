use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use hashbrown::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

fn iter_trigram_keys(data: &[u8]) -> impl Iterator<Item = u32> + '_ {
    data.windows(3)
        .map(|w| ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32))
}

#[derive(Default, Serialize, Deserialize)]
struct ByteTrigramIndexCore {
    postings: HashMap<u32, RoaringBitmap>,
    doc_ids: HashSet<u32>,
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

        let seen: HashSet<u32> = iter_trigram_keys(data).collect();
        for tg in seen {
            self.postings.entry(tg).or_default().insert(doc_id);
        }

        Ok(())
    }

    fn optimize(&mut self) -> usize {
        let mut changed = 0;
        for (_tg, bm) in self.postings.iter_mut() {
            if bm.optimize() {
                changed += 1;
            }
        }
        changed
    }

    fn trigram_df(&self, tg: u32) -> u64 {
        self.postings.get(&tg).map(|bm| bm.len()).unwrap_or(0)
    }

    fn rarest_query_trigrams_k(&self, needle: &[u8], k: usize) -> Vec<u32> {
        let set: HashSet<u32> = iter_trigram_keys(needle).collect();
        let mut uniq: Vec<u32> = set.into_iter().collect();

        if uniq.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(uniq.len());

        if uniq.len() > k {
            uniq.select_nth_unstable_by_key(k - 1, |&tg| self.trigram_df(tg));
            uniq.truncate(k);
        }

        uniq.sort_unstable_by_key(|&tg| self.trigram_df(tg));
        uniq
    }

    fn candidate_docs(&self, needle: &[u8], max_trigrams: usize) -> RoaringBitmap {
        if max_trigrams == 0 {
            return RoaringBitmap::new();
        }

        let tgs = self.rarest_query_trigrams_k(needle, max_trigrams);
        if tgs.is_empty() {
            return RoaringBitmap::new();
        }

        for tg in &tgs {
            if !self.postings.contains_key(tg) {
                return RoaringBitmap::new();
            }
        }

        let mut result = self.postings[&tgs[0]].clone();

        for tg in &tgs[1..] {
            result &= &self.postings[tg];
            if result.is_empty() {
                break;
            }
        }

        result
    }

    fn search_candidate_topk(&self, needle: &[u8], k: usize, max_trigrams: usize) -> Vec<u32> {
        self.candidate_docs(needle, max_trigrams)
            .iter()
            .take(k)
            .collect()
    }

    fn save_to_path(&mut self, path: &str) -> Result<(), String> {
        self.optimize();

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

    fn doc_count(&self) -> usize {
        self.doc_ids.len()
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

    #[pyo3(signature = (needle, k, max_trigrams=4))]
    fn search(
        &self,
        py: Python<'_>,
        needle: &[u8],
        k: usize,
        max_trigrams: usize,
    ) -> PyResult<Vec<u32>> {
        if needle.len() < 3 {
            return Err(PyValueError::new_err("needle must be at least 3 bytes"));
        }
        if max_trigrams == 0 {
            return Err(PyValueError::new_err("max_trigrams must be >= 1"));
        }

        Ok(py.allow_threads(|| {
            self.inner.search_candidate_topk(needle, k, max_trigrams)
        }))
    }

    fn optimize(&mut self) -> usize {
        self.inner.optimize()
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner.save_to_path(path).map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = ByteTrigramIndexCore::load_from_path(path)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn doc_count(&self) -> usize {
        self.inner.doc_count()
    }
}

#[pymodule]
fn roaring_trigram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TrigramIndex>()?;
    Ok(())
}