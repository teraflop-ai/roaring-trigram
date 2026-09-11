use hashbrown::{HashMap, HashSet};
use memmap2::{Mmap, MmapOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use roaring::RoaringBitmap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

const MAGIC: &[u8; 8] = b"BTGIDX01";
const HEADER: usize = 24;
const ROW: usize = 28;
const MAX_KEYS: u64 = 1 << 24;

fn trigrams(data: &[u8]) -> impl Iterator<Item = u32> + '_ {
    data.windows(3)
        .map(|w| u32::from_be_bytes([0, w[0], w[1], w[2]]))
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

#[derive(Clone, Copy)]
struct Entry {
    trigram: u32,
    offset: u64,
    length: u64,
    df: u64,
}

#[pyclass]
#[derive(Default)]
struct TrigramIndexBuilder {
    postings: HashMap<u32, RoaringBitmap>,
    doc_ids: HashSet<u32>,
}

#[pymethods]
impl TrigramIndexBuilder {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, doc_id: u32, data: &[u8]) -> PyResult<()> {
        if data.len() < 3 {
            return Err(PyValueError::new_err("data must be at least 3 bytes"));
        }
        if !self.doc_ids.insert(doc_id) {
            return Err(PyValueError::new_err(format!("duplicate doc_id: {doc_id}")));
        }
        for tg in trigrams(data).collect::<HashSet<_>>() {
            self.postings.entry(tg).or_default().insert(doc_id);
        }
        Ok(())
    }

    fn save(&mut self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        Ok(py.detach(|| self.save_snapshot(&path))?)
    }

    fn doc_count(&self) -> usize {
        self.doc_ids.len()
    }
}

impl TrigramIndexBuilder {
    fn save_snapshot(&mut self, path: &Path) -> io::Result<()> {
        for bitmap in self.postings.values_mut() {
            bitmap.optimize();
        }
        let mut postings: Vec<_> = self.postings.iter().collect();
        postings.sort_unstable_by_key(|&(tg, _)| *tg);

        let parent = path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temp = NamedTempFile::new_in(parent)?;
        {
            let mut writer = BufWriter::new(temp.as_file_mut());
            writer.write_all(MAGIC)?;
            writer.write_all(&(self.doc_ids.len() as u64).to_le_bytes())?;
            writer.write_all(&(postings.len() as u64).to_le_bytes())?;

            let mut offset = HEADER as u64 + postings.len() as u64 * ROW as u64;
            for &(tg, bitmap) in &postings {
                let length = bitmap.serialized_size() as u64;
                writer.write_all(&tg.to_le_bytes())?;
                writer.write_all(&offset.to_le_bytes())?;
                writer.write_all(&length.to_le_bytes())?;
                writer.write_all(&bitmap.len().to_le_bytes())?;
                offset = offset.checked_add(length).ok_or_else(|| invalid("index too large"))?;
            }
            for (_, bitmap) in postings {
                bitmap.serialize_into(&mut writer)?;
            }
            writer.flush()?;
        }
        temp.as_file().sync_all()?;
        temp.persist(path).map_err(|e| e.error)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    }
}

#[pyclass(frozen)]
struct TrigramIndex {
    mmap: Mmap,
    entries: usize,
    docs: u64,
}

#[pymethods]
impl TrigramIndex {
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Ok(py.detach(|| unsafe { Self::open_snapshot(&path) })?)
    }

    #[pyo3(signature = (needle, k, max_trigrams=4))]
    fn search(&self, py: Python<'_>, needle: &[u8], k: usize, max_trigrams: usize) -> PyResult<Vec<u32>> {
        if needle.len() < 3 {
            return Err(PyValueError::new_err("needle must be at least 3 bytes"));
        }
        if max_trigrams == 0 {
            return Err(PyValueError::new_err("max_trigrams must be >= 1"));
        }
        Ok(py.detach(|| self.candidates(needle, k, max_trigrams))?)
    }

    fn doc_count(&self) -> u64 {
        self.docs
    }
}

impl TrigramIndex {
    unsafe fn open_snapshot(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() < HEADER || &mmap[..8] != MAGIC {
            return Err(invalid("invalid index header"));
        }
        let docs = u64_le(&mmap[8..]);
        let count = u64_le(&mmap[16..]);
        if count > MAX_KEYS || docs > (1u64 << 32) || (count == 0) != (docs == 0) {
            return Err(invalid("invalid index counts"));
        }
        let entries = count as usize;
        let start = HEADER + entries * ROW;
        if start > mmap.len() {
            return Err(invalid("truncated directory"));
        }
        let index = Self { mmap, entries, docs };
        let mut previous = None;
        let mut end = start as u64;
        for i in 0..entries {
            let entry = index.entry(i);
            if u64::from(entry.trigram) >= MAX_KEYS
                || previous.is_some_and(|tg| tg >= entry.trigram)
                || entry.offset != end
                || entry.length == 0
                || entry.df == 0
                || entry.df > docs
            {
                return Err(invalid("invalid directory entry"));
            }
            end = end.checked_add(entry.length).ok_or_else(|| invalid("offset overflow"))?;
            if end > index.mmap.len() as u64 {
                return Err(invalid("truncated posting"));
            }
            previous = Some(entry.trigram);
        }
        if end != index.mmap.len() as u64 {
            return Err(invalid("unexpected trailing bytes"));
        }
        Ok(index)
    }

    fn entry(&self, i: usize) -> Entry {
        let row = &self.mmap[HEADER + i * ROW..][..ROW];
        Entry {
            trigram: u32::from_le_bytes(row[..4].try_into().unwrap()),
            offset: u64_le(&row[4..]),
            length: u64_le(&row[12..]),
            df: u64_le(&row[20..]),
        }
    }

    fn find(&self, trigram: u32) -> Option<Entry> {
        let (mut low, mut high) = (0, self.entries);
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.entry(mid);
            match entry.trigram.cmp(&trigram) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(entry),
            }
        }
        None
    }

    fn read_posting(&self, entry: Entry) -> io::Result<RoaringBitmap> {
        let mut bytes = &self.mmap[entry.offset as usize..(entry.offset + entry.length) as usize];
        let bitmap = RoaringBitmap::deserialize_from(&mut bytes)?;
        if !bytes.is_empty() || bitmap.len() != entry.df {
            return Err(invalid("posting length or cardinality mismatch"));
        }
        Ok(bitmap)
    }

    fn candidates(&self, needle: &[u8], k: usize, max_trigrams: usize) -> io::Result<Vec<u32>> {
        if k == 0 || max_trigrams == 0 || needle.len() < 3 {
            return Ok(Vec::new());
        }
        let unique: HashSet<_> = trigrams(needle).collect();
        let mut selected = Vec::with_capacity(unique.len());
        for tg in unique {
            let Some(entry) = self.find(tg) else {
                return Ok(Vec::new());
            };
            selected.push(entry);
        }
        if selected.len() > max_trigrams {
            selected.select_nth_unstable_by_key(max_trigrams - 1, |e| (e.df, e.trigram));
            selected.truncate(max_trigrams);
        }
        selected.sort_unstable_by_key(|e| (e.df, e.trigram));
        let mut result = self.read_posting(selected[0])?;
        for &entry in &selected[1..] {
            if result.is_empty() {
                break;
            }
            result &= self.read_posting(entry)?;
        }
        Ok(result.iter().take(k).collect())
    }
}

#[pymodule]
fn roaring_trigram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TrigramIndexBuilder>()?;
    m.add_class::<TrigramIndex>()?;
    Ok(())
}
