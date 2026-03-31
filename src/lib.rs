//! contextwell_core — Rust extension module for contextwell (PyO3).
//!
//! This module exposes performance-critical operations to the Python layer:
//! - `MemoryRecord`: a lightweight data container for memory entries
//! - `search_candidates`: Reciprocal Rank Fusion (RRF) over dense + sparse result sets
//!
//! The Python MCP server delegates fusion and scoring here once the design stabilises.

use pyo3::prelude::*;
use std::collections::HashMap;

/// A lightweight memory record passed between Rust and Python.
#[pyclass]
#[derive(Clone)]
pub struct MemoryRecord {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub score: f32,
}

#[pymethods]
impl MemoryRecord {
    #[new]
    pub fn new(id: String, content: String, score: f32) -> Self {
        Self { id, content, score }
    }

    fn __repr__(&self) -> String {
        format!("MemoryRecord(id={:?}, score={:.4})", &self.id[..8.min(self.id.len())], self.score)
    }
}

/// Reciprocal Rank Fusion over two ranked result lists.
///
/// `dense_ids` and `sparse_ids` are ordered lists of memory IDs (best first).
/// Returns a list of `(id, rrf_score)` tuples sorted by descending score.
///
/// RRF formula: score(d) = Σ 1 / (k + rank(d))  where k=60 is the standard constant.
#[pyfunction]
#[pyo3(signature = (dense_ids, sparse_ids, k=60.0))]
pub fn search_candidates(
    dense_ids: Vec<String>,
    sparse_ids: Vec<String>,
    k: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (rank, id) in dense_ids.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + (rank as f32) + 1.0);
    }
    for (rank, id) in sparse_ids.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + (rank as f32) + 1.0);
    }

    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// The contextwell._core extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MemoryRecord>()?;
    m.add_function(wrap_pyfunction!(search_candidates, m)?)?;
    Ok(())
}
