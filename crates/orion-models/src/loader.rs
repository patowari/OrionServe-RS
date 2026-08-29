//! Safetensors checkpoint loading.
//!
//! Weights are memory-mapped rather than read into a buffer. For a
//! multi-gigabyte checkpoint that avoids doubling peak memory during load and
//! lets the OS page in only what is touched. The mapping is held for the
//! lifetime of the loader.
//!
//! Tensors are converted to `f32` on access, because the CPU reference path
//! computes in `f32`. A GPU backend would keep the native dtype instead; that
//! is why conversion lives here at the boundary rather than in the layers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use memmap2::Mmap;
use orion_core::ModelError;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::tensor::Matrix;

/// A memory-mapped safetensors file.
///
/// A parsed `SafeTensors` borrows from the mapping, so storing both in one
/// struct would be self-referential. Rather than reach for a self-referential
/// struct crate, the header is re-parsed on each access through
/// [`with_tensors`](ShardFile::with_tensors). Header parsing is cheap relative
/// to the tensor reads it guards, and it keeps the borrow local.
#[derive(Debug)]
struct ShardFile {
    path: PathBuf,
    mmap: Mmap,
}

impl ShardFile {
    fn open(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path).map_err(|source| ModelError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: `Mmap::map` is unsafe because the mapping becomes undefined
        // behaviour if another process truncates or writes the file while it is
        // mapped. The invariant we rely on is that the model directory is
        // operator-controlled and static for the lifetime of the process --
        // the same assumption every inference server makes about its weights.
        // Nothing in OrionServe writes to the model directory. If a checkpoint
        // must be swapped at runtime, the server should be restarted rather
        // than this assumption weakened.
        #[allow(unsafe_code)]
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| ModelError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
        })
    }

    fn with_tensors<T>(
        &self,
        f: impl FnOnce(&SafeTensors<'_>) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let st = SafeTensors::deserialize(&self.mmap).map_err(|e| ModelError::Malformed {
            file: self.path.display().to_string(),
            reason: e.to_string(),
        })?;
        f(&st)
    }
}

/// Converts a safetensors view into `f32` values.
fn view_to_f32(name: &str, view: &TensorView<'_>) -> Result<Vec<f32>, ModelError> {
    let bytes = view.data();
    let out = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            return Err(ModelError::UnsupportedDtype(format!(
                "{other:?} (tensor `{name}`)"
            )))
        }
    };
    Ok(out)
}

/// Loads tensors by name from one or more safetensors shards.
#[derive(Debug)]
pub struct CheckpointLoader {
    shards: Vec<ShardFile>,
    /// Tensor name to the shard holding it.
    index: HashMap<String, usize>,
}

impl CheckpointLoader {
    /// Opens every `*.safetensors` file in a model directory and indexes their
    /// contents.
    ///
    /// Large checkpoints are sharded across several files; the index makes
    /// lookup independent of how many.
    pub fn open(dir: &Path) -> Result<Self, ModelError> {
        if !dir.exists() {
            return Err(ModelError::PathNotFound(dir.to_path_buf()));
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|source| ModelError::Io {
                path: dir.to_path_buf(),
                source,
            })?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        // Deterministic order so a duplicated tensor name always resolves the
        // same way across runs.
        paths.sort();

        if paths.is_empty() {
            return Err(ModelError::MissingFile(dir.join("*.safetensors")));
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (i, path) in paths.iter().enumerate() {
            let shard = ShardFile::open(path)?;
            let names = shard.with_tensors(|st| {
                Ok(st
                    .names()
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>())
            })?;
            for name in names {
                index.entry(name).or_insert(i);
            }
            shards.push(shard);
        }

        tracing::info!(
            shards = shards.len(),
            tensors = index.len(),
            "opened checkpoint"
        );
        Ok(Self { shards, index })
    }

    /// Number of distinct tensors available.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Whether a tensor is present.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.keys().cloned().collect();
        v.sort();
        v
    }

    /// Loads a tensor as a flat `f32` vector with its shape.
    pub fn tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), ModelError> {
        let shard_idx = *self
            .index
            .get(name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_string()))?;

        self.shards[shard_idx].with_tensors(|st| {
            let view = st
                .tensor(name)
                .map_err(|_| ModelError::TensorNotFound(name.to_string()))?;
            let shape = view.shape().to_vec();
            let data = view_to_f32(name, &view)?;
            Ok((data, shape))
        })
    }

    /// Loads a 2-D tensor as a [`Matrix`], checking its shape.
    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix, ModelError> {
        let (data, shape) = self.tensor(name)?;
        if shape != vec![rows, cols] {
            return Err(ModelError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![rows, cols],
                actual: shape,
            });
        }
        Matrix::new(data, rows, cols).map_err(|e| ModelError::Malformed {
            file: name.to_string(),
            reason: e.to_string(),
        })
    }

    /// Loads a 1-D tensor, checking its length.
    pub fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, ModelError> {
        let (data, shape) = self.tensor(name)?;
        if shape != vec![len] {
            return Err(ModelError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![len],
                actual: shape,
            });
        }
        Ok(data)
    }

    /// Loads a 1-D tensor if present, returning `None` when absent.
    ///
    /// Used for optional biases, which distinguish Qwen2 from Llama.
    pub fn optional_vector(&self, name: &str, len: usize) -> Result<Option<Vec<f32>>, ModelError> {
        if !self.contains(name) {
            return Ok(None);
        }
        self.vector(name, len).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::serialize_to_file;
    use std::collections::BTreeMap;

    /// Writes a small checkpoint to a temporary directory.
    fn write_checkpoint(dir: &Path, tensors: Vec<(&str, Vec<usize>, Vec<f32>)>) {
        let mut map: BTreeMap<String, safetensors::tensor::TensorView> = BTreeMap::new();
        // Keep the byte buffers alive for the duration of serialization.
        let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
            .into_iter()
            .map(|(name, shape, data)| {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (name.to_string(), shape, bytes)
            })
            .collect();
        for (name, shape, bytes) in &owned {
            map.insert(
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            );
        }
        std::fs::create_dir_all(dir).unwrap();
        serialize_to_file(&map, None, &dir.join("model.safetensors")).unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("orion-loader-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn loads_tensors_by_name_with_their_shapes() {
        let dir = temp_dir("basic");
        write_checkpoint(
            &dir,
            vec![
                ("weight", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ("norm", vec![3], vec![0.5, 0.5, 0.5]),
            ],
        );

        let loader = CheckpointLoader::open(&dir).unwrap();
        assert_eq!(loader.len(), 2);
        assert!(loader.contains("weight"));
        assert!(!loader.contains("absent"));
        assert_eq!(loader.names(), vec!["norm", "weight"]);

        let (data, shape) = loader.tensor("weight").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn matrix_loading_checks_the_shape() {
        let dir = temp_dir("shape");
        write_checkpoint(&dir, vec![("w", vec![2, 3], vec![1.0; 6])]);
        let loader = CheckpointLoader::open(&dir).unwrap();

        let m = loader.matrix("w", 2, 3).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);

        let err = loader.matrix("w", 3, 2).unwrap_err();
        assert!(
            matches!(err, ModelError::ShapeMismatch { ref expected, ref actual, .. }
                     if *expected == vec![3, 2] && *actual == vec![2, 3]),
            "got {err:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_tensor_names_itself() {
        let dir = temp_dir("missing");
        write_checkpoint(&dir, vec![("present", vec![2], vec![1.0, 2.0])]);
        let loader = CheckpointLoader::open(&dir).unwrap();

        let err = loader.tensor("absent").unwrap_err();
        assert!(matches!(err, ModelError::TensorNotFound(n) if n == "absent"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optional_tensors_return_none_when_absent() {
        let dir = temp_dir("optional");
        write_checkpoint(&dir, vec![("bias", vec![2], vec![1.0, 2.0])]);
        let loader = CheckpointLoader::open(&dir).unwrap();

        assert_eq!(
            loader.optional_vector("bias", 2).unwrap(),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(loader.optional_vector("no_bias", 2).unwrap(), None);
        // A present-but-wrong-shape optional tensor is still an error.
        assert!(loader.optional_vector("bias", 5).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_directory_with_no_checkpoint_is_refused() {
        let dir = temp_dir("empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            CheckpointLoader::open(&dir).unwrap_err(),
            ModelError::MissingFile(_)
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_nonexistent_directory_is_refused() {
        assert!(matches!(
            CheckpointLoader::open(Path::new("/definitely/not/here")).unwrap_err(),
            ModelError::PathNotFound(_)
        ));
    }

    #[test]
    fn tensors_are_indexed_across_multiple_shards() {
        let dir = temp_dir("shards");
        std::fs::create_dir_all(&dir).unwrap();

        // Two shards, each with its own tensor.
        for (file, name) in [("a.safetensors", "first"), ("b.safetensors", "second")] {
            let bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut map = BTreeMap::new();
            map.insert(
                name.to_string(),
                TensorView::new(Dtype::F32, vec![2], &bytes).unwrap(),
            );
            serialize_to_file(&map, None, &dir.join(file)).unwrap();
        }

        let loader = CheckpointLoader::open(&dir).unwrap();
        assert_eq!(loader.len(), 2);
        assert!(loader.contains("first") && loader.contains("second"));
        assert_eq!(loader.vector("second", 2).unwrap(), vec![1.0, 2.0]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
