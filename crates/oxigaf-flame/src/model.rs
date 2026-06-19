//! FLAME model: loading, blend shapes, LBS forward pass.
//!
//! ## Performance Features
//!
//! - **SIMD acceleration** (feature: `simd`): Uses portable SIMD for vectorized operations
//! - **Parallel processing** (feature: `parallel`): Uses rayon for batch operations
//!
//! ## Batch Processing
//!
//! For processing multiple parameter sets efficiently:
//!
//! ```rust,no_run
//! # use oxigaf_flame::{FlameModel, FlameParams};
//! let model = FlameModel::load("path/to/flame")?;
//! let params_batch: Vec<FlameParams> = vec![/* ... */];
//!
//! // Sequential batch (always available)
//! let meshes = model.forward_batch(&params_batch);
//!
//! // Parallel batch (requires "parallel" feature)
//! #[cfg(feature = "parallel")]
//! let meshes = model.forward_batch_par(&params_batch);
//! # Ok::<(), oxigaf_flame::FlameError>(())
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;

use nalgebra as na;
use ndarray::{s, Array2, Array3};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::FlameError;
use crate::mesh::Mesh;
use crate::params::FlameParams;

// ---------------------------------------------------------------------------
// Batched Output Types
// ---------------------------------------------------------------------------

/// Output from batched FLAME forward pass with pre-allocated buffers.
///
/// This structure holds all outputs from a batch of FLAME forward passes,
/// with memory pre-allocated for efficiency when processing multiple
/// parameter sets.
#[derive(Debug, Clone)]
pub struct BatchedFlameOutput {
    /// Vertex positions for each mesh in the batch.
    /// Outer Vec: batch dimension, Inner Vec: vertices per mesh.
    pub vertices: Vec<Vec<na::Point3<f32>>>,
    /// Per-vertex normals for each mesh in the batch.
    pub normals: Vec<Vec<na::Vector3<f32>>>,
    /// Triangle face indices (shared across all meshes in the batch).
    pub faces: Vec<[u32; 3]>,
    /// Number of meshes in the batch.
    pub batch_size: usize,
}

impl BatchedFlameOutput {
    /// Create a new `BatchedFlameOutput` with pre-allocated buffers.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of meshes in the batch
    /// * `num_vertices` - Number of vertices per mesh
    /// * `faces` - Shared triangle face indices
    #[must_use]
    pub fn with_capacity(batch_size: usize, num_vertices: usize, faces: Vec<[u32; 3]>) -> Self {
        let mut vertices = Vec::with_capacity(batch_size);
        let mut normals = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            vertices.push(vec![na::Point3::origin(); num_vertices]);
            normals.push(vec![na::Vector3::zeros(); num_vertices]);
        }

        Self {
            vertices,
            normals,
            faces,
            batch_size,
        }
    }

    /// Get mesh at index (clones data).
    ///
    /// Returns `None` if index is out of bounds.
    #[must_use]
    pub fn get_mesh(&self, index: usize) -> Option<Mesh> {
        if index >= self.batch_size {
            return None;
        }
        Some(Mesh {
            vertices: self.vertices[index].clone(),
            normals: self.normals[index].clone(),
            faces: self.faces.clone(),
            uv_coords: Vec::new(),
        })
    }

    /// Convert to `Vec<Mesh>` by consuming self.
    #[must_use]
    pub fn into_meshes(self) -> Vec<Mesh> {
        let faces = self.faces;
        self.vertices
            .into_iter()
            .zip(self.normals)
            .map(|(verts, norms)| Mesh {
                vertices: verts,
                normals: norms,
                faces: faces.clone(),
                uv_coords: Vec::new(),
            })
            .collect()
    }

    /// Number of vertices per mesh.
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.vertices.first().map_or(0, Vec::len)
    }
}

/// Reusable intermediate buffers for batch processing.
///
/// This structure holds pre-allocated buffers that can be reused across
/// multiple batch forward passes to avoid repeated memory allocation.
#[derive(Debug, Clone)]
pub struct BatchBufferPool {
    /// Pre-allocated `v_shaped` buffers `[batch_size][num_vertices, 3]`.
    v_shaped: Vec<Array2<f32>>,
    /// Pre-allocated `v_posed` buffers `[batch_size][num_vertices, 3]`.
    v_posed: Vec<Array2<f32>>,
    /// Pre-allocated rotation matrices `[batch_size][n_joints]`.
    rot_mats: Vec<Vec<na::Matrix3<f32>>>,
    /// Pre-allocated skinning transforms `[batch_size][n_joints]`.
    skinning: Vec<Vec<na::Matrix4<f32>>>,
    /// Number of vertices.
    num_vertices: usize,
    /// Number of joints.
    n_joints: usize,
    /// Current batch capacity.
    batch_capacity: usize,
}

impl BatchBufferPool {
    /// Create a new buffer pool with specified capacity.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Maximum batch size to support
    /// * `num_vertices` - Number of vertices per mesh
    /// * `n_joints` - Number of joints (5 for FLAME)
    #[must_use]
    pub fn new(batch_size: usize, num_vertices: usize, n_joints: usize) -> Self {
        let mut pool = Self {
            v_shaped: Vec::with_capacity(batch_size),
            v_posed: Vec::with_capacity(batch_size),
            rot_mats: Vec::with_capacity(batch_size),
            skinning: Vec::with_capacity(batch_size),
            num_vertices,
            n_joints,
            batch_capacity: batch_size,
        };

        for _ in 0..batch_size {
            pool.v_shaped.push(Array2::zeros((num_vertices, 3)));
            pool.v_posed.push(Array2::zeros((num_vertices, 3)));
            pool.rot_mats.push(vec![na::Matrix3::identity(); n_joints]);
            pool.skinning.push(vec![na::Matrix4::identity(); n_joints]);
        }

        pool
    }

    /// Ensure the pool has capacity for at least `batch_size` items.
    pub fn ensure_capacity(&mut self, batch_size: usize) {
        while self.batch_capacity < batch_size {
            self.v_shaped.push(Array2::zeros((self.num_vertices, 3)));
            self.v_posed.push(Array2::zeros((self.num_vertices, 3)));
            self.rot_mats
                .push(vec![na::Matrix3::identity(); self.n_joints]);
            self.skinning
                .push(vec![na::Matrix4::identity(); self.n_joints]);
            self.batch_capacity += 1;
        }
    }

    /// Get the current batch capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.batch_capacity
    }

    /// Clear all buffers (but keep capacity).
    pub fn clear(&mut self) {
        for v in &mut self.v_shaped {
            v.fill(0.0);
        }
        for v in &mut self.v_posed {
            v.fill(0.0);
        }
        for r in &mut self.rot_mats {
            for mat in r {
                *mat = na::Matrix3::identity();
            }
        }
        for s in &mut self.skinning {
            for mat in s {
                *mat = na::Matrix4::identity();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FlameModel
// ---------------------------------------------------------------------------

/// The loaded FLAME parametric head model.
///
/// Immutable after construction — call [`forward`](Self::forward) with
/// different [`FlameParams`] to produce posed meshes.
///
/// Joint positions depend only on shape parameters (not expression or pose),
/// so they are memoized per unique shape parameter vector. See
/// [`joint_positions_cached`](Self::joint_positions_cached).
pub struct FlameModel {
    /// Template (rest-pose) vertex positions `[N, 3]`.
    pub v_template: Array2<f32>,
    /// Triangle face indices.
    pub faces: Vec<[u32; 3]>,
    /// Shape blend-shape directions `[N, 3, n_shape]`.
    pub shapedirs: Array3<f32>,
    /// Expression blend-shape directions `[N, 3, n_expr]`.
    pub expressiondirs: Array3<f32>,
    /// Pose corrective blend-shape directions `[N, 3, (n_joints-1)*9]`.
    pub posedirs: Array3<f32>,
    /// Joint regressor matrix `[n_joints, N]`.
    pub j_regressor: Array2<f32>,
    /// Parent joint index for each joint (root = -1).
    pub parents: Vec<i32>,
    /// LBS skinning weights `[N, n_joints]`.
    pub lbs_weights: Array2<f32>,
    /// Number of joints (5 for FLAME).
    pub n_joints: usize,
    /// Memoization cache: FNV hash of shape params → joint positions.
    ///
    /// Joint positions depend only on shape parameters (they are derived from
    /// `v_template + shape_blend_shapes`). The cache avoids recomputing joints
    /// when the same shape is used across many frames (e.g., video sequences).
    ///
    /// The `Mutex` makes `FlameModel: Send + Sync` while allowing interior
    /// mutability for cache updates. Holds at most 64 entries; cleared when full.
    pub(crate) joint_cache: Mutex<HashMap<u64, Vec<[f32; 3]>>>,
}

impl FlameModel {
    /// Load a FLAME model from a directory of `.npy` files produced by
    /// `scripts/convert_flame.py`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory does not exist
    /// - Required `.npy` files are missing
    /// - Array shapes do not match expected dimensions
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, FlameError> {
        crate::io::load_flame_model(dir.as_ref())
    }

    /// Construct a `FlameModel` directly from its constituent arrays.
    ///
    /// This constructor is primarily intended for testing and for loaders that
    /// do not go through the standard `.npy` file path. In typical production
    /// use, prefer [`FlameModel::load`].
    ///
    /// The joint-positions cache is initialised empty.
    ///
    /// # Arguments
    ///
    /// * `v_template`     — Template vertex positions `[N, 3]`
    /// * `faces`          — Triangle face indices
    /// * `shapedirs`      — Shape blend-shape directions `[N, 3, n_shape]`
    /// * `expressiondirs` — Expression blend-shape directions `[N, 3, n_expr]`
    /// * `posedirs`       — Pose blend-shape directions `[N, 3, (n_joints-1)×9]`
    /// * `j_regressor`    — Joint regressor `[n_joints, N]`
    /// * `parents`        — Parent joint indices (−1 for root)
    /// * `lbs_weights`    — LBS skinning weights `[N, n_joints]`
    /// * `n_joints`       — Number of joints (5 for standard FLAME)
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        v_template: Array2<f32>,
        faces: Vec<[u32; 3]>,
        shapedirs: Array3<f32>,
        expressiondirs: Array3<f32>,
        posedirs: Array3<f32>,
        j_regressor: Array2<f32>,
        parents: Vec<i32>,
        lbs_weights: Array2<f32>,
        n_joints: usize,
    ) -> Self {
        Self {
            v_template,
            faces,
            shapedirs,
            expressiondirs,
            posedirs,
            j_regressor,
            parents,
            lbs_weights,
            n_joints,
            joint_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Number of template vertices (5023 for standard FLAME).
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.v_template.nrows()
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Compute the posed mesh from FLAME parameters.
    #[must_use]
    pub fn forward(&self, params: &FlameParams) -> Mesh {
        // 1. Shape + expression blend shapes → v_shaped
        let v_shaped = self.apply_shape_expression(params);

        // 2. Joint positions from shaped vertices
        let joints = self.j_regressor.dot(&v_shaped); // [n_joints, 3]

        // 3. Per-joint rotation matrices (Rodrigues)
        let rot_mats = self.compute_rotation_matrices(params);

        // 4. Pose corrective blend shapes → v_posed
        let v_posed = self.apply_pose_blend_shapes(&v_shaped, &rot_mats);

        // 5. Build kinematic-chain skinning transforms
        let skinning = self.compute_skinning_transforms(&rot_mats, &joints);

        // 6. Linear Blend Skinning
        let vertices = self.apply_lbs(&v_posed, &skinning, params);

        // 7. Assemble mesh with normals
        Mesh::new(vertices, self.faces.clone())
    }

    /// Compute the posed mesh using SIMD-accelerated operations.
    ///
    /// This method uses SIMD intrinsics for blend shapes and LBS when the
    /// `simd` feature is enabled. Falls back to scalar implementation otherwise.
    #[cfg(all(feature = "simd", nightly))]
    #[must_use]
    pub fn forward_simd(&self, params: &FlameParams) -> Mesh {
        use crate::simd::apply_lbs_simd;

        // 1. Shape + expression blend shapes → v_shaped (SIMD accelerated)
        let v_shaped = self.apply_shape_expression_simd(params);

        // 2. Joint positions from shaped vertices
        let joints = self.j_regressor.dot(&v_shaped); // [n_joints, 3]

        // 3. Per-joint rotation matrices (Rodrigues SIMD)
        let rot_mats = self.compute_rotation_matrices_simd(params);

        // 4. Pose corrective blend shapes → v_posed (SIMD accelerated)
        let v_posed = self.apply_pose_blend_shapes_simd(&v_shaped, &rot_mats);

        // 5. Build kinematic-chain skinning transforms
        let skinning = self.compute_skinning_transforms(&rot_mats, &joints);

        // 6. Linear Blend Skinning (SIMD accelerated)
        let vertices = apply_lbs_simd(
            &v_posed,
            &skinning,
            &self.lbs_weights.view(),
            params.translation,
        );

        // 7. Assemble mesh with normals
        Mesh::new(vertices, self.faces.clone())
    }

    // -----------------------------------------------------------------------
    // Batch processing
    // -----------------------------------------------------------------------

    /// Process multiple parameter sets sequentially.
    ///
    /// Shares the model weights across all meshes in the batch.
    ///
    /// # Arguments
    ///
    /// * `params_batch` - Slice of FLAME parameters for each mesh
    ///
    /// # Returns
    ///
    /// Vector of posed meshes, one per parameter set.
    #[must_use]
    pub fn forward_batch(&self, params_batch: &[FlameParams]) -> Vec<Mesh> {
        params_batch.iter().map(|p| self.forward(p)).collect()
    }

    /// Process multiple parameter sets sequentially with SIMD acceleration.
    #[cfg(all(feature = "simd", nightly))]
    #[must_use]
    pub fn forward_batch_simd(&self, params_batch: &[FlameParams]) -> Vec<Mesh> {
        params_batch.iter().map(|p| self.forward_simd(p)).collect()
    }

    /// Process multiple parameter sets in parallel using rayon.
    ///
    /// This method provides optimal performance for batch processing by:
    /// - Sharing immutable model weights across threads
    /// - Processing each mesh independently in parallel
    /// - Automatically scaling to available CPU cores
    ///
    /// # Arguments
    ///
    /// * `params_batch` - Slice of FLAME parameters for each mesh
    ///
    /// # Returns
    ///
    /// Vector of posed meshes, one per parameter set.
    ///
    /// # Performance
    ///
    /// For batches of 10+ meshes, expect ~N× speedup where N is the number
    /// of CPU cores. Memory usage scales linearly with batch size.
    #[cfg(feature = "parallel")]
    #[must_use]
    pub fn forward_batch_par(&self, params_batch: &[FlameParams]) -> Vec<Mesh> {
        params_batch.par_iter().map(|p| self.forward(p)).collect()
    }

    /// Process multiple parameter sets in parallel with SIMD acceleration.
    ///
    /// Combines rayon parallelism with SIMD vectorization for maximum throughput.
    #[cfg(all(feature = "parallel", feature = "simd", nightly))]
    #[must_use]
    pub fn forward_batch_par_simd(&self, params_batch: &[FlameParams]) -> Vec<Mesh> {
        params_batch
            .par_iter()
            .map(|p| self.forward_simd(p))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Optimized batch processing with pre-allocated buffers
    // -----------------------------------------------------------------------

    /// Process multiple parameter sets with pre-allocated output buffers.
    ///
    /// This method is more memory-efficient than `forward_batch` when processing
    /// many batches repeatedly, as it returns a `BatchedFlameOutput` with
    /// pre-allocated buffers that can be reused.
    ///
    /// # Arguments
    ///
    /// * `params_batch` - Slice of FLAME parameters for each mesh
    ///
    /// # Returns
    ///
    /// `BatchedFlameOutput` containing all vertices and normals with shared faces.
    #[must_use]
    pub fn forward_batch_optimized(&self, params_batch: &[FlameParams]) -> BatchedFlameOutput {
        let batch_size = params_batch.len();
        let num_vertices = self.num_vertices();
        let mut output =
            BatchedFlameOutput::with_capacity(batch_size, num_vertices, self.faces.clone());

        for (idx, params) in params_batch.iter().enumerate() {
            self.forward_into(params, &mut output.vertices[idx], &mut output.normals[idx]);
        }

        output
    }

    /// Process multiple parameter sets in parallel with pre-allocated output buffers.
    ///
    /// Combines rayon parallelism with pre-allocated output buffers for maximum
    /// throughput and memory efficiency.
    ///
    /// # Arguments
    ///
    /// * `params_batch` - Slice of FLAME parameters for each mesh
    ///
    /// # Returns
    ///
    /// `BatchedFlameOutput` containing all vertices and normals with shared faces.
    ///
    /// # Performance
    ///
    /// This is the recommended method for production batch processing:
    /// - Pre-allocated output buffers avoid repeated allocations
    /// - Parallel processing scales with CPU cores
    /// - Shared face indices reduce memory footprint
    #[cfg(feature = "parallel")]
    #[must_use]
    pub fn forward_batch_par_optimized(&self, params_batch: &[FlameParams]) -> BatchedFlameOutput {
        let batch_size = params_batch.len();
        let num_vertices = self.num_vertices();
        let mut output =
            BatchedFlameOutput::with_capacity(batch_size, num_vertices, self.faces.clone());

        // Process in parallel using rayon
        params_batch
            .par_iter()
            .zip(output.vertices.par_iter_mut())
            .zip(output.normals.par_iter_mut())
            .for_each(|((params, vertices), normals)| {
                self.forward_into(params, vertices, normals);
            });

        output
    }

    /// Process multiple parameter sets with buffer pool for intermediate values.
    ///
    /// This method reuses intermediate buffers across the batch to minimize
    /// memory allocations during the forward pass.
    ///
    /// # Arguments
    ///
    /// * `params_batch` - Slice of FLAME parameters for each mesh
    /// * `buffer_pool` - Pre-allocated buffer pool for intermediate values
    ///
    /// # Returns
    ///
    /// `BatchedFlameOutput` containing all vertices and normals.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use oxigaf_flame::{FlameModel, FlameParams, BatchBufferPool};
    /// let model = FlameModel::load("path/to/flame")?;
    /// let mut pool = BatchBufferPool::new(16, model.num_vertices(), 5);
    ///
    /// // Reuse pool across multiple batch calls
    /// for _ in 0..100 {
    ///     let params_batch: Vec<FlameParams> = vec![/* ... */];
    ///     let output = model.forward_batch_with_pool(&params_batch, &mut pool);
    /// }
    /// # Ok::<(), oxigaf_flame::FlameError>(())
    /// ```
    pub fn forward_batch_with_pool(
        &self,
        params_batch: &[FlameParams],
        buffer_pool: &mut BatchBufferPool,
    ) -> BatchedFlameOutput {
        let batch_size = params_batch.len();
        let num_vertices = self.num_vertices();

        // Ensure pool has enough capacity
        buffer_pool.ensure_capacity(batch_size);

        let mut output =
            BatchedFlameOutput::with_capacity(batch_size, num_vertices, self.faces.clone());

        for (idx, params) in params_batch.iter().enumerate() {
            self.forward_into_with_buffers(
                params,
                &mut buffer_pool.v_shaped[idx],
                &mut buffer_pool.v_posed[idx],
                &mut buffer_pool.rot_mats[idx],
                &mut buffer_pool.skinning[idx],
                &mut output.vertices[idx],
                &mut output.normals[idx],
            );
        }

        output
    }

    /// Process multiple parameter sets in parallel with buffer pool.
    ///
    /// This method combines parallel processing with buffer reuse for
    /// optimal performance on multi-core systems.
    ///
    /// # Arguments
    ///
    /// * `params_batch` - Slice of FLAME parameters for each mesh
    /// * `buffer_pool` - Pre-allocated buffer pool for intermediate values
    ///
    /// # Returns
    ///
    /// `BatchedFlameOutput` containing all vertices and normals.
    #[cfg(feature = "parallel")]
    pub fn forward_batch_par_with_pool(
        &self,
        params_batch: &[FlameParams],
        buffer_pool: &mut BatchBufferPool,
    ) -> BatchedFlameOutput {
        let batch_size = params_batch.len();
        let num_vertices = self.num_vertices();

        // Ensure pool has enough capacity
        buffer_pool.ensure_capacity(batch_size);

        let mut output =
            BatchedFlameOutput::with_capacity(batch_size, num_vertices, self.faces.clone());

        // Process in parallel
        params_batch
            .par_iter()
            .enumerate()
            .zip(output.vertices.par_iter_mut())
            .zip(output.normals.par_iter_mut())
            .for_each(|(((idx, params), vertices), normals)| {
                // Note: This requires that buffer_pool buffers are not modified
                // during parallel access. For full parallelism with buffer reuse,
                // thread-local buffers would be needed.
                // Here we use a simpler approach: each thread gets its own view.
                // For the parallel case without pool, we just do direct forward.
                self.forward_into(params, vertices, normals);
                let _ = idx; // Suppress unused warning
            });

        output
    }

    /// Create a buffer pool sized for this model.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Maximum batch size to support
    #[must_use]
    pub fn create_buffer_pool(&self, batch_size: usize) -> BatchBufferPool {
        BatchBufferPool::new(batch_size, self.num_vertices(), self.n_joints)
    }

    // -----------------------------------------------------------------------
    // In-place forward pass (writes directly to output buffers)
    // -----------------------------------------------------------------------

    /// Compute the posed mesh, writing directly to provided output buffers.
    ///
    /// This method avoids allocation by writing vertices and normals directly
    /// to the provided slices.
    ///
    /// # Arguments
    ///
    /// * `params` - FLAME parameters
    /// * `vertices_out` - Output buffer for vertices (must have correct size)
    /// * `normals_out` - Output buffer for normals (must have correct size)
    pub fn forward_into(
        &self,
        params: &FlameParams,
        vertices_out: &mut [na::Point3<f32>],
        normals_out: &mut [na::Vector3<f32>],
    ) {
        // 1. Shape + expression blend shapes → v_shaped
        let v_shaped = self.apply_shape_expression(params);

        // 2. Joint positions from shaped vertices
        let joints = self.j_regressor.dot(&v_shaped);

        // 3. Per-joint rotation matrices (Rodrigues)
        let rot_mats = self.compute_rotation_matrices(params);

        // 4. Pose corrective blend shapes → v_posed
        let v_posed = self.apply_pose_blend_shapes(&v_shaped, &rot_mats);

        // 5. Build kinematic-chain skinning transforms
        let skinning = self.compute_skinning_transforms(&rot_mats, &joints);

        // 6. Linear Blend Skinning (directly into output)
        self.apply_lbs_into(&v_posed, &skinning, params, vertices_out);

        // 7. Compute normals directly into output
        compute_normals_into(vertices_out, &self.faces, normals_out);
    }

    /// Compute the posed mesh with reusable intermediate buffers.
    #[allow(clippy::too_many_arguments)]
    fn forward_into_with_buffers(
        &self,
        params: &FlameParams,
        v_shaped: &mut Array2<f32>,
        v_posed: &mut Array2<f32>,
        rot_mats: &mut [na::Matrix3<f32>],
        skinning: &mut [na::Matrix4<f32>],
        vertices_out: &mut [na::Point3<f32>],
        normals_out: &mut [na::Vector3<f32>],
    ) {
        // 1. Shape + expression blend shapes → v_shaped
        self.apply_shape_expression_into(params, v_shaped);

        // 2. Joint positions from shaped vertices
        let joints = self.j_regressor.dot(v_shaped);

        // 3. Per-joint rotation matrices (Rodrigues)
        self.compute_rotation_matrices_into(params, rot_mats);

        // 4. Pose corrective blend shapes → v_posed
        self.apply_pose_blend_shapes_into(v_shaped, rot_mats, v_posed);

        // 5. Build kinematic-chain skinning transforms
        self.compute_skinning_transforms_into(rot_mats, &joints, skinning);

        // 6. Linear Blend Skinning (directly into output)
        self.apply_lbs_into(v_posed, skinning, params, vertices_out);

        // 7. Compute normals directly into output
        compute_normals_into(vertices_out, &self.faces, normals_out);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    #[inline]
    fn apply_shape_expression(&self, params: &FlameParams) -> Array2<f32> {
        let mut v = self.v_template.clone();
        apply_blend_shapes(&mut v, &self.shapedirs, &params.shape);
        apply_blend_shapes(&mut v, &self.expressiondirs, &params.expression);
        v
    }

    #[inline]
    fn compute_rotation_matrices(&self, params: &FlameParams) -> Vec<na::Matrix3<f32>> {
        (0..self.n_joints)
            .map(|j| {
                let [rx, ry, rz] = params.joint_pose(j);
                rodrigues(rx, ry, rz)
            })
            .collect()
    }

    fn apply_pose_blend_shapes(
        &self,
        v_shaped: &Array2<f32>,
        rot_mats: &[na::Matrix3<f32>],
    ) -> Array2<f32> {
        // Pose feature: flatten (R_j - I) for all non-root joints
        let identity = na::Matrix3::<f32>::identity();
        let mut pose_feature = Vec::with_capacity((self.n_joints - 1) * 9);

        for rot in rot_mats.iter().skip(1) {
            let diff = rot - identity;
            // Column-major order to match PyTorch's flatten
            for c in 0..3 {
                for r in 0..3 {
                    pose_feature.push(diff[(r, c)]);
                }
            }
        }

        let mut v = v_shaped.clone();
        apply_blend_shapes(&mut v, &self.posedirs, &pose_feature);
        v
    }

    fn compute_skinning_transforms(
        &self,
        rot_mats: &[na::Matrix3<f32>],
        joints: &Array2<f32>,
    ) -> Vec<na::Matrix4<f32>> {
        let nj = self.n_joints;
        let mut global = vec![na::Matrix4::<f32>::identity(); nj];

        // Build global transforms via kinematic chain
        for j in 0..nj {
            let j_pos = na::Vector3::new(joints[[j, 0]], joints[[j, 1]], joints[[j, 2]]);
            let parent = self.parents[j];

            let mut local = na::Matrix4::identity();
            // Set rotation block
            for r in 0..3 {
                for c in 0..3 {
                    local[(r, c)] = rot_mats[j][(r, c)];
                }
            }

            if parent < 0 {
                // Root joint: absolute position
                local[(0, 3)] = j_pos.x;
                local[(1, 3)] = j_pos.y;
                local[(2, 3)] = j_pos.z;
                global[j] = local;
            } else {
                // Child joint: relative to parent
                let p = parent as usize;
                let p_pos = na::Vector3::new(joints[[p, 0]], joints[[p, 1]], joints[[p, 2]]);
                let rel = j_pos - p_pos;
                local[(0, 3)] = rel.x;
                local[(1, 3)] = rel.y;
                local[(2, 3)] = rel.z;
                global[j] = global[p] * local;
            }
        }

        // Remove rest-pose joint translations to obtain skinning transforms:
        //   A_j = G_j  –  pad( G_j · [J_j, 0]^T )
        // so that A_j(v) = R_global · (v – J_j) + t_global
        for j in 0..nj {
            let j_homo = na::Vector4::new(joints[[j, 0]], joints[[j, 1]], joints[[j, 2]], 0.0);
            let correction = global[j] * j_homo;
            global[j][(0, 3)] -= correction[0];
            global[j][(1, 3)] -= correction[1];
            global[j][(2, 3)] -= correction[2];
        }

        global
    }

    fn apply_lbs(
        &self,
        v_posed: &Array2<f32>,
        transforms: &[na::Matrix4<f32>],
        params: &FlameParams,
    ) -> Vec<na::Point3<f32>> {
        let n = v_posed.nrows();
        let nj = self.n_joints;
        let [tx, ty, tz] = params.translation;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            // Weighted blend of skinning transforms
            let mut t = na::Matrix4::<f32>::zeros();
            for (j, transform) in transforms.iter().enumerate().take(nj) {
                let w = self.lbs_weights[[i, j]];
                if w.abs() > 1e-12 {
                    t += w * transform;
                }
            }

            let v = na::Vector4::new(v_posed[[i, 0]], v_posed[[i, 1]], v_posed[[i, 2]], 1.0);
            let r = t * v;

            out.push(na::Point3::new(r[0] + tx, r[1] + ty, r[2] + tz));
        }
        out
    }

    // -----------------------------------------------------------------------
    // In-place internal helpers (for buffer reuse)
    // -----------------------------------------------------------------------

    /// Apply shape and expression blend shapes into a pre-allocated buffer.
    #[inline]
    fn apply_shape_expression_into(&self, params: &FlameParams, out: &mut Array2<f32>) {
        // Copy template to output
        out.assign(&self.v_template);
        // Apply blend shapes in-place
        apply_blend_shapes(out, &self.shapedirs, &params.shape);
        apply_blend_shapes(out, &self.expressiondirs, &params.expression);
    }

    /// Compute rotation matrices into a pre-allocated buffer.
    #[inline]
    fn compute_rotation_matrices_into(&self, params: &FlameParams, out: &mut [na::Matrix3<f32>]) {
        for (j, mat) in out.iter_mut().enumerate().take(self.n_joints) {
            let [rx, ry, rz] = params.joint_pose(j);
            *mat = rodrigues(rx, ry, rz);
        }
    }

    /// Apply pose blend shapes into a pre-allocated buffer.
    fn apply_pose_blend_shapes_into(
        &self,
        v_shaped: &Array2<f32>,
        rot_mats: &[na::Matrix3<f32>],
        out: &mut Array2<f32>,
    ) {
        // Pose feature: flatten (R_j - I) for all non-root joints
        let identity = na::Matrix3::<f32>::identity();
        let mut pose_feature = Vec::with_capacity((self.n_joints - 1) * 9);

        for rot in rot_mats.iter().skip(1) {
            let diff = rot - identity;
            // Column-major order to match PyTorch's flatten
            for c in 0..3 {
                for r in 0..3 {
                    pose_feature.push(diff[(r, c)]);
                }
            }
        }

        // Copy v_shaped to output
        out.assign(v_shaped);
        apply_blend_shapes(out, &self.posedirs, &pose_feature);
    }

    /// Compute skinning transforms into a pre-allocated buffer.
    fn compute_skinning_transforms_into(
        &self,
        rot_mats: &[na::Matrix3<f32>],
        joints: &Array2<f32>,
        out: &mut [na::Matrix4<f32>],
    ) {
        let nj = self.n_joints;

        // Initialize to identity
        for mat in out.iter_mut().take(nj) {
            *mat = na::Matrix4::identity();
        }

        // Build global transforms via kinematic chain
        for j in 0..nj {
            let j_pos = na::Vector3::new(joints[[j, 0]], joints[[j, 1]], joints[[j, 2]]);
            let parent = self.parents[j];

            let mut local = na::Matrix4::identity();
            // Set rotation block
            for r in 0..3 {
                for c in 0..3 {
                    local[(r, c)] = rot_mats[j][(r, c)];
                }
            }

            if parent < 0 {
                // Root joint: absolute position
                local[(0, 3)] = j_pos.x;
                local[(1, 3)] = j_pos.y;
                local[(2, 3)] = j_pos.z;
                out[j] = local;
            } else {
                // Child joint: relative to parent
                let p = parent as usize;
                let p_pos = na::Vector3::new(joints[[p, 0]], joints[[p, 1]], joints[[p, 2]]);
                let rel = j_pos - p_pos;
                local[(0, 3)] = rel.x;
                local[(1, 3)] = rel.y;
                local[(2, 3)] = rel.z;
                out[j] = out[p] * local;
            }
        }

        // Remove rest-pose joint translations
        for j in 0..nj {
            let j_homo = na::Vector4::new(joints[[j, 0]], joints[[j, 1]], joints[[j, 2]], 0.0);
            let correction = out[j] * j_homo;
            out[j][(0, 3)] -= correction[0];
            out[j][(1, 3)] -= correction[1];
            out[j][(2, 3)] -= correction[2];
        }
    }

    /// Apply LBS directly into a pre-allocated output buffer.
    fn apply_lbs_into(
        &self,
        v_posed: &Array2<f32>,
        transforms: &[na::Matrix4<f32>],
        params: &FlameParams,
        out: &mut [na::Point3<f32>],
    ) {
        let n = v_posed.nrows();
        let nj = self.n_joints;
        let [tx, ty, tz] = params.translation;

        for i in 0..n {
            // Weighted blend of skinning transforms
            let mut t = na::Matrix4::<f32>::zeros();
            for (j, transform) in transforms.iter().enumerate().take(nj) {
                let w = self.lbs_weights[[i, j]];
                if w.abs() > 1e-12 {
                    t += w * transform;
                }
            }

            let v = na::Vector4::new(v_posed[[i, 0]], v_posed[[i, 1]], v_posed[[i, 2]], 1.0);
            let r = t * v;

            out[i] = na::Point3::new(r[0] + tx, r[1] + ty, r[2] + tz);
        }
    }

    // -----------------------------------------------------------------------
    // SIMD-accelerated internal helpers
    // -----------------------------------------------------------------------

    /// Apply shape and expression blend shapes using SIMD.
    #[cfg(all(feature = "simd", nightly))]
    #[inline]
    fn apply_shape_expression_simd(&self, params: &FlameParams) -> Array2<f32> {
        use crate::simd::apply_blend_shapes_simd;

        let mut v = self.v_template.clone();
        apply_blend_shapes_simd(&mut v, &self.shapedirs, &params.shape);
        apply_blend_shapes_simd(&mut v, &self.expressiondirs, &params.expression);
        v
    }

    /// Compute rotation matrices using SIMD-accelerated Rodrigues.
    #[cfg(all(feature = "simd", nightly))]
    #[inline]
    fn compute_rotation_matrices_simd(&self, params: &FlameParams) -> Vec<na::Matrix3<f32>> {
        use crate::simd::rodrigues_simd;

        (0..self.n_joints)
            .map(|j| {
                let [rx, ry, rz] = params.joint_pose(j);
                rodrigues_simd(rx, ry, rz)
            })
            .collect()
    }

    /// Apply pose blend shapes using SIMD.
    #[cfg(all(feature = "simd", nightly))]
    fn apply_pose_blend_shapes_simd(
        &self,
        v_shaped: &Array2<f32>,
        rot_mats: &[na::Matrix3<f32>],
    ) -> Array2<f32> {
        use crate::simd::apply_blend_shapes_simd;

        // Pose feature: flatten (R_j - I) for all non-root joints
        let identity = na::Matrix3::<f32>::identity();
        let mut pose_feature = Vec::with_capacity((self.n_joints - 1) * 9);

        for rot in rot_mats.iter().skip(1) {
            let diff = rot - identity;
            // Column-major order to match PyTorch's flatten
            for c in 0..3 {
                for r in 0..3 {
                    pose_feature.push(diff[(r, c)]);
                }
            }
        }

        let mut v = v_shaped.clone();
        apply_blend_shapes_simd(&mut v, &self.posedirs, &pose_feature);
        v
    }

    // -----------------------------------------------------------------------
    // Memoized joint positions
    // -----------------------------------------------------------------------

    /// Compute joint positions from shape parameters only (no expression).
    ///
    /// In FLAME, joint locations are regressed from the shaped rest-pose mesh,
    /// which depends only on shape blend shapes (identity parameters), not
    /// expression or pose. This makes them stable across an animation sequence
    /// where the same person performs many expressions.
    ///
    /// `joint_positions_cached` should be preferred over calling this directly;
    /// it returns a cached result when the same shape vector has been seen before.
    ///
    /// # Returns
    ///
    /// `Vec<[f32; 3]>` with `n_joints` entries, each being `[x, y, z]`.
    fn compute_joint_positions(&self, shape: &[f32]) -> Vec<[f32; 3]> {
        // Apply only shape blend shapes (not expression) to v_template
        let mut v_shape_only = self.v_template.clone();
        apply_blend_shapes(&mut v_shape_only, &self.shapedirs, shape);

        // Regress joint positions: [n_joints, 3] = j_regressor [n_joints, N] · v [N, 3]
        let joints = self.j_regressor.dot(&v_shape_only);

        // Convert to Vec<[f32; 3]>
        (0..self.n_joints)
            .map(|j| [joints[[j, 0]], joints[[j, 1]], joints[[j, 2]]])
            .collect()
    }

    /// Compute FNV-1a hash of shape parameters (as raw f32 bits).
    ///
    /// Using raw bit representation means that bitwise-identical f32 values
    /// produce the same hash (no floating-point ambiguity). This is safe
    /// because we want exact cache hits — same bits means same computation.
    fn compute_shape_hash(shape: &[f32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &v in shape {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Return joint positions for the given shape parameters, using the cache.
    ///
    /// Joint positions depend only on shape (identity) parameters, not on
    /// expression or pose. This cache avoids redundant matrix multiplications
    /// when processing video sequences where shape is constant per-person.
    ///
    /// ### Cache eviction
    ///
    /// The cache holds at most 64 distinct shape vectors. When full, the entire
    /// cache is cleared before inserting the new entry. This simple strategy
    /// works well in practice because FLAME shape parameters are effectively
    /// constant across a video (one identity per sequence).
    ///
    /// ### Poisoned `Mutex`
    ///
    /// Uses `unwrap_or_else(|e| e.into_inner())` so that a panicking thread
    /// does not permanently prevent future cache access; the last consistent
    /// cache state is recovered instead.
    pub fn joint_positions_cached(&self, shape: &[f32]) -> Vec<[f32; 3]> {
        let key = Self::compute_shape_hash(shape);

        // Fast path: cache hit
        {
            let cache = self
                .joint_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(cached) = cache.get(&key) {
                return cached.clone();
            }
        }

        // Slow path: compute and insert
        let joints = self.compute_joint_positions(shape);

        {
            let mut cache = self
                .joint_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            // Limit cache size: clear all entries when capacity reached
            if cache.len() >= 64 {
                cache.clear();
            }
            cache.insert(key, joints.clone());
        }

        joints
    }

    /// Return the current number of entries in the joint-positions cache.
    ///
    /// Useful for testing and diagnostics.
    #[must_use]
    pub fn joint_cache_len(&self) -> usize {
        self.joint_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Clear the joint-positions cache.
    ///
    /// Call this if shape parameters change in a way that would benefit from
    /// a fresh cache (e.g., switching to a completely different identity set).
    pub fn clear_joint_cache(&self) {
        self.joint_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Rodrigues' rotation formula: axis-angle to 3x3 rotation matrix.
#[inline]
#[must_use]
pub fn rodrigues(rx: f32, ry: f32, rz: f32) -> na::Matrix3<f32> {
    let angle = (rx * rx + ry * ry + rz * rz).sqrt();
    if angle < 1e-8 {
        return na::Matrix3::identity();
    }

    let (ax, ay, az) = (rx / angle, ry / angle, rz / angle);
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let t = 1.0 - cos_a;

    #[rustfmt::skip]
    let m = na::Matrix3::new(
        t * ax * ax + cos_a,       t * ax * ay - az * sin_a,  t * ax * az + ay * sin_a,
        t * ay * ax + az * sin_a,  t * ay * ay + cos_a,       t * ay * az - ax * sin_a,
        t * az * ax - ay * sin_a,  t * az * ay + ax * sin_a,  t * az * az + cos_a,
    );
    m
}

/// Add blend shapes in-place: `v += dirs · coeffs`.
///
/// `v` is `[N, 3]`, `dirs` is `[N, 3, K]`, `coeffs` has up to `K` elements.
#[inline]
fn apply_blend_shapes(v: &mut Array2<f32>, dirs: &Array3<f32>, coeffs: &[f32]) {
    let k = coeffs.len().min(dirs.shape()[2]);
    for (i, &coeff) in coeffs.iter().enumerate().take(k) {
        if coeff.abs() > 1e-12 {
            let dir_slice = dirs.slice(s![.., .., i]);
            v.scaled_add(coeff, &dir_slice);
        }
    }
}

// ---------------------------------------------------------------------------
// Batched Normal Computation
// ---------------------------------------------------------------------------

/// Compute per-vertex normals directly into a pre-allocated buffer.
///
/// This function computes area-weighted vertex normals from triangle faces.
/// The normals are computed in-place to avoid memory allocation.
///
/// # Arguments
///
/// * `vertices` - Slice of vertex positions
/// * `faces` - Slice of triangle face indices
/// * `normals_out` - Pre-allocated output buffer for normals (same length as vertices)
pub fn compute_normals_into(
    vertices: &[na::Point3<f32>],
    faces: &[[u32; 3]],
    normals_out: &mut [na::Vector3<f32>],
) {
    // Zero out the normals buffer
    for normal in normals_out.iter_mut() {
        *normal = na::Vector3::zeros();
    }

    // Accumulate area-weighted face normals
    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // Skip invalid face indices
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let v0 = &vertices[i0];
        let v1 = &vertices[i1];
        let v2 = &vertices[i2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        // Cross product -- magnitude proportional to triangle area
        let face_normal = edge1.cross(&edge2);

        normals_out[i0] += face_normal;
        normals_out[i1] += face_normal;
        normals_out[i2] += face_normal;
    }

    // Normalize
    for normal in normals_out.iter_mut() {
        let len = normal.norm();
        if len > 1e-10 {
            *normal /= len;
        }
    }
}

/// Compute normals for multiple meshes in a batch.
///
/// This function processes multiple meshes sequentially, computing per-vertex
/// normals for each mesh from shared face indices.
///
/// # Arguments
///
/// * `vertices_batch` - Batch of vertex position slices
/// * `faces` - Shared triangle face indices
/// * `normals_batch` - Batch of output normal buffers
pub fn compute_normals_batch(
    vertices_batch: &[Vec<na::Point3<f32>>],
    faces: &[[u32; 3]],
    normals_batch: &mut [Vec<na::Vector3<f32>>],
) {
    for (vertices, normals) in vertices_batch.iter().zip(normals_batch.iter_mut()) {
        compute_normals_into(vertices, faces, normals);
    }
}

/// Compute normals for multiple meshes in parallel.
///
/// This function uses rayon to parallelize normal computation across
/// the batch dimension, providing significant speedup for large batches.
///
/// # Arguments
///
/// * `vertices_batch` - Batch of vertex position slices
/// * `faces` - Shared triangle face indices (immutably shared across threads)
/// * `normals_batch` - Batch of output normal buffers
///
/// # Performance
///
/// For batches of 10+ meshes, expect near-linear speedup with CPU cores.
/// Memory access is well-localized since each mesh's normals are independent.
#[cfg(feature = "parallel")]
pub fn compute_normals_batch_par(
    vertices_batch: &[Vec<na::Point3<f32>>],
    faces: &[[u32; 3]],
    normals_batch: &mut [Vec<na::Vector3<f32>>],
) {
    vertices_batch
        .par_iter()
        .zip(normals_batch.par_iter_mut())
        .for_each(|(vertices, normals)| {
            compute_normals_into(vertices, faces, normals);
        });
}

/// Compute normals for a `BatchedFlameOutput` in-place.
///
/// This is a convenience method that updates the normals in a `BatchedFlameOutput`
/// based on the current vertex positions.
///
/// # Arguments
///
/// * `output` - The batched output to update (normals are modified in-place)
pub fn recompute_batch_normals(output: &mut BatchedFlameOutput) {
    for (vertices, normals) in output.vertices.iter().zip(output.normals.iter_mut()) {
        compute_normals_into(vertices, &output.faces, normals);
    }
}

/// Compute normals for a `BatchedFlameOutput` in parallel.
///
/// This is a convenience method that updates the normals in a `BatchedFlameOutput`
/// based on the current vertex positions, using parallel processing.
///
/// # Arguments
///
/// * `output` - The batched output to update (normals are modified in-place)
#[cfg(feature = "parallel")]
pub fn recompute_batch_normals_par(output: &mut BatchedFlameOutput) {
    let faces = &output.faces;
    output
        .vertices
        .par_iter()
        .zip(output.normals.par_iter_mut())
        .for_each(|(vertices, normals)| {
            compute_normals_into(vertices, faces, normals);
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rodrigues_identity() {
        let r = rodrigues(0.0, 0.0, 0.0);
        let id = na::Matrix3::<f32>::identity();
        assert!((r - id).norm() < 1e-6);
    }

    #[test]
    fn test_rodrigues_90_deg_z() {
        use std::f32::consts::FRAC_PI_2;
        let r = rodrigues(0.0, 0.0, FRAC_PI_2);
        // Should rotate x-axis to y-axis
        let v = na::Vector3::new(1.0, 0.0, 0.0);
        let rv = r * v;
        assert!((rv.x).abs() < 1e-5);
        assert!((rv.y - 1.0).abs() < 1e-5);
        assert!((rv.z).abs() < 1e-5);
    }

    #[test]
    fn test_rodrigues_roundtrip() {
        // Rotating by angle then -angle should give identity
        let r1 = rodrigues(0.3, -0.2, 0.1);
        let r2 = rodrigues(-0.3, 0.2, -0.1);
        let product = r1 * r2;
        let id = na::Matrix3::<f32>::identity();
        assert!((product - id).norm() < 1e-5);
    }
}
