# oxigaf-flame: Implementation Plan

> **📋 Design Document Status**
>
> **Last Updated:** 2026-02-09
> **Implementation Status:** ~85% complete
> **Current Status:** See `crates/oxigaf-flame/TODO.md` for up-to-date progress
>
> **✨ Key Achievements Beyond Original Plan:**
> - **SIMD Acceleration** (not in plan): 2-4× speedup for Rodrigues rotation and blend shapes using portable_simd
> - **Parallel Batch Processing** (not in plan): Near-linear scalability with CPU cores using rayon
> - **FlameParamsBuilder** (not in plan): Ergonomic API for parameter construction
> - **Comprehensive Benchmarking Suite** (5 benchmarks): flame, LBS, Rodrigues, normal map, SIMD ops
> - **Property-based Testing** (not in plan): Using proptest for robust validation
> - **BatchBufferPool** (not in plan): Memory-efficient batch processing
>
> **⚠️ Significant Deviations:**
> - **Using .npy instead of .npz**: Individual .npy files instead of compressed .npz format (easier to work with)
> - **No safetensors yet**: Deferred for future version (plan suggested it as optional)
>
> **❌ Not Yet Implemented:**
> - **safetensors support** (0%) - Load FLAME model from .safetensors format
> - **FlameSequence** loader (0%) - Multi-frame parameter loading from tracking results
> - **Mesh export** (0%) - OBJ/PLY export functions
> - **KD-tree integration** (0%) - Nearest-vertex queries
> - **GPU normal map renderer** (0%) - wgpu-based rasterizer (optional feature)
> - **UV texture mapping** (0%) - Texture coordinate support (low priority)
> - **Landmarks** (0%) - Static/dynamic landmark extraction (low priority)
> - **Vertex masks** (0%) - Face region segmentation (medium priority)
>
> **📊 Implementation Details:**
> - ✅ Core LBS: 100% - All steps fully implemented and tested
> - ✅ CPU Normal Map Renderer: 100% - Z-buffer rasterization with world-space normals
> - ✅ Mesh Surface Sampling: 100% - Barycentric coordinate-based sampling
> - ✅ SIMD Optimizations: 100% - Feature-gated, nightly Rust required
> - ✅ Parallel Processing: 100% - Rayon-based batch operations
> - ⬜ GPU Rasterizer: 0% - Lower priority (CPU is fast enough)
> - ⬜ Sequence Loading: 0% - Needed for video input pipeline
> - ⬜ Format Support: 50% - .npy ✅, .safetensors ❌
>
> **🎯 To reach v1.0 (estimated 1-2 weeks):**
> 1. Implement safetensors support (~2-3 days)
> 2. Implement FlameSequence loader (~2-3 days)
> 3. Implement PLY export (~1-2 days)
> 4. Optional: Vertex masks (if needed for Gaussian init) (~1-2 days)
>
> **📈 Test Coverage: 43 tests (all passing)**
> - Unit tests: 10
> - Integration tests: 33
> - Property-based tests: Yes (proptest)
> - Benchmarks: 5 comprehensive suites
> - Performance: ~1-2ms for 5023 vertices on modern CPUs
>
> **🏆 Code Quality:**
> - No unwrap policy enforced
> - All files under 700 lines
> - Total: 2,934 lines (within refactoring policy)
> - SIMD features: Feature-gated for stable Rust compatibility

---

## FLAME Parametric Head Model Module for OxiGAF

---

## 1. FLAME Model Background Summary

### 1.1 What is FLAME?

**FLAME** (Faces Learned with an Articulated Model and Expressions) is a lightweight statistical 3D head model published at SIGGRAPH Asia 2017. It is learned from over 33,000 accurately aligned 3D head scans. FLAME combines:

- A **linear identity shape space** (trained from 3,800 subjects)
- An **articulated jaw, neck, and eyeball** joint system
- **Pose-dependent corrective blendshapes** (posedirs)
- **Global expression blendshapes** (expression component of shapedirs)

FLAME is the standard head model used across dozens of avatar reconstruction papers, including GaussianAvatars, GAF, DECA, EMOCA, MICA, and many more.

### 1.2 Mesh Topology

| Property | Value |
|---|---|
| **Vertices** | 5,023 |
| **Faces (triangles)** | 9,976 |
| **Joints** | 5 (root/global, neck, jaw, left eye, right eye) |
| **UV coordinates** | Available separately (texture_data files) |

The mesh includes the full head (face, ears, scalp, partial neck). Vertex region masks are available for semantic segmentation (forehead, nose, lips, eyes, etc.).

### 1.3 Parameter Spaces

| Parameter | Symbol | Dimensions | Description |
|---|---|---|---|
| **Shape** | β | 300 (typically use first 100) | Identity-dependent shape PCA coefficients |
| **Expression** | ψ | 100 (typically use first 50) | Expression PCA coefficients (independent of identity) |
| **Global rotation** | θ_global | 3 (axis-angle) | Head rotation in world space |
| **Neck pose** | θ_neck | 3 (axis-angle) | Neck rotation |
| **Jaw pose** | θ_jaw | 3 (axis-angle) | Jaw opening/closing rotation |
| **Left eye pose** | θ_leye | 3 (axis-angle) | Left eyeball rotation |
| **Right eye pose** | θ_reye | 3 (axis-angle) | Right eyeball rotation |
| **Translation** | t | 3 | Global 3D translation |

**Total pose DoF**: 5 joints × 3 = 15 (axis-angle representation).

**Full pose vector layout** (as used in FLAME_PyTorch):
```
full_pose = [θ_global(3), θ_neck(3), θ_jaw(3), θ_leye(3), θ_reye(3)]  → 15 values
```

Note: In the FLAME_PyTorch codebase, pose_params is [6] = [global_rot(3), jaw(3)], with neck and eye poses separate. In smplx codebase, each is a separate argument.

### 1.4 Model Weight File Format

The official FLAME model is distributed as **Python pickle (.pkl)** files:

- `generic_model.pkl` — gender-neutral model
- `female_model.pkl` — female model
- `male_model.pkl` — male model
- `flame2023.pkl` — Updated 2023 version with revised eye region
- `flame2023_Open.pkl` — CC-BY-4.0 licensed open version (Nov 2025)

**PKL file keys** (loaded via `pickle.load(f, encoding="latin1")`):

| Key | Shape | Description |
|---|---|---|
| `v_template` | [5023, 3] | Mean template mesh vertices |
| `f` | [9976, 3] | Triangle face indices (int) |
| `shapedirs` | [5023, 3, 400] | Shape + Expression blend shape displacements. First 300 cols = shape, last 100 cols = expression |
| `posedirs` | [5023, 3, 36] | Pose-dependent corrective blend shapes. 36 = (5-1 joints) × 9 rotation matrix entries |
| `J_regressor` | [5, 5023] sparse | Joint regressor: joint_positions = J_regressor × shaped_vertices |
| `kintree_table` | [2, 5] | Kinematic tree. Row 0 = parent indices, Row 1 = child indices. Root parent = -1 (set manually) |
| `weights` | [5023, 5] | LBS skinning weights per vertex per joint |
| `J` | [5, 3] | Default joint locations (before shape deformation) |

Some PKL files use chumpy arrays (lazy-evaluation arrays from the chumpy library). These must be converted to plain numpy arrays with `np.array(chumpy_obj)`.

### 1.5 Additional Data Files

| File | Description |
|---|---|
| `flame_static_embedding.pkl` | Static 3D landmark face indices + barycentric coordinates (51 landmarks) |
| `flame_dynamic_embedding.npy` | Dynamic contour landmarks look-up table (pose-dependent) |
| `FLAME_masks.pkl` | Per-vertex face region masks |
| `texture_data_*.npy` | UV mapping data for various resolutions (256, 512, 1024, 2048) |

### 1.6 Licensing

- **FLAME 2023 Open** (`flame2023_Open.pkl`): **CC-BY-4.0** (open source, commercial use allowed) — Released Nov 2025.
- Earlier FLAME versions: Non-commercial research only.
- For OxiGAF, we should target `flame2023_Open.pkl` as the default.

---

## 2. Linear Blend Skinning (LBS) for FLAME

The FLAME forward pass follows the SMPL family LBS algorithm. The canonical reference is `smplx/lbs.py::lbs()`.

### 2.1 Algorithm Steps

Given inputs: β (shape), ψ (expression), θ (full_pose as 5×3 axis-angles), t (translation):

```
Step 1: Shape + Expression Blending
    betas_combined = concat(β, ψ)                          # [400]
    v_shaped = v_template + Σ_l (betas_combined[l] * shapedirs[:,:,l])  # [5023, 3]
    // Equivalent: v_shaped = v_template + einsum('bl,mkl->bmk', betas, shapedirs)

Step 2: Joint Regression
    J = J_regressor × v_shaped                              # [5, 3]
    // Joints are inferred from the shape-deformed vertices (not fixed!)

Step 3: Pose → Rotation Matrices (Rodrigues)
    For each of 5 joints:
        R_i = rodrigues(θ_i)                                # [3, 3]
    rot_mats = [R_0, R_1, R_2, R_3, R_4]                   # [5, 3, 3]

Step 4: Pose-Dependent Corrective Blendshapes
    pose_feature = flatten(rot_mats[1:] - I_3×3)           # [36] (4 non-root joints × 9)
    pose_offsets = posedirs × pose_feature                  # [5023, 3]
    v_posed = v_shaped + pose_offsets

Step 5: Kinematic Chain — Global Rigid Transformations
    For each joint i, compute world-space transform A_i:
        Build local transform: T_i = [R_i | (J_i - R_i × J_parent_i); 0 0 0 1]
        Chain: A_i = A_parent(i) × T_i
    Subtract rest-pose joint to get relative transforms:
        A'_i = A_i - [I | A_i × J_homogeneous_i]

Step 6: Skinning
    For each vertex v:
        T_v = Σ_j (weights[v, j] * A'_j)                   # [4, 4] weighted transform
        v_final = T_v × [v_posed; 1]                        # [3]

Step 7: Translation
    v_final += t
```

### 2.2 Rodrigues Formula (Axis-Angle → Rotation Matrix)

```
Input: axis-angle vector r ∈ R³
angle = ||r||
axis = r / angle
K = skew_symmetric(axis)     // [0 -z y; z 0 -x; -y x 0]
R = I + sin(angle) * K + (1 - cos(angle)) * K²
```

This is implemented as `batch_rodrigues()` in smplx.

### 2.3 Kinematic Tree for FLAME

```
Joint 0: Root (global)     parent = -1 (none)
Joint 1: Neck              parent = 0
Joint 2: Jaw               parent = 1
Joint 3: Left Eye          parent = 1
Joint 4: Right Eye         parent = 1
```

Chain: Global → Neck → {Jaw, Left Eye, Right Eye}

---

## 3. Data Format Handling Strategy

### 3.1 Problem: PKL Files in Rust

Python pickle files cannot be trivially loaded in Rust:
- They contain Python-specific serialization (opcodes, class references).
- FLAME PKLs may contain chumpy arrays and scipy sparse matrices.
- There is no production-quality Rust pickle parser.

### 3.2 Recommended Strategy: Offline Conversion + .npz/.safetensors Loading

**Two-phase approach:**

#### Phase A: Conversion Script (Python, one-time)

Provide a Python script `convert_flame.py` that:
1. Loads the `.pkl` file with pickle (handling chumpy/scipy).
2. Converts all arrays to dense numpy float32/int32.
3. Saves as one of:
   - **`.npz`** (numpy zip format — simplest, well-understood)
   - **`.safetensors`** (HuggingFace format — Rust crate exists)
   - **Custom `.bin` + `.json` manifest** (most control)

Recommended output: **`.npz`** for Phase 1, migrate to **`.safetensors`** later.

```python
# convert_flame.py (sketch)
import pickle, numpy as np

with open("flame2023_Open.pkl", "rb") as f:
    model = pickle.load(f, encoding="latin1")

data = {
    "v_template": np.array(model["v_template"], dtype=np.float32),     # [5023, 3]
    "f": np.array(model["f"], dtype=np.int32),                         # [9976, 3]
    "shapedirs": np.array(model["shapedirs"], dtype=np.float32),       # [5023, 3, 400]
    "posedirs": np.array(model["posedirs"], dtype=np.float32),         # [5023, 3, 36]
    "J_regressor": np.array(model["J_regressor"].todense(), dtype=np.float32),  # [5, 5023]
    "kintree_table": np.array(model["kintree_table"], dtype=np.int32), # [2, 5]
    "weights": np.array(model["weights"], dtype=np.float32),           # [5023, 5]
}
np.savez("flame2023_open.npz", **data)
```

#### Phase B: Rust .npz Loader

Use the `ndarray-npy` crate to read `.npy`/`.npz` files. Alternatively, use `safetensors` crate for `.safetensors` format.

### 3.3 Per-Frame FLAME Parameters

GAF and most trackers output per-frame FLAME parameters (from DECA, MICA, metrical-tracker, etc.) as:
- **JSON/CSV** per frame, or
- **numpy .npy arrays** with shape `[N_frames, param_dim]`

We need a loader that accepts:
- A directory of per-frame `.json` files
- A single `.npz` file with `shape`, `expression`, `pose`, `neck_pose`, `eye_pose`, `translation` arrays

### 3.4 Future: Direct PKL Support

Optionally, a Rust-native pickle parser could be built later for convenience, but this is low priority given the conversion approach.

---

## 4. Key Structs and Their Fields

### 4.1 Core Model Data

```rust
/// The loaded FLAME model weights (immutable after loading).
pub struct FlameModel {
    /// Mean template vertices [5023, 3]
    pub v_template: Array2<f32>,          // nalgebra: Matrix<f32, Dyn, U3>

    /// Triangle face indices [9976, 3]
    pub faces: Array2<u32>,

    /// Shape + Expression blend shapes [5023, 3, 400]
    /// First 300 columns = shape, last 100 = expression
    pub shapedirs: Array3<f32>,

    /// Pose-dependent corrective blend shapes [36, 5023*3]
    /// Transposed for efficient matmul: pose_feature × posedirs → offsets
    pub posedirs: Array2<f32>,

    /// Joint regressor (dense) [5, 5023]
    pub j_regressor: Array2<f32>,

    /// Kinematic tree parent indices [5]
    /// parents[0] = -1 (root), parents[1] = 0, parents[2] = 1, etc.
    pub parents: Vec<i32>,

    /// LBS skinning weights [5023, 5]
    pub lbs_weights: Array2<f32>,

    /// Number of shape components available (300)
    pub num_shape_params: usize,

    /// Number of expression components available (100)
    pub num_expression_params: usize,

    /// Number of joints (5)
    pub num_joints: usize,

    /// Number of vertices (5023)
    pub num_vertices: usize,
}
```

### 4.2 Per-Frame Parameters

```rust
/// FLAME parameters for a single frame.
#[derive(Clone, Debug)]
pub struct FlameParams {
    /// Identity shape coefficients [n_shape] (typically 100 of 300)
    pub shape: Vec<f32>,

    /// Expression coefficients [n_expr] (typically 50 of 100)
    pub expression: Vec<f32>,

    /// Global head rotation, axis-angle [3]
    pub global_orient: [f32; 3],

    /// Neck rotation, axis-angle [3]
    pub neck_pose: [f32; 3],

    /// Jaw rotation, axis-angle [3]
    pub jaw_pose: [f32; 3],

    /// Left eye rotation, axis-angle [3]
    pub left_eye_pose: [f32; 3],

    /// Right eye rotation, axis-angle [3]
    pub right_eye_pose: [f32; 3],

    /// Global translation [3]
    pub translation: [f32; 3],
}

impl Default for FlameParams {
    fn default() -> Self {
        // All zeros = mean shape, neutral expression, no rotation, no translation
    }
}
```

### 4.3 Output Mesh

```rust
/// Triangle mesh output from the FLAME forward pass.
pub struct FlameMesh {
    /// Vertex positions [N_vertices, 3]
    pub vertices: Vec<[f32; 3]>,

    /// Triangle face indices [N_faces, 3] (shared reference from FlameModel)
    pub faces: Arc<Vec<[u32; 3]>>,

    /// Joint positions [N_joints, 3]
    pub joints: Vec<[f32; 3]>,

    /// Per-vertex normals (computed on demand) [N_vertices, 3]
    pub normals: Option<Vec<[f32; 3]>>,
}

impl FlameMesh {
    /// Compute per-vertex normals by averaging adjacent face normals.
    pub fn compute_normals(&mut self) { ... }
}
```

### 4.4 Normal Map Renderer

```rust
/// Configuration for Normal Map rendering.
pub struct NormalMapConfig {
    /// Output image width
    pub width: u32,
    /// Output image height
    pub height: u32,
    /// Camera intrinsics (focal length, principal point)
    pub camera: CameraIntrinsics,
    /// Background value (typically [0, 0, 0] or [128, 128, 128])
    pub background: [u8; 3],
}

pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Renders normal maps from FLAME meshes.
pub struct NormalMapRenderer {
    config: NormalMapConfig,
    // Internal rasterizer state (CPU or GPU)
}
```

### 4.5 Sequence Data

```rust
/// A sequence of FLAME parameters loaded from a tracking result.
pub struct FlameSequence {
    /// Shared shape parameters (constant across frames for one identity)
    pub shape: Vec<f32>,

    /// Per-frame parameters
    pub frames: Vec<FlameFrameData>,
}

pub struct FlameFrameData {
    pub expression: Vec<f32>,
    pub global_orient: [f32; 3],
    pub neck_pose: [f32; 3],
    pub jaw_pose: [f32; 3],
    pub left_eye_pose: [f32; 3],
    pub right_eye_pose: [f32; 3],
    pub translation: [f32; 3],
}
```

---

## 5. Key Functions/Methods with Signatures

### 5.1 Model Loading

```rust
impl FlameModel {
    /// Load FLAME model from a converted .npz file.
    pub fn from_npz(path: &Path) -> Result<Self, FlameError>;

    /// Load FLAME model from a .safetensors file.
    pub fn from_safetensors(path: &Path) -> Result<Self, FlameError>;

    /// Load from raw arrays (for embedding or testing).
    pub fn from_arrays(
        v_template: Array2<f32>,
        faces: Array2<u32>,
        shapedirs: Array3<f32>,
        posedirs: Array2<f32>,
        j_regressor: Array2<f32>,
        parents: Vec<i32>,
        lbs_weights: Array2<f32>,
    ) -> Self;
}
```

### 5.2 Forward Pass (LBS)

```rust
impl FlameModel {
    /// Compute mesh vertices from FLAME parameters.
    /// This is the core forward pass implementing Linear Blend Skinning.
    pub fn forward(&self, params: &FlameParams) -> FlameMesh;

    /// Batch forward pass for multiple frames (parallelizable).
    pub fn forward_batch(&self, params: &[FlameParams]) -> Vec<FlameMesh>;
}
```

### 5.3 Internal LBS Functions

```rust
/// Rodrigues rotation: axis-angle [3] → rotation matrix [3, 3]
fn rodrigues(axis_angle: &[f32; 3]) -> Matrix3<f32>;

/// Compute blend shape displacements.
/// betas: [N_betas], shapedirs: [V, 3, N_betas] → vertex offsets [V, 3]
fn blend_shapes(betas: &[f32], shapedirs: &Array3<f32>) -> Array2<f32>;

/// Compute joint locations from shaped vertices.
/// j_regressor: [J, V], v_shaped: [V, 3] → joints [J, 3]
fn vertices_to_joints(j_regressor: &Array2<f32>, v_shaped: &Array2<f32>) -> Array2<f32>;

/// Build kinematic chain transforms.
/// rot_mats: [J, 3, 3], joints: [J, 3], parents: [J] → (posed_joints, rel_transforms)
fn batch_rigid_transform(
    rot_mats: &[Matrix3<f32>],
    joints: &Array2<f32>,
    parents: &[i32],
) -> (Array2<f32>, Vec<Matrix4<f32>>);

/// Full LBS function.
fn lbs(
    betas: &[f32],        // combined shape + expression coefficients
    full_pose: &[[f32; 3]; 5], // 5 axis-angle joint rotations
    v_template: &Array2<f32>,
    shapedirs: &Array3<f32>,
    posedirs: &Array2<f32>,
    j_regressor: &Array2<f32>,
    parents: &[i32],
    lbs_weights: &Array2<f32>,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>);  // (vertices, joints)
```

### 5.4 Normal Map Rendering

```rust
impl NormalMapRenderer {
    /// Create a new renderer with given configuration.
    pub fn new(config: NormalMapConfig) -> Self;

    /// Render a normal map from a FLAME mesh.
    /// Returns an RGB image where (R, G, B) = ((nx+1)/2, (ny+1)/2, (nz+1)/2) × 255.
    pub fn render(&self, mesh: &FlameMesh) -> image::RgbImage;

    /// Render normal maps for a batch of meshes (for multi-view).
    pub fn render_batch(
        &self,
        mesh: &FlameMesh,
        extrinsics: &[CameraExtrinsics],
    ) -> Vec<image::RgbImage>;
}
```

### 5.5 Mesh Utilities

```rust
impl FlameMesh {
    /// Compute per-vertex normals from face normals (area-weighted average).
    pub fn compute_normals(&mut self);

    /// Sample a point on the mesh surface given a face index and barycentric coordinates.
    pub fn sample_point(&self, face_idx: usize, bary: [f32; 3]) -> [f32; 3];

    /// Build a KD-tree for nearest-vertex queries.
    pub fn build_kdtree(&self) -> KdTree<f32, usize, [f32; 3]>;

    /// Export mesh to .obj file.
    pub fn export_obj(&self, path: &Path) -> Result<(), std::io::Error>;

    /// Export mesh to .ply file.
    pub fn export_ply(&self, path: &Path) -> Result<(), std::io::Error>;
}
```

### 5.6 Parameter Loading

```rust
impl FlameSequence {
    /// Load per-frame parameters from a directory of JSON files.
    pub fn from_json_dir(dir: &Path) -> Result<Self, FlameError>;

    /// Load per-frame parameters from a .npz file with tracked data.
    pub fn from_npz(path: &Path) -> Result<Self, FlameError>;

    /// Get parameters for a specific frame.
    pub fn get_frame(&self, idx: usize) -> FlameParams;

    /// Number of frames.
    pub fn len(&self) -> usize;
}
```

---

## 6. Normal Map Generation Approach

### 6.1 Pipeline

```
FlameParams → FlameModel::forward() → FlameMesh
    → compute_normals()
    → project vertices to image plane (camera intrinsics/extrinsics)
    → rasterize triangles with interpolated normals
    → encode normals to RGB
    → output image::RgbImage
```

### 6.2 Rasterization Strategy

**Recommended: CPU software rasterizer for Phase 1, wgpu for Phase 2.**

#### Phase 1: CPU Software Rasterizer

A minimal triangle rasterizer that:
1. Projects 3D vertices to 2D using perspective projection.
2. Rasterizes each triangle using scanline or half-edge approach.
3. Interpolates per-vertex normals across triangles using barycentric coords.
4. Uses a Z-buffer for depth testing.
5. Encodes the interpolated normal `(nx, ny, nz) ∈ [-1, 1]³` as `((n+1)/2 * 255)` RGB.

**Rationale:**
- Normal map rendering is not on the critical training path (generated once per optimization step).
- CPU rasterization at 512×512 for ~10K triangles is fast enough (< 5ms).
- No GPU dependency for this module — simplifies testing and CI.
- The `rasterize` crate or custom implementation (< 300 lines).

#### Phase 2: wgpu Rasterizer (Optional)

If performance becomes an issue (batch rendering many views), a wgpu compute/render pass can:
1. Upload mesh as vertex/index buffers.
2. Use a simple vertex+fragment shader that outputs world-space normals.
3. Read back the framebuffer.

### 6.3 Normal Encoding Convention

Following the GAF paper's convention:
- World-space normals (or camera-space, depending on conditioning approach).
- RGB encoding: `R = (nx + 1) / 2 * 255`, `G = (ny + 1) / 2 * 255`, `B = (nz + 1) / 2 * 255`.
- Background pixels: `[0, 0, 0]` or `[128, 128, 128]` (to be determined from GAF reference code).

### 6.4 Camera Model

```rust
pub struct CameraExtrinsics {
    /// Rotation matrix [3, 3] (world → camera)
    pub rotation: Matrix3<f32>,
    /// Translation vector [3] (world → camera)
    pub translation: [f32; 3],
}

/// Full projection: world point → pixel coordinates
fn project_point(
    point: &[f32; 3],
    extrinsics: &CameraExtrinsics,
    intrinsics: &CameraIntrinsics,
) -> (f32, f32, f32);  // (pixel_x, pixel_y, depth)
```

---

## 7. Integration Points with Other Crates

### 7.1 oxigaf-flame → oxigaf-diffusion

**What flows**: Normal map images as conditioning inputs.

```
oxigaf-flame produces:
    - Normal map images (image::RgbImage or raw tensor)
    - Camera pose information (for each rendered view)

oxigaf-diffusion consumes:
    - Normal maps as additional conditioning channels to the U-Net
    - Encoded via VAE or concatenated directly to latent space
```

**Interface contract:**
```rust
// Trait that oxigaf-diffusion expects:
pub trait NormalMapProvider {
    fn render_normal_map(
        &self,
        params: &FlameParams,
        camera: &CameraExtrinsics,
        resolution: (u32, u32),
    ) -> image::RgbImage;
}
```

### 7.2 oxigaf-flame → oxigaf-render (3DGS)

**What flows**: Initial Gaussian positions bound to the FLAME mesh surface.

The GAF paper initializes Gaussians on the FLAME mesh surface, split into:
- **Rigid Gaussians**: Fixed relative to a mesh triangle (move with FLAME).
- **Flexible Gaussians**: Loosely bound, with learnable offsets.

```rust
// Trait that oxigaf-render uses to query mesh data:
pub trait MeshSurfaceSampler {
    /// Sample N points uniformly on the mesh surface.
    /// Returns (positions, normals, face_indices, barycentric_coords).
    fn sample_surface(&self, n_points: usize) -> SurfaceSamples;

    /// Given a mesh deformation (new FlameParams), re-position the bound points.
    fn deform_bound_points(
        &self,
        face_indices: &[u32],
        bary_coords: &[[f32; 3]],
        params: &FlameParams,
    ) -> Vec<[f32; 3]>;
}

pub struct SurfaceSamples {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub face_indices: Vec<u32>,
    pub bary_coords: Vec<[f32; 3]>,
}
```

### 7.3 oxigaf-flame → oxigaf-trainer

**What flows**: Per-frame FLAME parameters drive the optimization loop.

```rust
// The trainer iterates over frames:
for frame in sequence.iter() {
    let params = sequence.get_frame(frame_idx);
    let mesh = flame_model.forward(&params);

    // Generate normal maps for conditioning
    let normal_map = renderer.render(&mesh);

    // Update Gaussian positions based on FLAME mesh
    let new_positions = sampler.deform_bound_points(&bindings, &params);

    // Feed to diffusion model and optimization...
}
```

### 7.4 Data Flow Diagram

```
                        ┌─────────────────┐
                        │  FLAME .pkl/.npz │
                        └────────┬────────┘
                                 │ load
                        ┌────────▼────────┐
                        │   FlameModel    │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
  Per-frame params        forward(params)           Surface sampling
  (from tracker)               │                        │
        │               ┌──────▼──────┐           ┌─────▼─────┐
        │               │  FlameMesh  │           │ Gaussian   │
        │               └──────┬──────┘           │ Init Pos   │
        │                      │                  └─────┬─────┘
        │               ┌──────▼──────┐                 │
        │               │ Normal Map  │                 │
        │               │  Renderer   │                 │
        │               └──────┬──────┘                 │
        │                      │                        │
        ▼                      ▼                        ▼
  ┌─────────────┐    ┌─────────────────┐    ┌───────────────────┐
  │ oxigaf-     │    │ oxigaf-         │    │ oxigaf-render     │
  │ trainer     │◄───│ diffusion       │    │ (3DGS rasterizer) │
  └─────────────┘    └─────────────────┘    └───────────────────┘
```

---

## 8. Crate Dependencies

```toml
[package]
name = "oxigaf-flame"
version = "0.1.0"
edition = "2021"

[dependencies]
# Linear algebra
nalgebra = "0.33"                   # Matrix operations, Rodrigues, transforms

# N-dimensional arrays
ndarray = "0.16"                    # For loading and manipulating model weight arrays
ndarray-npy = "0.9"                 # .npz/.npy file loading

# Image output
image = "0.25"                      # Normal map output as RGB images

# Spatial search
kiddo = "4"                         # KD-tree for nearest vertex queries (alternative to kdtree crate)

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"                  # Per-frame parameter loading

# Error handling
thiserror = "2"                     # Typed error definitions

# Optional: safetensors support
safetensors = { version = "0.4", optional = true }

# Optional: GPU normal map rendering
wgpu = { version = "23", optional = true }

[features]
default = []
safetensors = ["dep:safetensors"]
gpu-raster = ["dep:wgpu"]
```

**Note on nalgebra vs ndarray**: Use `ndarray` for bulk array storage (model weights with 3+ dimensions), and `nalgebra` for the per-vertex/per-joint math (3×3 rotation matrices, 4×4 transforms, vector operations). Convert between them at the boundary.

---

## 9. Module/File Structure

```
oxigaf-flame/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API re-exports
│   ├── model.rs                # FlameModel struct and loading
│   ├── params.rs               # FlameParams, FlameSequence, loading
│   ├── lbs.rs                  # Linear Blend Skinning implementation
│   │                           #   rodrigues(), blend_shapes(), vertices_to_joints(),
│   │                           #   batch_rigid_transform(), lbs()
│   ├── mesh.rs                 # FlameMesh, normals, sampling, export
│   ├── render/
│   │   ├── mod.rs              # NormalMapRenderer trait + config
│   │   ├── cpu_rasterizer.rs   # Software rasterizer implementation
│   │   └── gpu_rasterizer.rs   # Optional wgpu-based rasterizer (behind feature flag)
│   ├── camera.rs               # CameraIntrinsics, CameraExtrinsics, projection
│   └── error.rs                # FlameError enum
├── scripts/
│   └── convert_flame.py        # Python conversion script (pkl → npz)
└── tests/
    ├── test_lbs.rs             # Verify LBS against Python reference outputs
    ├── test_model_loading.rs   # Test .npz loading
    ├── test_normals.rs         # Verify normal computation
    └── test_render.rs          # Visual regression tests for normal maps
```

---

## 10. Testing & Validation Strategy

### 10.1 Ground Truth from Python

Generate reference data using the FLAME_PyTorch or smplx code:

```python
# generate_test_data.py
import torch, numpy as np
from smplx import FLAME

model = FLAME(model_path="flame2023_Open.pkl", ...)
output = model(betas=..., expression=..., global_orient=..., jaw_pose=..., ...)

np.savez("test_reference.npz",
    input_shape=betas.numpy(),
    input_expression=expression.numpy(),
    input_pose=global_orient.numpy(),
    output_vertices=output.vertices.detach().numpy(),
    output_joints=output.joints.detach().numpy(),
)
```

### 10.2 Test Cases

1. **Zero parameters**: All params = 0 → output should equal `v_template`.
2. **Shape only**: Set β[0] = ±3σ, verify vertex displacement matches Python.
3. **Expression only**: Set ψ[0] = ±3σ, verify.
4. **Jaw rotation**: Open jaw by π/8, compare vertices.
5. **Full combination**: Random params, compare vertices to Python output with tolerance < 1e-4.
6. **Normal computation**: Compare computed normals against known analytical cases (e.g., a flat triangle).
7. **Normal map rendering**: Pixel-level comparison against a reference render.

### 10.3 Tolerance

- Vertex positions: `max |v_rust - v_python| < 1e-4` (accounting for f32 vs f64).
- Normals: `max |n_rust - n_python| < 1e-3`.
- Normal map pixels: `max |pixel_rust - pixel_python| < 2` (rounding differences).

---

## 11. Risks and Open Questions

### 11.1 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **PKL format complexity**: Some FLAME PKLs use chumpy lazy arrays and scipy sparse matrices | Medium | Conversion script handles this. Document which PKL versions are supported. Recommend `flame2023_Open.pkl`. |
| **Numerical precision**: f32 vs f64 differences in Rodrigues / LBS may accumulate | Low | Test against Python reference. Use f32 throughout (matches GPU precision). The PyTorch impl uses f32 by default. |
| **FLAME version fragmentation**: 2017, 2019, 2020, 2023, 2023_Open all have slightly different parameters | Medium | Target `flame2023_Open.pkl` (CC-BY-4.0). Support older versions via same PKL key schema. |
| **Normal map convention mismatch**: GAF may use a specific normal encoding that differs from standard | Medium | Need to verify against GAF reference code. Parameterize the encoding. |
| **Performance for batch rendering**: CPU rasterizer may be slow for many views | Low | 10K triangles at 512×512 is manageable. Add wgpu path behind feature flag. |
| **Sparse J_regressor**: Original is scipy sparse, conversion to dense increases memory | Low | Dense [5, 5023] is only 100KB. Acceptable. |

### 11.2 Open Questions

1. **Which FLAME version does GAF use exactly?** The paper likely uses FLAME 2020 or 2023. Need to check the GAF code release (if available) or paper supplementary.

2. **Normal map coordinate space**: Does GAF use world-space normals or camera-space normals for conditioning? This affects the rendering pipeline.

3. **Expression parameter conversion**: FLAME 2023 and FLAME 2023_Open have different expression spaces. A conversion tool exists (`flame_to_flame_open_converter`). Do we need to support both?

4. **Texture UV mapping**: Is UV mapping needed for oxigaf-flame, or only for oxigaf-render? If the diffusion model needs texture-mapped inputs, we'll need the UV data too.

5. **Dynamic landmarks**: Are the dynamic contour landmarks (pose-dependent) needed for OxiGAF, or only for fitting? If only the forward pass and normal maps are needed, we can skip the landmark system initially.

6. **Thread safety**: Should `FlameModel` be `Send + Sync`? Yes — it's immutable after loading. `ndarray` arrays are `Send + Sync` when owned. This enables parallel frame processing.

7. **WASM target**: Should oxigaf-flame compile to WASM for browser demos? If so, avoid filesystem-dependent code in the core (accept byte slices instead of paths).

8. **Vertex masks**: Does GAF use FLAME vertex masks (e.g., only face region) for Gaussian initialization? If so, we need to load `FLAME_masks.pkl` too.

---

## 12. Implementation Phases

### Phase 1: Core LBS (Week 1-2)
- [x] `convert_flame.py` script
- [x] `FlameModel::from_npz()`
- [x] `FlameParams` struct + defaults
- [x] `lbs.rs`: `rodrigues()`, `blend_shapes()`, `vertices_to_joints()`, `batch_rigid_transform()`, `lbs()`
- [x] `FlameModel::forward()`
- [x] Unit tests against Python reference
- [x] `FlameMesh::compute_normals()`

### Phase 2: Normal Map Rendering (Week 3)
- [x] `CameraIntrinsics`, `CameraExtrinsics`, projection
- [x] CPU software rasterizer with Z-buffer
- [x] Normal encoding to RGB image
- [x] `NormalMapRenderer::render()`
- [x] Visual tests

### Phase 3: Integration Interfaces (Week 4)
- [ ] `FlameSequence` loader (JSON + npz)
- [x] `MeshSurfaceSampler` implementation
- [x] `NormalMapProvider` trait
- [ ] `FlameMesh::export_obj()` / `export_ply()`
- [ ] KD-tree integration
- [x] Documentation + examples

### Phase 4: Polish (Week 5)
- [ ] Optional safetensors support
- [ ] Optional wgpu normal map renderer
- [x] Benchmarks
- [x] CI integration
- [ ] FLAME model auto-download/caching utility

---

## 13. References

1. **FLAME Paper**: Li et al., "Learning a model of facial shape and expression from 4D scans", SIGGRAPH Asia 2017. https://doi.org/10.1145/3130800.3130813
2. **GAF Paper**: Tang et al., "GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-View Diffusion", CVPR 2025. https://arxiv.org/abs/2412.10209
3. **FLAME_PyTorch**: https://github.com/soubhiksanyal/FLAME_PyTorch — Reference PyTorch implementation
4. **smplx**: https://github.com/vchoutas/smplx — Canonical LBS implementation (`smplx/lbs.py`)
5. **TF_FLAME**: https://github.com/TimoBolkart/TF_FLAME — TensorFlow implementation with batch_smpl.py
6. **FLAME Universe**: https://github.com/TimoBolkart/FLAME-Universe — Comprehensive list of FLAME resources
7. **FLAME 2023 Open**: CC-BY-4.0 licensed model, https://flame.is.tue.mpg.de/
