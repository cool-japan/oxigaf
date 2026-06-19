//! Gaussian initialization from a mesh surface.
//!
//! This module places 3D Gaussians on a triangle mesh surface using area-weighted
//! random sampling. It is designed to be used at the start of training to initialize
//! Gaussians on the FLAME face mesh before optimization.
//!
//! ## Design
//!
//! - No dependency on `oxigaf-flame` — takes raw geometry arrays.
//! - Scale is stored in log-space: `actual_scale = exp(stored_scale)`.
//! - Opacity is stored in logit-space: `actual_opacity = sigmoid(stored_opacity)`.
//! - Rotation is stored as quaternion `[x, y, z, w]`.
//! - SH coefficients: degree-0 DC term encodes color via `color = 0.5 + SH_C0 * sh_dc`.

use crate::{
    gaussian::{GaussianAttributes, GaussianModel},
    RenderError,
};

/// The zeroth-degree spherical harmonics constant.
const SH_C0: f32 = 0.28209_f32;

/// Configuration for Gaussian initialization from a mesh surface.
#[derive(Debug, Clone)]
pub struct GaussianInitConfig {
    /// Target number of Gaussians to place on the mesh surface.
    pub num_gaussians: usize,
    /// Scale factor multiplied with `sqrt(face_area)` to obtain actual scale.
    /// Default: 0.05.
    pub scale_factor: f32,
    /// Minimum log-scale value (clamp from below). Default: -6.0.
    pub min_scale_log: f32,
    /// Maximum log-scale value (clamp from above). Default: 0.0.
    pub max_scale_log: f32,
    /// Initial opacity stored in logit-space. Default: `ln(0.1/0.9) ≈ -2.197`.
    pub initial_opacity_logit: f32,
    /// Spherical harmonics degree (0 = DC term only). Default: 0.
    pub sh_degree: u32,
    /// Mean color used to initialize SH DC coefficients. Default: `[0.5, 0.5, 0.5]`.
    pub mean_color: [f32; 3],
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for GaussianInitConfig {
    fn default() -> Self {
        Self {
            num_gaussians: 10_000,
            scale_factor: 0.05,
            min_scale_log: -6.0,
            max_scale_log: 0.0,
            // ln(0.1 / 0.9)
            initial_opacity_logit: (0.1_f32 / 0.9_f32).ln(),
            sh_degree: 0,
            mean_color: [0.5, 0.5, 0.5],
            seed: 42,
        }
    }
}

/// A minimal xorshift64 PRNG for deterministic, dependency-free random sampling.
struct Xorshift64(u64);

impl Xorshift64 {
    /// Create a new `Xorshift64` with the given seed.
    ///
    /// If `seed` is zero, substitutes `0xDEAD_BEEF` to avoid the all-zero fixed point.
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0xDEAD_BEEF_u64 } else { seed };
        Self(state)
    }

    /// Advance the state and return the next pseudo-random `u64`.
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Return a pseudo-random `f32` in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

/// Initializes a [`GaussianModel`] by sampling Gaussians on a triangle mesh surface.
///
/// Call [`GaussianInitializer::new`] once (precomputes areas), then call
/// [`GaussianInitializer::initialize`] with a [`GaussianInitConfig`] to produce a model.
pub struct GaussianInitializer {
    /// Triangle face indices stored as owned data.
    faces: Vec<[u32; 3]>,
    /// Vertex positions stored as owned data.
    vertices: Vec<[f32; 3]>,
    /// Per-vertex normals (optional).
    normals: Option<Vec<[f32; 3]>>,
    /// Area of each triangle. Entries for degenerate (zero-area) faces are 0.0.
    face_areas: Vec<f32>,
    /// Prefix sum of areas (length = num_faces + 1). `prefix[i+1] = prefix[i] + area[i]`.
    area_prefix: Vec<f64>,
    /// Total surface area (sum of all face areas).
    total_area: f64,
}

impl GaussianInitializer {
    /// Build a `GaussianInitializer` from raw mesh geometry.
    ///
    /// # Arguments
    /// - `vertices`: positions of mesh vertices.
    /// - `faces`: triangle face indices (each entry references three vertices).
    /// - `normals`: optional per-vertex normals; if `None`, normals are computed from the triangle.
    ///
    /// # Errors
    /// Returns [`RenderError::Rasterize`] if the mesh has no faces or zero total area.
    pub fn new(
        vertices: &[[f32; 3]],
        faces: &[[u32; 3]],
        normals: Option<&[[f32; 3]]>,
    ) -> Result<Self, RenderError> {
        if faces.is_empty() {
            return Err(RenderError::Rasterize(
                "Cannot initialize Gaussians from an empty mesh (no faces)".to_string(),
            ));
        }

        let num_faces = faces.len();
        let mut face_areas = Vec::with_capacity(num_faces);
        let mut area_prefix = Vec::with_capacity(num_faces + 1);
        area_prefix.push(0.0_f64);

        for (face_idx, &[i0, i1, i2]) in faces.iter().enumerate() {
            let v0 = get_vertex(vertices, i0, face_idx)?;
            let v1 = get_vertex(vertices, i1, face_idx)?;
            let v2 = get_vertex(vertices, i2, face_idx)?;

            let area = triangle_area(v0, v1, v2);
            face_areas.push(area);
            let prev = area_prefix[face_idx];
            area_prefix.push(prev + area as f64);
        }

        let total_area = *area_prefix.last().unwrap_or(&0.0);
        if total_area <= 0.0 {
            return Err(RenderError::Rasterize(
                "Mesh has zero total surface area; cannot sample Gaussians".to_string(),
            ));
        }

        Ok(Self {
            faces: faces.to_vec(),
            vertices: vertices.to_vec(),
            normals: normals.map(|n| n.to_vec()),
            face_areas,
            area_prefix,
            total_area,
        })
    }

    /// Initialize a [`GaussianModel`] by placing `config.num_gaussians` Gaussians
    /// on the mesh surface using area-weighted random triangle sampling.
    ///
    /// # Errors
    /// Returns [`RenderError::Rasterize`] on invalid configuration.
    pub fn initialize(&self, config: &GaussianInitConfig) -> Result<GaussianModel, RenderError> {
        if config.num_gaussians == 0 {
            return Err(RenderError::Rasterize(
                "num_gaussians must be greater than zero".to_string(),
            ));
        }

        let n = config.num_gaussians;
        let mut rng = Xorshift64::new(config.seed);

        // Determine SH layout.
        let sh_total = ((config.sh_degree + 1) * (config.sh_degree + 1) * 3) as usize;

        // Precompute SH DC values from mean_color.
        // sh_dc[c] = (mean_color[c] - 0.5) / SH_C0
        let sh_dc_r = (config.mean_color[0] - 0.5) / SH_C0;
        let sh_dc_g = (config.mean_color[1] - 0.5) / SH_C0;
        let sh_dc_b = (config.mean_color[2] - 0.5) / SH_C0;

        let mut gaussians = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(n * sh_total);
        let mut face_indices = Vec::with_capacity(n);
        let mut barycentric = Vec::with_capacity(n);
        let mut local_offsets = Vec::with_capacity(n);
        let mut is_rigid = Vec::with_capacity(n);

        for _ in 0..n {
            // --- Area-weighted triangle selection ---
            let face_idx = self.sample_face(&mut rng);

            let [i0, i1, i2] = self.faces[face_idx];
            let v0 = self.vertices[i0 as usize];
            let v1 = self.vertices[i1 as usize];
            let v2 = self.vertices[i2 as usize];
            let face_area = self.face_areas[face_idx];

            // --- Barycentric coordinates ---
            let (bary_u, bary_v, bary_w) = sample_barycentric(&mut rng);

            // --- Position via barycentric interpolation ---
            let position = [
                bary_u * v0[0] + bary_v * v1[0] + bary_w * v2[0],
                bary_u * v0[1] + bary_v * v1[1] + bary_w * v2[1],
                bary_u * v0[2] + bary_v * v1[2] + bary_w * v2[2],
            ];

            // --- Log-scale: ln(sqrt(face_area) * scale_factor), clamped ---
            let log_scale = if face_area > 0.0 {
                let s = (face_area.sqrt() * config.scale_factor).ln();
                s.clamp(config.min_scale_log, config.max_scale_log)
            } else {
                config.min_scale_log
            };

            // --- Rotation from surface normal → TBN → quaternion ---
            let rotation = compute_rotation(FaceGeom {
                i0,
                i1,
                i2,
                v0: &v0,
                v1: &v1,
                v2: &v2,
                bary: [bary_u, bary_v, bary_w],
                normals: self.normals.as_deref(),
            });

            // --- Gaussian attributes ---
            gaussians.push(GaussianAttributes {
                position,
                _pad0: 0.0,
                rotation,
                scale: [log_scale, log_scale, log_scale],
                opacity: config.initial_opacity_logit,
            });

            // --- SH coefficients ---
            // DC terms (first 3 floats per Gaussian, one per channel).
            sh_coeffs.push(sh_dc_r);
            sh_coeffs.push(sh_dc_g);
            sh_coeffs.push(sh_dc_b);
            // Higher-degree SH coefficients initialized to 0.
            let higher_count = sh_total.saturating_sub(3);
            sh_coeffs.extend(std::iter::repeat_n(0.0_f32, higher_count));

            // --- Binding fields ---
            face_indices.push(face_idx as u32);
            barycentric.push([bary_u, bary_v, bary_w]);
            local_offsets.push([0.0_f32, 0.0_f32, 0.0_f32]);
            is_rigid.push(false);
        }

        Ok(GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree: config.sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        })
    }

    /// Sample a face index using area-weighted probability via binary search on the prefix sum.
    fn sample_face(&self, rng: &mut Xorshift64) -> usize {
        let r = rng.next_f32() as f64 * self.total_area;
        // Binary search for the first prefix entry > r.
        let mut lo = 0usize;
        let mut hi = self.face_areas.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.area_prefix[mid + 1] <= r {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Clamp to valid range (handles floating-point edge at total_area boundary).
        lo.min(self.face_areas.len() - 1)
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Compute the area of a triangle given its three vertex positions.
fn triangle_area(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> f32 {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let cross = cross3(e1, e2);
    0.5 * len3(cross)
}

/// Retrieve a vertex by index, returning an error on out-of-bounds access.
fn get_vertex(vertices: &[[f32; 3]], idx: u32, face_idx: usize) -> Result<[f32; 3], RenderError> {
    vertices.get(idx as usize).copied().ok_or_else(|| {
        RenderError::Rasterize(format!(
            "Face {face_idx}: vertex index {idx} is out of bounds (mesh has {} vertices)",
            vertices.len()
        ))
    })
}

/// Sample barycentric coordinates `(u, v, w)` uniformly on a triangle.
///
/// Uses the method: `u = rand, v = rand * (1 - u), w = 1 - u - v`.
/// A mirror fold is applied if `u + v > 1` (safety guard).
fn sample_barycentric(rng: &mut Xorshift64) -> (f32, f32, f32) {
    let r1 = rng.next_f32();
    let r2 = rng.next_f32();

    let (u, v) = if r1 + r2 > 1.0 {
        (1.0 - r1, 1.0 - r2)
    } else {
        (r1, r2)
    };
    let w = 1.0 - u - v;
    // Clamp to ensure non-negative w due to floating-point noise.
    let w = w.max(0.0);
    (u, v, w)
}

/// Face geometry used for rotation computation, bundled to avoid a wide function signature.
struct FaceGeom<'a> {
    i0: u32,
    i1: u32,
    i2: u32,
    v0: &'a [f32; 3],
    v1: &'a [f32; 3],
    v2: &'a [f32; 3],
    bary: [f32; 3],
    normals: Option<&'a [[f32; 3]]>,
}

/// Compute the rotation quaternion `[x, y, z, w]` for a Gaussian placed on a face.
///
/// Builds a TBN frame from the triangle edges and surface normal, then converts
/// the rotation matrix to a unit quaternion using the Shepperd method.
fn compute_rotation(geom: FaceGeom<'_>) -> [f32; 4] {
    let FaceGeom {
        i0,
        i1,
        i2,
        v0,
        v1,
        v2,
        bary: [bary_u, bary_v, bary_w],
        normals,
    } = geom;

    // Compute or interpolate the surface normal.
    let normal = if let Some(nrm) = normals {
        let n0 = nrm.get(i0 as usize).copied().unwrap_or([0.0, 0.0, 1.0]);
        let n1 = nrm.get(i1 as usize).copied().unwrap_or([0.0, 0.0, 1.0]);
        let n2 = nrm.get(i2 as usize).copied().unwrap_or([0.0, 0.0, 1.0]);
        let interp = [
            bary_u * n0[0] + bary_v * n1[0] + bary_w * n2[0],
            bary_u * n0[1] + bary_v * n1[1] + bary_w * n2[1],
            bary_u * n0[2] + bary_v * n1[2] + bary_w * n2[2],
        ];
        normalize3_safe(interp, [0.0, 0.0, 1.0])
    } else {
        let e1 = sub3(*v1, *v0);
        let e2 = sub3(*v2, *v0);
        let cross = cross3(e1, e2);
        normalize3_safe(cross, [0.0, 0.0, 1.0])
    };

    // Build TBN: tangent = normalize(v1 - v0), bitangent = cross(normal, tangent).
    let edge = sub3(*v1, *v0);
    let tangent = normalize3_safe(edge, [1.0, 0.0, 0.0]);
    let bitangent = normalize3(cross3(normal, tangent));

    // Rotation matrix columns: [tangent, bitangent, normal].
    // R = [[t.x, b.x, n.x],
    //      [t.y, b.y, n.y],
    //      [t.z, b.z, n.z]]
    let m = [
        [tangent[0], bitangent[0], normal[0]],
        [tangent[1], bitangent[1], normal[1]],
        [tangent[2], bitangent[2], normal[2]],
    ];

    rotation_matrix_to_quaternion(m)
}

/// Convert a 3×3 rotation matrix to a quaternion `[x, y, z, w]` using the Shepperd method.
///
/// The matrix is in column-major order: `m[row][col]`.
fn rotation_matrix_to_quaternion(m: [[f32; 3]; 3]) -> [f32; 4] {
    // Trace = m[0][0] + m[1][1] + m[2][2]
    let trace = m[0][0] + m[1][1] + m[2][2];

    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (m[2][1] - m[1][2]) * s;
        let y = (m[0][2] - m[2][0]) * s;
        let z = (m[1][0] - m[0][1]) * s;
        [x, y, z, w]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        [x, y, z, w]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        [x, y, z, w]
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        [x, y, z, w]
    }
}

// ---------------------------------------------------------------------------
// Vector math helpers (pure f32, no external dependencies)
// ---------------------------------------------------------------------------

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn len3(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Normalize a vector; returns `fallback` if the vector is (near-)zero.
#[inline]
fn normalize3_safe(a: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let l = len3(a);
    if l < 1e-10 {
        fallback
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

/// Normalize a vector; panics if zero (only call when length is guaranteed non-zero).
#[inline]
fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let l = len3(a);
    if l < 1e-10 {
        [0.0, 1.0, 0.0] // safe fallback
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple unit triangle in the XY plane.
    fn unit_triangle_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0u32, 1, 2]];
        (vertices, faces)
    }

    /// A simple mesh with two triangles of different areas.
    fn two_triangle_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        // Triangle 0: area = 0.5 (unit right triangle)
        // Triangle 1: area = 2.0 (base=2, height=2)
        let vertices = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0], // 2
            [2.0, 0.0, 0.0], // 3
            [0.0, 2.0, 0.0], // 4
        ];
        let faces = vec![[0u32, 1, 2], [0, 3, 4]];
        (vertices, faces)
    }

    // -----------------------------------------------------------------------
    // test_config_defaults
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = GaussianInitConfig::default();
        assert_eq!(cfg.num_gaussians, 10_000);
        assert!((cfg.scale_factor - 0.05).abs() < 1e-6);
        assert!((cfg.min_scale_log - (-6.0)).abs() < 1e-6);
        assert!((cfg.max_scale_log - 0.0).abs() < 1e-6);
        assert!((cfg.initial_opacity_logit - (0.1_f32 / 0.9_f32).ln()).abs() < 1e-4);
        assert_eq!(cfg.sh_degree, 0);
        assert_eq!(cfg.mean_color, [0.5, 0.5, 0.5]);
        assert_eq!(cfg.seed, 42);
    }

    // -----------------------------------------------------------------------
    // test_empty_mesh_error
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_mesh_error() {
        let vertices = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces: Vec<[u32; 3]> = vec![];
        let result = GaussianInitializer::new(&vertices, &faces, None);
        assert!(result.is_err(), "Empty mesh must produce an error");
        let err = result.err().unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("empty") || msg.contains("no faces"),
            "Error should mention empty mesh, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // test_single_triangle_init
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_triangle_init() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 10,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");
        assert_eq!(model.gaussians.len(), 10);
    }

    // -----------------------------------------------------------------------
    // test_gaussian_count_matches_config
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_count_matches_config() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        for &n in &[1usize, 50, 100, 500] {
            let cfg = GaussianInitConfig {
                num_gaussians: n,
                ..Default::default()
            };
            let model = init.initialize(&cfg).expect("Should initialize model");
            assert_eq!(
                model.gaussians.len(),
                n,
                "Expected {n} Gaussians, got {}",
                model.gaussians.len()
            );
            assert_eq!(model.face_indices.len(), n);
            assert_eq!(model.barycentric.len(), n);
            assert_eq!(model.local_offsets.len(), n);
            assert_eq!(model.is_rigid.len(), n);
        }
    }

    // -----------------------------------------------------------------------
    // test_barycentric_valid_sum_to_one
    // -----------------------------------------------------------------------
    #[test]
    fn test_barycentric_valid_sum_to_one() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 200,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");
        for (i, &[u, v, w]) in model.barycentric.iter().enumerate() {
            assert!((0.0..=1.0).contains(&u), "Gaussian {i}: u={u} out of [0,1]");
            assert!((0.0..=1.0).contains(&v), "Gaussian {i}: v={v} out of [0,1]");
            assert!((0.0..=1.0).contains(&w), "Gaussian {i}: w={w} out of [0,1]");
            let sum = u + v + w;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Gaussian {i}: barycentric sum = {sum}, expected 1.0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_scale_within_bounds
    // -----------------------------------------------------------------------
    #[test]
    fn test_scale_within_bounds() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 100,
            min_scale_log: -4.0,
            max_scale_log: -0.5,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");
        for (i, g) in model.gaussians.iter().enumerate() {
            for (dim, &s) in g.scale.iter().enumerate() {
                assert!(
                    s >= cfg.min_scale_log - 1e-6 && s <= cfg.max_scale_log + 1e-6,
                    "Gaussian {i} dim {dim}: log-scale {s} outside [{}, {}]",
                    cfg.min_scale_log,
                    cfg.max_scale_log
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_opacity_initial_value
    // -----------------------------------------------------------------------
    #[test]
    fn test_opacity_initial_value() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 50,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        let expected_logit = (0.1_f32 / 0.9_f32).ln();
        // sigmoid(logit) should be ≈ 0.1
        let expected_opacity = 1.0 / (1.0 + (-expected_logit).exp());

        for (i, g) in model.gaussians.iter().enumerate() {
            assert!(
                (g.opacity - expected_logit).abs() < 1e-5,
                "Gaussian {i}: opacity logit = {}, expected {}",
                g.opacity,
                expected_logit
            );
            let decoded = 1.0 / (1.0 + (-g.opacity).exp());
            assert!(
                (decoded - expected_opacity).abs() < 1e-4,
                "Gaussian {i}: sigmoid(opacity) = {decoded}, expected {expected_opacity}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_sh_dc_from_mean_color
    // -----------------------------------------------------------------------
    #[test]
    fn test_sh_dc_from_mean_color() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let mean_color = [0.8_f32, 0.3, 0.5];
        let cfg = GaussianInitConfig {
            num_gaussians: 10,
            mean_color,
            sh_degree: 0,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        // sh_coeffs layout for degree 0: [r_dc, g_dc, b_dc] * N
        assert_eq!(model.sh_coeffs.len(), 10 * 3);

        for i in 0..10 {
            let sh_r = model.sh_coeffs[i * 3];
            let sh_g = model.sh_coeffs[i * 3 + 1];
            let sh_b = model.sh_coeffs[i * 3 + 2];

            // Decoded: color = 0.5 + SH_C0 * sh_dc
            let decoded_r = 0.5 + SH_C0 * sh_r;
            let decoded_g = 0.5 + SH_C0 * sh_g;
            let decoded_b = 0.5 + SH_C0 * sh_b;

            assert!(
                (decoded_r - mean_color[0]).abs() < 1e-5,
                "Gaussian {i}: decoded R = {decoded_r}, expected {}",
                mean_color[0]
            );
            assert!(
                (decoded_g - mean_color[1]).abs() < 1e-5,
                "Gaussian {i}: decoded G = {decoded_g}, expected {}",
                mean_color[1]
            );
            assert!(
                (decoded_b - mean_color[2]).abs() < 1e-5,
                "Gaussian {i}: decoded B = {decoded_b}, expected {}",
                mean_color[2]
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_rotation_is_unit_quaternion
    // -----------------------------------------------------------------------
    #[test]
    fn test_rotation_is_unit_quaternion() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 100,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");
        for (i, g) in model.gaussians.iter().enumerate() {
            let [x, y, z, w] = g.rotation;
            let norm = (x * x + y * y + z * z + w * w).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Gaussian {i}: quaternion norm = {norm}, expected 1.0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_all_positions_on_mesh_surface
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_positions_on_mesh_surface() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 100,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        for (i, (g, (&face_idx, &[bu, bv, bw]))) in model
            .gaussians
            .iter()
            .zip(model.face_indices.iter().zip(model.barycentric.iter()))
            .enumerate()
        {
            let [fi0, fi1, fi2] = faces[face_idx as usize];
            let v0 = vertices[fi0 as usize];
            let v1 = vertices[fi1 as usize];
            let v2 = vertices[fi2 as usize];

            let expected_pos = [
                bu * v0[0] + bv * v1[0] + bw * v2[0],
                bu * v0[1] + bv * v1[1] + bw * v2[1],
                bu * v0[2] + bv * v1[2] + bw * v2[2],
            ];

            for (dim, (&pos_val, &exp_val)) in
                g.position.iter().zip(expected_pos.iter()).enumerate()
            {
                assert!(
                    (pos_val - exp_val).abs() < 1e-5,
                    "Gaussian {i} dim {dim}: position {} != expected {}",
                    pos_val,
                    exp_val
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_deterministic_with_same_seed
    // -----------------------------------------------------------------------
    #[test]
    fn test_deterministic_with_same_seed() {
        let (vertices, faces) = two_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let cfg = GaussianInitConfig {
            num_gaussians: 50,
            seed: 12345,
            ..Default::default()
        };

        let model_a = init.initialize(&cfg).expect("First init");
        let model_b = init.initialize(&cfg).expect("Second init");

        for (i, (ga, gb)) in model_a
            .gaussians
            .iter()
            .zip(model_b.gaussians.iter())
            .enumerate()
        {
            for dim in 0..3 {
                assert_eq!(
                    ga.position[dim], gb.position[dim],
                    "Gaussian {i} position differs between identical seeds"
                );
            }
        }
        assert_eq!(model_a.face_indices, model_b.face_indices);
    }

    // -----------------------------------------------------------------------
    // test_different_seeds_give_different_results
    // -----------------------------------------------------------------------
    #[test]
    fn test_different_seeds_give_different_results() {
        let (vertices, faces) = two_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let cfg_a = GaussianInitConfig {
            num_gaussians: 100,
            seed: 1,
            ..Default::default()
        };
        let cfg_b = GaussianInitConfig {
            num_gaussians: 100,
            seed: 2,
            ..Default::default()
        };

        let model_a = init.initialize(&cfg_a).expect("First init");
        let model_b = init.initialize(&cfg_b).expect("Second init");

        let any_different = model_a
            .gaussians
            .iter()
            .zip(model_b.gaussians.iter())
            .any(|(ga, gb)| ga.position[0] != gb.position[0]);
        assert!(
            any_different,
            "Different seeds should produce different Gaussian positions"
        );
    }

    // -----------------------------------------------------------------------
    // test_area_weighted_bias
    // -----------------------------------------------------------------------
    #[test]
    fn test_area_weighted_bias() {
        // Two triangles: T0 area=0.5, T1 area=2.0 → ratio 1:4
        // With many samples, T1 should receive ~4x more Gaussians.
        let (vertices, faces) = two_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let cfg = GaussianInitConfig {
            num_gaussians: 5000,
            seed: 99,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        let count_t0 = model.face_indices.iter().filter(|&&f| f == 0).count();
        let count_t1 = model.face_indices.iter().filter(|&&f| f == 1).count();

        // Expected ratio: T1/T0 ≈ 4.0. Allow ±30% tolerance.
        let ratio = count_t1 as f64 / count_t0.max(1) as f64;
        assert!(
            ratio > 2.8 && ratio < 5.2,
            "Expected T1/T0 ratio ≈ 4.0, got {ratio:.2} (T0={count_t0}, T1={count_t1})"
        );
    }

    // -----------------------------------------------------------------------
    // test_face_indices_in_range
    // -----------------------------------------------------------------------
    #[test]
    fn test_face_indices_in_range() {
        let (vertices, faces) = two_triangle_mesh();
        let num_faces = faces.len();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let cfg = GaussianInitConfig {
            num_gaussians: 200,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        for (i, &fi) in model.face_indices.iter().enumerate() {
            assert!(
                (fi as usize) < num_faces,
                "Gaussian {i}: face_index {fi} >= num_faces {num_faces}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_local_offsets_are_zero
    // -----------------------------------------------------------------------
    #[test]
    fn test_local_offsets_are_zero() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 50,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        for (i, &offset) in model.local_offsets.iter().enumerate() {
            assert_eq!(
                offset,
                [0.0_f32, 0.0, 0.0],
                "Gaussian {i}: local_offset is not zero, got {offset:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_is_rigid_all_false
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_rigid_all_false() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");
        let cfg = GaussianInitConfig {
            num_gaussians: 50,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        for (i, &rigid) in model.is_rigid.iter().enumerate() {
            assert!(!rigid, "Gaussian {i}: is_rigid should be false");
        }
    }

    // -----------------------------------------------------------------------
    // test_large_mesh_init
    // -----------------------------------------------------------------------
    #[test]
    fn test_large_mesh_init() {
        // Build a grid mesh with 1000 triangles.
        let grid_size = 23usize; // 23×23 grid → 22×22 quads → 968 triangles ≈ 1000
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for row in 0..=grid_size {
            for col in 0..=grid_size {
                vertices.push([col as f32, row as f32, 0.0_f32]);
            }
        }
        let stride = grid_size + 1;
        for row in 0..grid_size {
            for col in 0..grid_size {
                let tl = (row * stride + col) as u32;
                let tr = tl + 1;
                let bl = tl + stride as u32;
                let br = bl + 1;
                faces.push([tl, tr, bl]);
                faces.push([tr, br, bl]);
            }
        }

        assert!(
            faces.len() >= 900,
            "Expected ~1000 triangles, got {}",
            faces.len()
        );

        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid large mesh");

        let cfg = GaussianInitConfig {
            num_gaussians: 5000,
            ..Default::default()
        };
        let model = init
            .initialize(&cfg)
            .expect("Should initialize model from large mesh");
        assert_eq!(model.gaussians.len(), 5000);

        // All face indices must be in range.
        for &fi in &model.face_indices {
            assert!((fi as usize) < faces.len());
        }
    }

    // -----------------------------------------------------------------------
    // test_sh_higher_degree_zero
    // -----------------------------------------------------------------------
    #[test]
    fn test_sh_higher_degree_zero() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let cfg = GaussianInitConfig {
            num_gaussians: 5,
            sh_degree: 1,
            mean_color: [0.5, 0.5, 0.5],
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        // sh_degree=1 → (1+1)^2 * 3 = 12 coeffs per Gaussian.
        assert_eq!(model.sh_coeffs.len(), 5 * 12);

        for i in 0..5 {
            // First 3 are DC (may be nonzero), rest (indices 3..12) must be 0.
            for k in 3..12 {
                let val = model.sh_coeffs[i * 12 + k];
                assert_eq!(val, 0.0, "Gaussian {i} SH coeff {k} should be 0, got {val}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // test_rotation_with_normals
    // -----------------------------------------------------------------------
    #[test]
    fn test_rotation_with_normals() {
        let (vertices, faces) = unit_triangle_mesh();
        let normals = vec![
            [0.0_f32, 0.0, 1.0],
            [0.0_f32, 0.0, 1.0],
            [0.0_f32, 0.0, 1.0],
        ];
        let init = GaussianInitializer::new(&vertices, &faces, Some(&normals))
            .expect("Should construct from valid mesh with normals");
        let cfg = GaussianInitConfig {
            num_gaussians: 20,
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Should initialize model");

        // All rotations must be unit quaternions.
        for (i, g) in model.gaussians.iter().enumerate() {
            let [x, y, z, w] = g.rotation;
            let norm = (x * x + y * y + z * z + w * w).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "Gaussian {i} with normals: quaternion norm = {norm}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_zero_seed_handled
    // -----------------------------------------------------------------------
    #[test]
    fn test_zero_seed_handled() {
        let (vertices, faces) = unit_triangle_mesh();
        let init = GaussianInitializer::new(&vertices, &faces, None)
            .expect("Should construct from valid mesh");

        let cfg = GaussianInitConfig {
            num_gaussians: 10,
            seed: 0, // zero seed should be handled gracefully
            ..Default::default()
        };
        let model = init.initialize(&cfg).expect("Zero seed should not panic");
        assert_eq!(model.gaussians.len(), 10);
    }
}
