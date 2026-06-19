//! Lighting models for FLAME head mesh rendering.
//!
//! This module provides CPU-side illumination for FLAME meshes:
//! - Lambertian diffuse shading
//! - Phong specular highlights
//! - Ambient occlusion approximation
//! - Spherical harmonics (SH) environment lighting
//!
//! All math is implemented manually using `[f32; 3]` arrays without nalgebra.
//! Conversion from `Mesh` nalgebra types occurs at the boundary.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in lighting computations.
#[derive(Debug, Error)]
pub enum LightingError {
    /// The mesh has no vertices.
    #[error("Mesh has no vertices")]
    EmptyMesh,
    /// Normal count does not match vertex count.
    #[error("Normal count {normals} does not match vertex count {vertices}")]
    NormalMismatch {
        /// Number of normals in the mesh.
        normals: usize,
        /// Number of vertices in the mesh.
        vertices: usize,
    },
    /// SH degree out of range.
    #[error("Invalid SH degree {degree}: must be 0, 1, 2, or 3")]
    InvalidShDegree {
        /// The invalid degree value.
        degree: usize,
    },
    /// SH coefficient slice length mismatch.
    #[error(
        "SH coefficient count {actual} does not match expected {expected} for degree {degree}"
    )]
    ShCoefficientMismatch {
        /// Actual number of coefficients provided.
        actual: usize,
        /// Expected number of coefficients.
        expected: usize,
        /// Degree that was requested.
        degree: usize,
    },
    /// Light color values are out of `[0, 1]` range.
    #[error("Invalid light color: values must be in [0, 1]")]
    InvalidLightColor,
}

// ---------------------------------------------------------------------------
// Math helpers (private)
// ---------------------------------------------------------------------------

/// Compute dot product of two 3-vectors.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalize a 3-vector. Returns `[0, 0, 0]` if the norm is less than `1e-8`.
#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-8 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

/// Add two 3-vectors component-wise.
#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Subtract `b` from `a` component-wise.
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scale a 3-vector by a scalar.
#[inline]
fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Compute the reflection of `incident` about `normal`.
///
/// `reflect = incident - 2 * dot(incident, normal) * normal`
#[inline]
fn reflect3(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let d = dot3(incident, normal);
    sub3(incident, scale3(normal, 2.0 * d))
}

/// Linear interpolation between two 3-vectors.
///
/// `t = 0` → `a`, `t = 1` → `b`.
#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Clamp each component of a 3-vector to `[lo, hi]`.
#[inline]
fn clamp3(v: [f32; 3], lo: f32, hi: f32) -> [f32; 3] {
    [v[0].clamp(lo, hi), v[1].clamp(lo, hi), v[2].clamp(lo, hi)]
}

/// Component-wise multiplication of two 3-vectors.
#[inline]
fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

// ---------------------------------------------------------------------------
// Light source types
// ---------------------------------------------------------------------------

/// A directional light source (infinitely far away, uniform direction).
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    /// Direction the light travels from (unit vector pointing toward the light source).
    pub direction: [f32; 3],
    /// Light color in RGB `[0, 1]` per channel.
    pub color: [f32; 3],
    /// Intensity multiplier (default `1.0`).
    pub intensity: f32,
}

impl DirectionalLight {
    /// Create a new directional light with the given direction, color, and intensity.
    #[must_use]
    pub fn new(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            direction: normalize3(direction),
            color,
            intensity,
        }
    }

    /// Validate that color components are in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`LightingError::InvalidLightColor`] if any channel is outside `[0, 1]`.
    pub fn validate(&self) -> Result<(), LightingError> {
        for &c in &self.color {
            if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                return Err(LightingError::InvalidLightColor);
            }
        }
        Ok(())
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: normalize3([0.0, 1.0, 0.0]),
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

/// A positional (point) light source with distance attenuation.
#[derive(Debug, Clone)]
pub struct PointLight {
    /// World-space position of the light.
    pub position: [f32; 3],
    /// Light color in RGB `[0, 1]` per channel.
    pub color: [f32; 3],
    /// Intensity multiplier (default `1.0`).
    pub intensity: f32,
    /// Quadratic attenuation factor: effective intensity = `I / (1 + atten * dist²)`.
    pub attenuation: f32,
}

impl PointLight {
    /// Create a new point light.
    #[must_use]
    pub fn new(position: [f32; 3], color: [f32; 3], intensity: f32, attenuation: f32) -> Self {
        Self {
            position,
            color,
            intensity,
            attenuation,
        }
    }

    /// Validate that color components are in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`LightingError::InvalidLightColor`] if any channel is outside `[0, 1]`.
    pub fn validate(&self) -> Result<(), LightingError> {
        for &c in &self.color {
            if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                return Err(LightingError::InvalidLightColor);
            }
        }
        Ok(())
    }
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0, 2.0, 2.0],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            attenuation: 1.0,
        }
    }
}

/// A uniform ambient light source illuminating all surfaces equally.
#[derive(Debug, Clone)]
pub struct AmbientLight {
    /// Light color in RGB `[0, 1]` per channel.
    pub color: [f32; 3],
    /// Intensity multiplier (default `0.1`).
    pub intensity: f32,
}

impl AmbientLight {
    /// Create a new ambient light.
    #[must_use]
    pub fn new(color: [f32; 3], intensity: f32) -> Self {
        Self { color, intensity }
    }
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Phong reflection model material parameters.
#[derive(Debug, Clone)]
pub struct PhongMaterial {
    /// Diffuse color coefficient `Kd` (default skin-like `[0.8, 0.7, 0.6]`).
    pub diffuse_color: [f32; 3],
    /// Specular color coefficient `Ks` (default `[0.3, 0.3, 0.3]`).
    pub specular_color: [f32; 3],
    /// Specular shininess exponent `n` (default `32.0`).
    pub shininess: f32,
    /// Ambient color coefficient `Ka` (default `[0.2, 0.15, 0.1]`).
    pub ambient_color: [f32; 3],
}

impl Default for PhongMaterial {
    fn default() -> Self {
        Self {
            diffuse_color: [0.8, 0.7, 0.6],
            specular_color: [0.3, 0.3, 0.3],
            shininess: 32.0,
            ambient_color: [0.2, 0.15, 0.1],
        }
    }
}

impl PhongMaterial {
    /// Create the default skin-like material.
    #[must_use]
    pub fn skin() -> Self {
        Self::default()
    }

    /// Create a matte/diffuse-only material (low shininess, minimal specular).
    #[must_use]
    pub fn matte() -> Self {
        Self {
            diffuse_color: [0.75, 0.65, 0.55],
            specular_color: [0.05, 0.05, 0.05],
            shininess: 4.0,
            ambient_color: [0.15, 0.12, 0.09],
        }
    }

    /// Create a glossy/high-sheen skin material.
    #[must_use]
    pub fn glossy() -> Self {
        Self {
            diffuse_color: [0.7, 0.6, 0.5],
            specular_color: [0.6, 0.6, 0.6],
            shininess: 128.0,
            ambient_color: [0.1, 0.08, 0.06],
        }
    }
}

// ---------------------------------------------------------------------------
// Lighting result
// ---------------------------------------------------------------------------

/// Per-vertex lighting result for an entire mesh.
pub struct LightingResult {
    /// Per-vertex RGB colors in `[0, 1]`.
    pub vertex_colors: Vec<[f32; 3]>,
    /// Total number of vertices.
    pub n_vertices: usize,
}

impl LightingResult {
    /// Create a new result with all vertices set to black.
    #[must_use]
    pub fn zeros(n_vertices: usize) -> Self {
        Self {
            vertex_colors: vec![[0.0, 0.0, 0.0]; n_vertices],
            n_vertices,
        }
    }

    /// Convert per-vertex colors to RGBA u8 (alpha = 255).
    ///
    /// Output length is `4 * n_vertices`.
    #[must_use]
    pub fn to_rgba_u8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.n_vertices * 4);
        for &[r, g, b] in &self.vertex_colors {
            out.push((r.clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((g.clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((b.clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push(255u8);
        }
        out
    }

    /// Compute the mean luminance (perceptual brightness) across all vertices.
    ///
    /// Uses the standard Rec. 709 luminance coefficients: `0.2126 R + 0.7152 G + 0.0722 B`.
    #[must_use]
    pub fn mean_luminance(&self) -> f32 {
        if self.vertex_colors.is_empty() {
            return 0.0;
        }
        let total: f32 = self
            .vertex_colors
            .iter()
            .map(|&[r, g, b]| 0.2126 * r + 0.7152 * g + 0.0722 * b)
            .sum();
        total / self.vertex_colors.len() as f32
    }

    /// Flatten vertex colors to a `[r0, g0, b0, r1, g1, b1, ...]` slice.
    ///
    /// Output length is `3 * n_vertices`.
    #[must_use]
    pub fn to_flat_rgb(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_vertices * 3);
        for &[r, g, b] in &self.vertex_colors {
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Lighting functions
// ---------------------------------------------------------------------------

/// Compute Lambertian diffuse color for a surface point lit by a directional light.
///
/// Returns `max(0, N · L) * light.color * light.intensity`.
#[must_use]
pub fn lambertian_diffuse(normal: [f32; 3], light: &DirectionalLight) -> [f32; 3] {
    let l = normalize3(light.direction);
    let n_dot_l = dot3(normal, l).max(0.0);
    scale3(mul3(light.color, [n_dot_l; 3]), light.intensity)
}

/// Compute Phong shading for a single vertex with one directional light and ambient.
///
/// Combines ambient, diffuse, and specular components clamped to `[0, 1]`.
///
/// - `normal`: unit surface normal
/// - `view_dir`: unit vector from the vertex toward the camera
/// - `light`: directional light source
/// - `ambient`: ambient light source
/// - `material`: Phong material properties
#[must_use]
pub fn phong_vertex(
    normal: [f32; 3],
    view_dir: [f32; 3],
    light: &DirectionalLight,
    ambient: &AmbientLight,
    material: &PhongMaterial,
) -> [f32; 3] {
    // Ambient component: Ka * ambient.color * ambient.intensity
    let ambient_term = scale3(
        mul3(material.ambient_color, ambient.color),
        ambient.intensity,
    );

    // Diffuse component: Kd * light.color * max(0, N·L) * light.intensity
    let l = normalize3(light.direction);
    let n_dot_l = dot3(normal, l).max(0.0);
    let diffuse_term = scale3(
        mul3(material.diffuse_color, light.color),
        n_dot_l * light.intensity,
    );

    // Specular component: Ks * light.color * max(0, R·V)^n * light.intensity
    // R = reflect(-L, N)
    let neg_l = scale3(l, -1.0);
    let r = reflect3(neg_l, normal);
    let r_dot_v = dot3(r, view_dir).max(0.0);
    let specular_factor = r_dot_v.powf(material.shininess);
    let specular_term = scale3(
        mul3(material.specular_color, light.color),
        specular_factor * light.intensity,
    );

    clamp3(
        add3(add3(ambient_term, diffuse_term), specular_term),
        0.0,
        1.0,
    )
}

/// Compute Phong shading for a single vertex lit by a point light with distance attenuation.
///
/// Attenuation factor: `1 / (1 + atten * dist²)`.
#[must_use]
pub fn phong_vertex_point(
    vertex_pos: [f32; 3],
    normal: [f32; 3],
    view_dir: [f32; 3],
    light: &PointLight,
    ambient: &AmbientLight,
    material: &PhongMaterial,
) -> [f32; 3] {
    // Vector from vertex to light
    let to_light = sub3(light.position, vertex_pos);
    let dist_sq = dot3(to_light, to_light);
    let dist = dist_sq.sqrt();

    // Attenuation: 1 / (1 + atten * dist²)
    let atten = 1.0 / (1.0 + light.attenuation * dist_sq);

    // Effective intensity
    let effective_intensity = light.intensity * atten;

    // Light direction (normalized)
    let l = if dist < 1e-8 {
        [0.0, 1.0, 0.0]
    } else {
        normalize3(to_light)
    };

    // Ambient component
    let ambient_term = scale3(
        mul3(material.ambient_color, ambient.color),
        ambient.intensity,
    );

    // Diffuse component
    let n_dot_l = dot3(normal, l).max(0.0);
    let diffuse_term = scale3(
        mul3(material.diffuse_color, light.color),
        n_dot_l * effective_intensity,
    );

    // Specular component
    let neg_l = scale3(l, -1.0);
    let r = reflect3(neg_l, normal);
    let r_dot_v = dot3(r, view_dir).max(0.0);
    let specular_factor = r_dot_v.powf(material.shininess);
    let specular_term = scale3(
        mul3(material.specular_color, light.color),
        specular_factor * effective_intensity,
    );

    clamp3(
        add3(add3(ambient_term, diffuse_term), specular_term),
        0.0,
        1.0,
    )
}

/// Validate that a mesh has vertices and matching normals.
///
/// # Errors
///
/// Returns [`LightingError::EmptyMesh`] or [`LightingError::NormalMismatch`].
fn validate_mesh(mesh: &crate::Mesh) -> Result<(), LightingError> {
    if mesh.vertices.is_empty() {
        return Err(LightingError::EmptyMesh);
    }
    if mesh.normals.len() != mesh.vertices.len() {
        return Err(LightingError::NormalMismatch {
            normals: mesh.normals.len(),
            vertices: mesh.vertices.len(),
        });
    }
    Ok(())
}

/// Extract vertex position as `[f32; 3]` from a `Mesh`.
#[inline]
fn vertex_to_arr(mesh: &crate::Mesh, i: usize) -> [f32; 3] {
    let v = &mesh.vertices[i];
    [v.x, v.y, v.z]
}

/// Extract vertex normal as `[f32; 3]` from a `Mesh`.
#[inline]
fn normal_to_arr(mesh: &crate::Mesh, i: usize) -> [f32; 3] {
    let n = &mesh.normals[i];
    [n.x, n.y, n.z]
}

/// Compute per-vertex Phong shading for an entire mesh with a single directional light.
///
/// # Errors
///
/// Returns an error if the mesh is empty or has mismatched normals.
pub fn shade_mesh_directional(
    mesh: &crate::Mesh,
    light: &DirectionalLight,
    ambient: &AmbientLight,
    material: &PhongMaterial,
    camera_pos: [f32; 3],
) -> Result<LightingResult, LightingError> {
    validate_mesh(mesh)?;
    let n_verts = mesh.vertices.len();
    let mut colors = Vec::with_capacity(n_verts);

    for i in 0..n_verts {
        let vertex = vertex_to_arr(mesh, i);
        let normal = normalize3(normal_to_arr(mesh, i));
        let view_dir = normalize3(sub3(camera_pos, vertex));
        let color = phong_vertex(normal, view_dir, light, ambient, material);
        colors.push(color);
    }

    Ok(LightingResult {
        vertex_colors: colors,
        n_vertices: n_verts,
    })
}

/// Compute per-vertex Phong shading with multiple directional and point lights.
///
/// Contributions from all lights are summed and clamped to `[0, 1]`.
///
/// # Errors
///
/// Returns an error if the mesh is empty or has mismatched normals.
pub fn shade_mesh_multi_light(
    mesh: &crate::Mesh,
    directional_lights: &[DirectionalLight],
    point_lights: &[PointLight],
    ambient: &AmbientLight,
    material: &PhongMaterial,
    camera_pos: [f32; 3],
) -> Result<LightingResult, LightingError> {
    validate_mesh(mesh)?;
    let n_verts = mesh.vertices.len();
    let mut colors = Vec::with_capacity(n_verts);

    // Ambient term (applied once regardless of light count)
    let ambient_term = scale3(
        mul3(material.ambient_color, ambient.color),
        ambient.intensity,
    );

    for i in 0..n_verts {
        let vertex = vertex_to_arr(mesh, i);
        let normal = normalize3(normal_to_arr(mesh, i));
        let view_dir = normalize3(sub3(camera_pos, vertex));

        let mut color = ambient_term;

        // Accumulate directional lights
        for dlight in directional_lights {
            let l = normalize3(dlight.direction);
            let n_dot_l = dot3(normal, l).max(0.0);
            let diffuse = scale3(
                mul3(material.diffuse_color, dlight.color),
                n_dot_l * dlight.intensity,
            );
            let neg_l = scale3(l, -1.0);
            let r = reflect3(neg_l, normal);
            let r_dot_v = dot3(r, view_dir).max(0.0);
            let spec_factor = r_dot_v.powf(material.shininess);
            let specular = scale3(
                mul3(material.specular_color, dlight.color),
                spec_factor * dlight.intensity,
            );
            color = add3(color, add3(diffuse, specular));
        }

        // Accumulate point lights
        for plight in point_lights {
            let to_light = sub3(plight.position, vertex);
            let dist_sq = dot3(to_light, to_light);
            let atten = 1.0 / (1.0 + plight.attenuation * dist_sq);
            let effective = plight.intensity * atten;

            let l = normalize3(to_light);
            let n_dot_l = dot3(normal, l).max(0.0);
            let diffuse = scale3(
                mul3(material.diffuse_color, plight.color),
                n_dot_l * effective,
            );
            let neg_l = scale3(l, -1.0);
            let r = reflect3(neg_l, normal);
            let r_dot_v = dot3(r, view_dir).max(0.0);
            let spec_factor = r_dot_v.powf(material.shininess);
            let specular = scale3(
                mul3(material.specular_color, plight.color),
                spec_factor * effective,
            );
            color = add3(color, add3(diffuse, specular));
        }

        colors.push(clamp3(color, 0.0, 1.0));
    }

    Ok(LightingResult {
        vertex_colors: colors,
        n_vertices: n_verts,
    })
}

/// Evaluate the 9 real spherical harmonic basis functions (degree ≤ 2) for a direction.
///
/// Basis ordering (Ambiant Dice / standard):
/// - `Y_00`, Y_1-1, `Y_10`, `Y_11`, Y_2-2, Y_2-1, `Y_20`, `Y_21`, `Y_22`
#[inline]
fn sh_basis_9(nx: f32, ny: f32, nz: f32) -> [f32; 9] {
    [
        0.282_095,                         // Y_00 (l=0, m=0)
        0.488_603 * ny,                    // Y_1-1 (l=1, m=-1)
        0.488_603 * nz,                    // Y_10  (l=1, m=0)
        0.488_603 * nx,                    // Y_11  (l=1, m=1)
        1.092_548 * nx * ny,               // Y_2-2 (l=2, m=-2)
        1.092_548 * ny * nz,               // Y_2-1 (l=2, m=-1)
        0.315_392 * (3.0 * nz * nz - 1.0), // Y_20  (l=2, m=0)
        1.092_548 * nx * nz,               // Y_21  (l=2, m=1)
        0.546_274 * (nx * nx - ny * ny),   // Y_22  (l=2, m=2)
    ]
}

/// Compute per-vertex environment lighting from spherical harmonics coefficients.
///
/// `sh_coeffs` must contain 27 values: 9 per RGB channel, ordered `[R0..R8, G0..G8, B0..B8]`.
///
/// Uses degree-2 SH (9 coefficients per channel).
///
/// # Errors
///
/// Returns [`LightingError::ShCoefficientMismatch`] if `sh_coeffs.len() != 27`,
/// or [`LightingError::EmptyMesh`] / [`LightingError::NormalMismatch`] for invalid meshes.
pub fn shade_mesh_sh_lighting(
    mesh: &crate::Mesh,
    sh_coeffs: &[f32],
) -> Result<LightingResult, LightingError> {
    const EXPECTED: usize = 27; // 9 coeffs × 3 channels
    if sh_coeffs.len() != EXPECTED {
        return Err(LightingError::ShCoefficientMismatch {
            actual: sh_coeffs.len(),
            expected: EXPECTED,
            degree: 2,
        });
    }
    validate_mesh(mesh)?;

    let n_verts = mesh.vertices.len();
    let mut colors = Vec::with_capacity(n_verts);

    for i in 0..n_verts {
        let n = normalize3(normal_to_arr(mesh, i));
        let basis = sh_basis_9(n[0], n[1], n[2]);

        let r: f32 = sh_coeffs[0..9]
            .iter()
            .zip(basis.iter())
            .map(|(c, y)| c * y)
            .sum();
        let g: f32 = sh_coeffs[9..18]
            .iter()
            .zip(basis.iter())
            .map(|(c, y)| c * y)
            .sum();
        let b: f32 = sh_coeffs[18..27]
            .iter()
            .zip(basis.iter())
            .map(|(c, y)| c * y)
            .sum();

        colors.push(clamp3([r, g, b], 0.0, 1.0));
    }

    Ok(LightingResult {
        vertex_colors: colors,
        n_vertices: n_verts,
    })
}

/// Compute a simple per-vertex ambient occlusion approximation.
///
/// For each vertex the average alignment of its neighboring face normals with
/// its own normal is computed. Vertices whose neighbors tend to point away are
/// considered more occluded.
///
/// Returns a `Vec<f32>` of per-vertex AO factors in `[0, 1]`:
/// - `1.0` = fully unoccluded
/// - `0.0` = fully occluded (when `strength = 1.0`)
///
/// `strength` in `[0, 1]` scales the effect.
///
/// # Errors
///
/// Returns [`LightingError::EmptyMesh`] or [`LightingError::NormalMismatch`].
pub fn approximate_ambient_occlusion(
    mesh: &crate::Mesh,
    strength: f32,
) -> Result<Vec<f32>, LightingError> {
    validate_mesh(mesh)?;

    let n_verts = mesh.vertices.len();

    // Build per-vertex face adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
    for (fi, face) in mesh.faces.iter().enumerate() {
        for &vi in face {
            adj[vi as usize].push(fi);
        }
    }

    // Precompute face normals
    let face_normals: Vec<[f32; 3]> = mesh
        .faces
        .iter()
        .map(|face| {
            let v0 = vertex_to_arr(mesh, face[0] as usize);
            let v1 = vertex_to_arr(mesh, face[1] as usize);
            let v2 = vertex_to_arr(mesh, face[2] as usize);
            let e1 = sub3(v1, v0);
            let e2 = sub3(v2, v0);
            // Cross product
            let cx = e1[1] * e2[2] - e1[2] * e2[1];
            let cy = e1[2] * e2[0] - e1[0] * e2[2];
            let cz = e1[0] * e2[1] - e1[1] * e2[0];
            normalize3([cx, cy, cz])
        })
        .collect();

    let clamped_strength = strength.clamp(0.0, 1.0);
    let low = 1.0 - clamped_strength;

    let mut ao = Vec::with_capacity(n_verts);
    for (vert_idx, neighbor_faces) in adj.iter().enumerate().take(n_verts) {
        let vn = normalize3(normal_to_arr(mesh, vert_idx));

        let alignment_score = if neighbor_faces.is_empty() {
            1.0_f32
        } else {
            let sum: f32 = neighbor_faces
                .iter()
                .map(|&fi| dot3(vn, face_normals[fi]).max(0.0))
                .sum();
            sum / neighbor_faces.len() as f32
        };

        // lerp between (1 - strength) and 1.0 based on alignment
        let ao_factor = lerp3([low; 3], [1.0; 3], alignment_score)[0];
        ao.push(ao_factor.clamp(0.0, 1.0));
    }

    Ok(ao)
}

/// Build a standard studio three-point lighting setup.
///
/// Returns `(directional_lights, ambient_light)` where `directional_lights`
/// contains:
/// 1. **Key light** — slightly above-right, warm white, intensity `1.0`
/// 2. **Fill light** — left side, cool, intensity `0.4`
/// 3. **Rim/back light** — behind, intensity `0.3`
#[must_use]
pub fn studio_lighting() -> (Vec<DirectionalLight>, AmbientLight) {
    let key = DirectionalLight::new(
        normalize3([0.5, 1.0, 0.8]), // above, slightly right and front
        [1.0, 0.95, 0.88],           // warm white
        1.0,
    );
    let fill = DirectionalLight::new(
        normalize3([-1.0, 0.3, 0.5]), // left, slightly above/front
        [0.7, 0.82, 1.0],             // cool blue-ish
        0.4,
    );
    let rim = DirectionalLight::new(
        normalize3([0.0, 0.2, -1.0]), // mostly behind
        [1.0, 1.0, 1.0],
        0.3,
    );
    let ambient = AmbientLight::new([1.0, 1.0, 1.0], 0.05);
    (vec![key, fill, rim], ambient)
}

/// Add a rim lighting effect to an existing [`LightingResult`].
///
/// The rim factor is `max(0, -dot(N, V))^2` — surfaces that face away from
/// the camera receive the most rim color.
///
/// `rim_color` and `rim_intensity` control the added hue and strength.
///
/// # Errors
///
/// Returns [`LightingError::EmptyMesh`] or [`LightingError::NormalMismatch`].
pub fn apply_rim_lighting(
    result: &mut LightingResult,
    mesh: &crate::Mesh,
    camera_pos: [f32; 3],
    rim_color: [f32; 3],
    rim_intensity: f32,
) -> Result<(), LightingError> {
    validate_mesh(mesh)?;
    if mesh.vertices.len() != result.n_vertices {
        return Err(LightingError::NormalMismatch {
            normals: result.n_vertices,
            vertices: mesh.vertices.len(),
        });
    }

    for i in 0..result.n_vertices {
        let vertex = vertex_to_arr(mesh, i);
        let normal = normalize3(normal_to_arr(mesh, i));
        let view_dir = normalize3(sub3(camera_pos, vertex));

        // Rim factor: surfaces facing away from camera
        let rim_factor = (-dot3(normal, view_dir)).max(0.0).powi(2);
        let rim_add = scale3(rim_color, rim_intensity * rim_factor);

        result.vertex_colors[i] = clamp3(add3(result.vertex_colors[i], rim_add), 0.0, 1.0);
    }

    Ok(())
}

/// Tone-map an HDR color slice to `[0, 1]` using the Reinhard operator.
///
/// For each channel: `c_out = c_in / (1 + c_in)`.
#[must_use]
pub fn reinhard_tone_map(colors: &[[f32; 3]]) -> Vec<[f32; 3]> {
    colors
        .iter()
        .map(|&[r, g, b]| [r / (1.0 + r), g / (1.0 + g), b / (1.0 + b)])
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a tiny mesh for tests
    // -----------------------------------------------------------------------

    /// Build a minimal single-triangle mesh (pointing in +Z direction).
    fn make_triangle_mesh() -> crate::Mesh {
        use nalgebra as na;
        let vertices = vec![
            na::Point3::new(0.0_f32, 0.0, 0.0),
            na::Point3::new(1.0_f32, 0.0, 0.0),
            na::Point3::new(0.0_f32, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        crate::Mesh::new(vertices, faces)
    }

    /// Build a quad mesh (two triangles, all normals pointing +Z).
    fn make_quad_mesh() -> crate::Mesh {
        use nalgebra as na;
        let vertices = vec![
            na::Point3::new(0.0_f32, 0.0, 0.0),
            na::Point3::new(1.0_f32, 0.0, 0.0),
            na::Point3::new(1.0_f32, 1.0, 0.0),
            na::Point3::new(0.0_f32, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2], [0, 2, 3]];
        crate::Mesh::new(vertices, faces)
    }

    // -----------------------------------------------------------------------
    // 1. Math helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot3_orthogonal() {
        assert_eq!(dot3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0);
    }

    #[test]
    fn test_dot3_parallel() {
        assert!((dot3([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize3_unit() {
        let v = normalize3([3.0, 0.0, 0.0]);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!(v[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize3_zero_returns_zero() {
        let v = normalize3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_reflect3_against_y_axis() {
        // Incident = [1, -1, 0], normal = [0, 1, 0]
        // reflect = [1,-1,0] - 2*(-1)*[0,1,0] = [1, 1, 0]
        let r = reflect3([1.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((r[0] - 1.0).abs() < 1e-5, "r[0] = {}", r[0]);
        assert!((r[1] - 1.0).abs() < 1e-5, "r[1] = {}", r[1]);
        assert!(r[2].abs() < 1e-5);
    }

    #[test]
    fn test_add3_basic() {
        let r = add3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(r, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub3_basic() {
        let r = sub3([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]);
        assert_eq!(r, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_scale3_basic() {
        let r = scale3([1.0, 2.0, 3.0], 2.0);
        assert_eq!(r, [2.0, 4.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // 2. lambertian_diffuse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lambertian_aligned_with_light() {
        // Normal = light direction → N·L = 1 → max illumination
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        let c = lambertian_diffuse([0.0, 0.0, 1.0], &light);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 1.0).abs() < 1e-5);
        assert!((c[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lambertian_perpendicular_is_zero() {
        let light = DirectionalLight::new([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        let c = lambertian_diffuse([0.0, 1.0, 0.0], &light);
        assert!(c[0].abs() < 1e-5);
        assert!(c[1].abs() < 1e-5);
        assert!(c[2].abs() < 1e-5);
    }

    #[test]
    fn test_lambertian_behind_is_zero() {
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        let c = lambertian_diffuse([0.0, 0.0, -1.0], &light);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_lambertian_intensity_scaling() {
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.5);
        let c = lambertian_diffuse([0.0, 0.0, 1.0], &light);
        assert!((c[0] - 0.5).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // 3. phong_vertex tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_phong_vertex_diffuse_only() {
        // Shininess = 0 means specular = 1 regardless → override with matte
        let material = PhongMaterial::matte();
        let ambient = AmbientLight::new([0.0, 0.0, 0.0], 0.0);
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        // Normal and view both +Z, light +Z
        let c = phong_vertex(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            &light,
            &ambient,
            &material,
        );
        // Diffuse = Kd * 1.0 (N·L = 1)
        assert!((c[0] - material.diffuse_color[0].min(1.0)).abs() < 0.1);
    }

    #[test]
    fn test_phong_vertex_ambient_only() {
        let material = PhongMaterial::default();
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 1.0);
        // Light behind surface → diffuse=0, specular=0
        let light = DirectionalLight::new([0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 1.0);
        let c = phong_vertex(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            &light,
            &ambient,
            &material,
        );
        // Result should be ≈ Ka (clamped)
        assert!(c[0] >= 0.0 && c[0] <= 1.0);
    }

    #[test]
    fn test_phong_vertex_specular_present() {
        // Perfect mirror setup: normal +Z, view +Z, light +Z → R·V should be 1
        let material = PhongMaterial {
            specular_color: [1.0, 1.0, 1.0],
            shininess: 8.0,
            ..Default::default()
        };
        let ambient = AmbientLight::new([0.0, 0.0, 0.0], 0.0);
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        let c = phong_vertex(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            &light,
            &ambient,
            &material,
        );
        // Fully lit surface — should be clamped to [0,1] and bright
        assert!(c[0] > 0.5, "c[0] = {}", c[0]);
    }

    #[test]
    fn test_phong_vertex_output_clamped() {
        // Even extreme inputs produce values in [0, 1]
        let material = PhongMaterial::glossy();
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 2.0);
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 5.0);
        let c = phong_vertex(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            &light,
            &ambient,
            &material,
        );
        for ch in c {
            assert!((0.0..=1.0).contains(&ch), "channel out of range: {ch}");
        }
    }

    // -----------------------------------------------------------------------
    // 4. shade_mesh_directional tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shade_mesh_directional_returns_one_per_vertex() {
        let mesh = make_triangle_mesh();
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let result = shade_mesh_directional(&mesh, &light, &ambient, &material, [0.0, 0.0, 1.0]);
        assert!(result.is_ok());
        let r = result.expect("expected ok result");
        assert_eq!(r.n_vertices, 3);
        assert_eq!(r.vertex_colors.len(), 3);
    }

    #[test]
    fn test_shade_mesh_directional_empty_mesh_error() {
        let mesh = crate::Mesh::new(vec![], vec![]);
        let light = DirectionalLight::default();
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let result = shade_mesh_directional(&mesh, &light, &ambient, &material, [0.0, 0.0, 1.0]);
        assert!(matches!(result, Err(LightingError::EmptyMesh)));
    }

    #[test]
    fn test_shade_mesh_directional_colors_clamped() {
        let mesh = make_quad_mesh();
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 10.0);
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 5.0);
        let material = PhongMaterial::glossy();
        let result = shade_mesh_directional(&mesh, &light, &ambient, &material, [0.0, 0.0, 2.0])
            .expect("shade_mesh_directional failed");
        for &[r, g, b] in &result.vertex_colors {
            assert!((0.0..=1.0).contains(&r));
            assert!((0.0..=1.0).contains(&g));
            assert!((0.0..=1.0).contains(&b));
        }
    }

    #[test]
    fn test_shade_mesh_directional_single_vertex_mesh() {
        use nalgebra as na;
        // A degenerate mesh with 1 vertex and no faces
        let mesh_no_face = crate::Mesh::new(vec![na::Point3::new(0.0_f32, 0.0, 0.0)], vec![]);
        // normals won't be set (no faces) but vertex count matches
        let light = DirectionalLight::default();
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let result =
            shade_mesh_directional(&mesh_no_face, &light, &ambient, &material, [0.0, 0.0, 1.0]);
        assert!(result.is_ok());
        assert_eq!(result.expect("result ok").n_vertices, 1);
    }

    #[test]
    fn test_shade_mesh_directional_quad_mesh() {
        let mesh = make_quad_mesh();
        let light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let result = shade_mesh_directional(&mesh, &light, &ambient, &material, [0.0, 0.0, 2.0])
            .expect("shade failed");
        assert_eq!(result.n_vertices, 4);
    }

    // -----------------------------------------------------------------------
    // 5. shade_mesh_multi_light tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shade_mesh_multi_light_no_lights() {
        let mesh = make_triangle_mesh();
        let ambient = AmbientLight::new([1.0, 1.0, 1.0], 0.2);
        let material = PhongMaterial::default();
        let result = shade_mesh_multi_light(&mesh, &[], &[], &ambient, &material, [0.0, 0.0, 1.0])
            .expect("multi-light failed");
        assert_eq!(result.n_vertices, 3);
        // With no lights only ambient, colors should be low but > 0
        for &[r, g, b] in &result.vertex_colors {
            assert!((0.0..=1.0).contains(&r));
            assert!((0.0..=1.0).contains(&g));
            assert!((0.0..=1.0).contains(&b));
        }
    }

    #[test]
    fn test_shade_mesh_multi_light_two_lights_brighter() {
        let mesh = make_quad_mesh();
        let ambient = AmbientLight::new([0.0, 0.0, 0.0], 0.0);
        let material = PhongMaterial {
            diffuse_color: [1.0, 1.0, 1.0],
            specular_color: [0.0, 0.0, 0.0],
            shininess: 32.0,
            ambient_color: [0.0, 0.0, 0.0],
        };

        let one_light = DirectionalLight::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.5);
        let result_one = shade_mesh_multi_light(
            &mesh,
            std::slice::from_ref(&one_light),
            &[],
            &ambient,
            &material,
            [0.0, 0.0, 2.0],
        )
        .expect("one light failed");

        let light2 = DirectionalLight::new([0.0, 0.5, 0.866], [1.0, 1.0, 1.0], 0.5);
        let result_two = shade_mesh_multi_light(
            &mesh,
            &[one_light, light2],
            &[],
            &ambient,
            &material,
            [0.0, 0.0, 2.0],
        )
        .expect("two lights failed");

        // Two lights should produce at least as bright a result as one
        let lum_one = result_one.mean_luminance();
        let lum_two = result_two.mean_luminance();
        assert!(
            lum_two >= lum_one - 1e-5,
            "two lights should be at least as bright"
        );
    }

    #[test]
    fn test_shade_mesh_multi_light_with_point_light() {
        let mesh = make_quad_mesh();
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let plight = PointLight::new([0.5, 0.5, 1.0], [1.0, 1.0, 1.0], 1.0, 0.5);
        let result =
            shade_mesh_multi_light(&mesh, &[], &[plight], &ambient, &material, [0.0, 0.0, 2.0])
                .expect("point light failed");
        assert_eq!(result.n_vertices, 4);
    }

    #[test]
    fn test_shade_mesh_multi_light_empty_mesh_error() {
        let mesh = crate::Mesh::new(vec![], vec![]);
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let result = shade_mesh_multi_light(&mesh, &[], &[], &ambient, &material, [0.0, 0.0, 1.0]);
        assert!(matches!(result, Err(LightingError::EmptyMesh)));
    }

    // -----------------------------------------------------------------------
    // 6. shade_mesh_sh_lighting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_lighting_uniform_ambient() {
        let mesh = make_quad_mesh();
        // All coefficients 0 except Y_00 (index 0, 9, 18) → constant color
        let mut sh_coeffs = vec![0.0_f32; 27];
        // Y_00 = 0.282095; to get R=0.5: coeff_r = 0.5 / 0.282095 ≈ 1.773
        sh_coeffs[0] = 0.5 / 0.282_095;
        sh_coeffs[9] = 0.4 / 0.282_095;
        sh_coeffs[18] = 0.3 / 0.282_095;
        let result = shade_mesh_sh_lighting(&mesh, &sh_coeffs).expect("SH lighting failed");
        // All vertices should get approximately the same color
        for &[r, g, b] in &result.vertex_colors {
            assert!((r - 0.5).abs() < 0.02, "r = {r}");
            assert!((g - 0.4).abs() < 0.02, "g = {g}");
            assert!((b - 0.3).abs() < 0.02, "b = {b}");
        }
    }

    #[test]
    fn test_sh_lighting_wrong_coeff_count() {
        let mesh = make_quad_mesh();
        let sh_wrong = vec![0.0_f32; 10];
        let result = shade_mesh_sh_lighting(&mesh, &sh_wrong);
        assert!(matches!(
            result,
            Err(LightingError::ShCoefficientMismatch { .. })
        ));
    }

    #[test]
    fn test_sh_lighting_empty_mesh_error() {
        let mesh = crate::Mesh::new(vec![], vec![]);
        let sh = vec![0.0_f32; 27];
        let result = shade_mesh_sh_lighting(&mesh, &sh);
        assert!(matches!(result, Err(LightingError::EmptyMesh)));
    }

    #[test]
    fn test_sh_lighting_output_clamped() {
        let mesh = make_quad_mesh();
        // Large coefficients → clamped to [0,1]
        let sh_big = vec![100.0_f32; 27];
        let result = shade_mesh_sh_lighting(&mesh, &sh_big).expect("SH with big coefficients");
        for &[r, g, b] in &result.vertex_colors {
            assert!((0.0..=1.0).contains(&r));
            assert!((0.0..=1.0).contains(&g));
            assert!((0.0..=1.0).contains(&b));
        }
    }

    // -----------------------------------------------------------------------
    // 7. approximate_ambient_occlusion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ao_flat_mesh_uniform() {
        // A quad mesh with all normals +Z: all adjacent face normals align → ao ≈ 1.0
        let mesh = make_quad_mesh();
        let ao = approximate_ambient_occlusion(&mesh, 1.0).expect("AO failed");
        assert_eq!(ao.len(), 4);
        for &v in &ao {
            assert!(v >= 0.5, "Expected ao ≥ 0.5 on flat mesh, got {v}");
        }
    }

    #[test]
    fn test_ao_strength_zero_returns_all_ones() {
        let mesh = make_quad_mesh();
        let ao = approximate_ambient_occlusion(&mesh, 0.0).expect("AO strength=0 failed");
        for &v in &ao {
            assert!((v - 1.0).abs() < 1e-5, "Expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_ao_empty_mesh_error() {
        let mesh = crate::Mesh::new(vec![], vec![]);
        let result = approximate_ambient_occlusion(&mesh, 0.5);
        assert!(matches!(result, Err(LightingError::EmptyMesh)));
    }

    // -----------------------------------------------------------------------
    // 8. LightingResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lighting_result_to_rgba_u8() {
        let result = LightingResult {
            vertex_colors: vec![[1.0, 0.5, 0.0]],
            n_vertices: 1,
        };
        let rgba = result.to_rgba_u8();
        assert_eq!(rgba.len(), 4);
        assert_eq!(rgba[0], 255);
        assert!((i32::from(rgba[1]) - 128).abs() <= 1, "green = {}", rgba[1]);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 255); // alpha = 255
    }

    #[test]
    fn test_lighting_result_mean_luminance_black() {
        let result = LightingResult {
            vertex_colors: vec![[0.0, 0.0, 0.0]; 4],
            n_vertices: 4,
        };
        assert_eq!(result.mean_luminance(), 0.0);
    }

    #[test]
    fn test_lighting_result_mean_luminance_white() {
        let result = LightingResult {
            vertex_colors: vec![[1.0, 1.0, 1.0]; 4],
            n_vertices: 4,
        };
        assert!((result.mean_luminance() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lighting_result_to_flat_rgb() {
        let result = LightingResult {
            vertex_colors: vec![[1.0, 0.5, 0.25], [0.1, 0.2, 0.3]],
            n_vertices: 2,
        };
        let flat = result.to_flat_rgb();
        assert_eq!(flat.len(), 6);
        assert!((flat[0] - 1.0).abs() < 1e-6);
        assert!((flat[1] - 0.5).abs() < 1e-6);
        assert!((flat[2] - 0.25).abs() < 1e-6);
        assert!((flat[3] - 0.1).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 9. PhongMaterial tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_phong_material_default() {
        let m = PhongMaterial::default();
        assert_eq!(m.diffuse_color, [0.8, 0.7, 0.6]);
        assert_eq!(m.specular_color, [0.3, 0.3, 0.3]);
        assert!((m.shininess - 32.0).abs() < 1e-5);
        assert_eq!(m.ambient_color, [0.2, 0.15, 0.1]);
    }

    #[test]
    fn test_phong_material_skin() {
        let m = PhongMaterial::skin();
        // Should be equivalent to default
        let d = PhongMaterial::default();
        assert_eq!(m.diffuse_color, d.diffuse_color);
        assert_eq!(m.shininess, d.shininess);
    }

    #[test]
    fn test_phong_material_matte_low_shininess() {
        let m = PhongMaterial::matte();
        assert!(
            m.shininess < 10.0,
            "matte shininess should be low, got {}",
            m.shininess
        );
        assert!(
            m.specular_color[0] < 0.2,
            "matte specular should be near zero, got {}",
            m.specular_color[0]
        );
    }

    #[test]
    fn test_phong_material_glossy_high_shininess() {
        let m = PhongMaterial::glossy();
        assert!(
            m.shininess > 64.0,
            "glossy shininess should be high, got {}",
            m.shininess
        );
        assert!(
            m.specular_color[0] > 0.3,
            "glossy specular should be high, got {}",
            m.specular_color[0]
        );
    }

    // -----------------------------------------------------------------------
    // 10. studio_lighting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_studio_lighting_returns_three_directional_lights() {
        let (lights, _ambient) = studio_lighting();
        assert_eq!(lights.len(), 3, "studio lighting should have 3 lights");
    }

    #[test]
    fn test_studio_lighting_key_light_intensity() {
        let (lights, _) = studio_lighting();
        // Key light is first and has intensity 1.0
        assert!((lights[0].intensity - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // 11. reinhard_tone_map tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reinhard_zero_maps_to_zero() {
        let result = reinhard_tone_map(&[[0.0, 0.0, 0.0]]);
        assert_eq!(result[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_reinhard_one_maps_to_half() {
        let result = reinhard_tone_map(&[[1.0, 1.0, 1.0]]);
        for &ch in &result[0] {
            assert!((ch - 0.5).abs() < 1e-5, "Expected 0.5, got {ch}");
        }
    }

    #[test]
    fn test_reinhard_two_maps_to_two_thirds() {
        let result = reinhard_tone_map(&[[2.0, 2.0, 2.0]]);
        for &ch in &result[0] {
            assert!((ch - 2.0 / 3.0).abs() < 1e-5, "Expected 2/3, got {ch}");
        }
    }

    // -----------------------------------------------------------------------
    // 12. apply_rim_lighting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rim_lighting_adds_color_from_behind() {
        // Camera at [0,0,2], rim light from behind
        // Mesh face normals are +Z; view_dir ≈ +Z for all vertices
        // dot(normal, view_dir) ≈ 1 → rim_factor = max(0, -1)^2 = 0
        // So rim on front-facing surface should be near zero.
        let mesh = make_quad_mesh();
        let light = DirectionalLight::default();
        let ambient = AmbientLight::default();
        let material = PhongMaterial::default();
        let mut result =
            shade_mesh_directional(&mesh, &light, &ambient, &material, [0.0, 0.0, 2.0])
                .expect("shade failed");
        let before = result.mean_luminance();
        apply_rim_lighting(&mut result, &mesh, [0.0, 0.0, 2.0], [1.0, 1.0, 1.0], 1.0)
            .expect("rim lighting failed");
        // After rim from behind camera, front-facing normals get near-zero rim; brightness ≈ same
        let after = result.mean_luminance();
        assert!(
            after >= before - 1e-4,
            "rim from behind camera should not reduce brightness"
        );
    }

    #[test]
    fn test_rim_lighting_from_behind_camera_adds_brightness() {
        // Camera behind mesh (+Z), rim color bright, rim from same direction
        // Flip: use camera at [0,0,-2] (behind mesh), normals +Z
        // view_dir = normalize([0,0,0] - v) ≈ -Z; normal = +Z → dot(n, view) ≈ -1 → rim_factor = 1
        let mesh = make_quad_mesh();
        let light = DirectionalLight::new([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], 0.0);
        let ambient = AmbientLight::new([0.0, 0.0, 0.0], 0.0);
        let material = PhongMaterial {
            diffuse_color: [0.0, 0.0, 0.0],
            specular_color: [0.0, 0.0, 0.0],
            shininess: 1.0,
            ambient_color: [0.0, 0.0, 0.0],
        };
        let mut result =
            shade_mesh_directional(&mesh, &light, &ambient, &material, [0.0, 0.0, -2.0])
                .expect("shade failed");

        apply_rim_lighting(
            &mut result,
            &mesh,
            [0.0, 0.0, -2.0], // camera behind mesh
            [1.0, 1.0, 1.0],
            1.0,
        )
        .expect("rim lighting failed");

        // All normals face +Z, camera is at -Z → view_dir is -Z → dot(+Z, -Z) = -1
        // rim_factor = max(0, -(-1))^2 = 1.0 → should add significant brightness
        let lum = result.mean_luminance();
        assert!(
            lum > 0.3,
            "Rim from behind should add brightness, got luminance = {lum}"
        );
    }

    #[test]
    fn test_rim_lighting_empty_mesh_error() {
        let mesh = crate::Mesh::new(vec![], vec![]);
        let mut result = LightingResult {
            vertex_colors: vec![],
            n_vertices: 0,
        };
        // validate_mesh returns EmptyMesh error
        let r = apply_rim_lighting(&mut result, &mesh, [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0);
        assert!(matches!(r, Err(LightingError::EmptyMesh)));
    }

    // -----------------------------------------------------------------------
    // Extra: DirectionalLight and PointLight validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_directional_light_invalid_color() {
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            color: [2.0, 0.5, 0.5], // out of range
            intensity: 1.0,
        };
        assert!(matches!(
            light.validate(),
            Err(LightingError::InvalidLightColor)
        ));
    }

    #[test]
    fn test_point_light_valid() {
        let light = PointLight::new([0.0, 1.0, 0.0], [0.5, 0.5, 0.5], 1.0, 1.0);
        assert!(light.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Extra: mul3 and lerp3 helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul3_component_wise() {
        let r = mul3([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]);
        assert_eq!(r, [10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_lerp3_midpoint() {
        let r = lerp3([0.0, 0.0, 0.0], [2.0, 4.0, 6.0], 0.5);
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp3_clamps_both_ends() {
        let r = clamp3([-0.5, 0.5, 1.5], 0.0, 1.0);
        assert_eq!(r[0], 0.0);
        assert!((r[1] - 0.5).abs() < 1e-5);
        assert_eq!(r[2], 1.0);
    }
}
