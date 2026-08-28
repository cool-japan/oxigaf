//! The heat method for geodesic distance (Crane, Weischedel & Wardetzky, 2013).
//!
//! # The algorithm
//!
//! Geodesic distance is recovered from a short burst of heat diffusion in three
//! steps:
//!
//! 1. **Diffuse.** Solve the backward-Euler heat equation
//!    `(M + t·Lc) u = δ_source`, where `M` is the lumped (barycentric) mass
//!    matrix, `Lc` the cotangent Laplacian in the positive-semidefinite
//!    convention used here, and `t` a time step on the order of the squared
//!    mean edge length.
//! 2. **Normalise the gradient.** Varadhan's formula says `u ≈ exp(−d²/4t)`, so
//!    `∇u` points *away* from the source and `X = −∇u/‖∇u‖` is a unit vector
//!    field pointing along geodesics *toward* it. Normalising discards the
//!    magnitude, which is exactly where the diffusion approximation is
//!    inaccurate — this is what makes the method robust.
//! 3. **Integrate.** Recover `φ` with `∇φ ≈ X` by solving the Poisson equation
//!    `Lc φ = ∇·X` in the least-squares sense, then shift so the source is 0.
//!
//! The result is a genuine distance in mesh units: it depends on vertex
//! positions, scales linearly with the mesh, and vanishes at the source.
//!
//! # Why not `−ln(u)`
//!
//! Inverting Varadhan's formula pointwise would give `d = sqrt(−4t·ln u)`, not
//! `−ln u`, and either way the pointwise inversion is useless in practice: `u`
//! decays so fast that it underflows to zero a few rings out from the source,
//! where any logarithm saturates at a constant that carries no geometry at all.
//! Steps 2 and 3 exist precisely to avoid that.
//!
//! # Linear solves
//!
//! Both systems are sparse, symmetric and positive (semi-)definite, so they are
//! solved with a Jacobi-preconditioned conjugate-gradient iteration built on
//! the CSR structures below — no external solver, no dense factorisation.

use super::{GeodesicError, GeodesicMesh};

/// Sparse symmetric matrix in CSR form, diagonal stored separately.
///
/// Row `i` holds the off-diagonal entries `(col_idx[k], values[k])` for
/// `k in row_ptr[i]..row_ptr[i + 1]`, plus `diagonal[i]`.
struct SparseSym {
    /// Row offsets, length `n + 1`.
    row_ptr: Vec<usize>,
    /// Column index per off-diagonal entry.
    col_idx: Vec<usize>,
    /// Value per off-diagonal entry.
    values: Vec<f32>,
    /// Diagonal entries, length `n`.
    diagonal: Vec<f32>,
}

impl SparseSym {
    /// Number of rows.
    fn n(&self) -> usize {
        self.diagonal.len()
    }

    /// `out = self · x`.
    fn mul_into(&self, x: &[f32], out: &mut [f32]) {
        for i in 0..self.n() {
            let mut acc = self.diagonal[i] * x[i];
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc += self.values[k] * x[self.col_idx[k]];
            }
            out[i] = acc;
        }
    }

    /// Return a copy of `self` with `scale * diag` added to the diagonal.
    ///
    /// The off-diagonal pattern is untouched, which is what makes adding the
    /// (diagonal, lumped) mass matrix cheap.
    fn plus_scaled_diagonal(&self, scale: f32, diag: &[f32]) -> Self {
        let diagonal = self
            .diagonal
            .iter()
            .zip(diag.iter())
            .map(|(&d, &m)| d + scale * m)
            .collect();
        Self {
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_idx.clone(),
            values: self.values.clone(),
            diagonal,
        }
    }
}

/// Cotangent of the angle at `a` in triangle `(a, b, c)`.
///
/// Computed as `cos/sin = dot(u, v) / ‖u × v‖` with `u = b − a`, `v = c − a`,
/// which is numerically stable and avoids an `acos`. Degenerate triangles
/// (zero cross-product) contribute nothing.
fn cot_at(vertex_a: [f32; 3], vertex_b: [f32; 3], vertex_c: [f32; 3]) -> f32 {
    let edge_u = [
        vertex_b[0] - vertex_a[0],
        vertex_b[1] - vertex_a[1],
        vertex_b[2] - vertex_a[2],
    ];
    let edge_v = [
        vertex_c[0] - vertex_a[0],
        vertex_c[1] - vertex_a[1],
        vertex_c[2] - vertex_a[2],
    ];
    let dot = edge_u[0] * edge_v[0] + edge_u[1] * edge_v[1] + edge_u[2] * edge_v[2];
    let cross = [
        edge_u[1] * edge_v[2] - edge_u[2] * edge_v[1],
        edge_u[2] * edge_v[0] - edge_u[0] * edge_v[2],
        edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0],
    ];
    let cross_norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if cross_norm < 1e-20 {
        0.0
    } else {
        dot / cross_norm
    }
}

/// Twice the area of triangle `(a, b, c)`.
fn double_area(vertex_a: [f32; 3], vertex_b: [f32; 3], vertex_c: [f32; 3]) -> f32 {
    let edge_u = [
        vertex_b[0] - vertex_a[0],
        vertex_b[1] - vertex_a[1],
        vertex_b[2] - vertex_a[2],
    ];
    let edge_v = [
        vertex_c[0] - vertex_a[0],
        vertex_c[1] - vertex_a[1],
        vertex_c[2] - vertex_a[2],
    ];
    let cross = [
        edge_u[1] * edge_v[2] - edge_u[2] * edge_v[1],
        edge_u[2] * edge_v[0] - edge_u[0] * edge_v[2],
        edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0],
    ];
    (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

/// Build the cotangent Laplacian `Lc` (positive semi-definite convention:
/// `Lc_ii = Σ w_ij`, `Lc_ij = −w_ij`) together with the lumped mass matrix.
///
/// `w_ij = (cot α + cot β) / 2` summed over the triangles sharing edge `(i, j)`;
/// `M_ii` is one third of the total area of the triangles incident to `i`.
///
/// `regularisation` is added to every diagonal entry so that the otherwise
/// singular (constant-nullspace) system stays solvable by conjugate gradients.
fn build_cotangent_operators(mesh: &GeodesicMesh, regularisation: f32) -> (SparseSym, Vec<f32>) {
    let n = mesh.n_vertices();

    // Accumulate symmetric edge weights in a per-vertex neighbour map.
    let mut neighbours: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    let mut mass = vec![0.0f32; n];

    /// Add `w` to the symmetric weight of edge `(i, j)`, creating it if new.
    fn add_weight(neighbours: &mut [Vec<(usize, f32)>], i: usize, j: usize, w: f32) {
        if let Some(entry) = neighbours[i].iter_mut().find(|(col, _)| *col == j) {
            entry.1 += w;
        } else {
            neighbours[i].push((j, w));
        }
    }

    for face in &mesh.faces {
        let [ia, ib, ic] = *face;
        let pa = mesh.vertices[ia];
        let pb = mesh.vertices[ib];
        let pc = mesh.vertices[ic];

        // Cotangent at each vertex weights the *opposite* edge.
        let cot_a = cot_at(pa, pb, pc);
        let cot_b = cot_at(pb, pc, pa);
        let cot_c = cot_at(pc, pa, pb);

        for (i, j, w) in [
            (ib, ic, cot_a * 0.5),
            (ic, ia, cot_b * 0.5),
            (ia, ib, cot_c * 0.5),
        ] {
            add_weight(&mut neighbours, i, j, w);
            add_weight(&mut neighbours, j, i, w);
        }

        // Barycentric lumped mass: each vertex takes a third of the area.
        let area = double_area(pa, pb, pc) * 0.5;
        let third = area / 3.0;
        mass[ia] += third;
        mass[ib] += third;
        mass[ic] += third;
    }

    // Flatten to CSR with `Lc_ij = -w_ij` off-diagonal and `Lc_ii = Σ w_ij`.
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let mut diagonal = vec![0.0f32; n];

    row_ptr.push(0);
    for (i, nbrs) in neighbours.iter().enumerate() {
        let mut deg = 0.0f32;
        for &(j, w) in nbrs {
            col_idx.push(j);
            values.push(-w);
            deg += w;
        }
        diagonal[i] = deg + regularisation;
        row_ptr.push(col_idx.len());
    }

    // A vertex touched by no non-degenerate triangle has an all-zero row, which
    // would make the system singular beyond what the regularisation absorbs.
    // Give it a unit diagonal so both solves stay well-posed; `heat_distances`
    // reports such a vertex as unreachable rather than trusting its value.
    for (d, &m) in diagonal.iter_mut().zip(mass.iter()) {
        if m <= 0.0 {
            *d = 1.0;
        }
    }

    (
        SparseSym {
            row_ptr,
            col_idx,
            values,
            diagonal,
        },
        mass,
    )
}

/// Solve `A x = b` by Jacobi-preconditioned conjugate gradients.
///
/// `A` must be symmetric positive (semi-)definite. Iteration stops when the
/// residual norm drops below `tol · ‖b‖` or after `max_iter` steps; the best
/// iterate is returned either way, since the heat method tolerates a partially
/// converged solve far better than it tolerates a failure.
fn solve_cg(matrix: &SparseSym, rhs: &[f32], max_iter: usize, tol: f32) -> Vec<f32> {
    let dim = matrix.n();
    let mut solution = vec![0.0f32; dim];
    let mut residual = rhs.to_vec();

    // Jacobi preconditioner: M⁻¹ = 1 / diag(A), guarded against zero pivots.
    let inv_diag: Vec<f32> = matrix
        .diagonal
        .iter()
        .map(|&d| if d.abs() > 1e-20 { 1.0 / d } else { 1.0 })
        .collect();

    let mut precond_residual: Vec<f32> = residual
        .iter()
        .zip(inv_diag.iter())
        .map(|(&ri, &m)| ri * m)
        .collect();
    let mut search_dir = precond_residual.clone();
    let mut rz: f32 = residual
        .iter()
        .zip(precond_residual.iter())
        .map(|(&ri, &zi)| ri * zi)
        .sum();

    let rhs_norm: f32 = rhs.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let threshold = if rhs_norm > 0.0 { tol * rhs_norm } else { tol };

    let mut mat_search_dir = vec![0.0f32; dim];

    for _ in 0..max_iter {
        let residual_norm: f32 = residual.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if residual_norm <= threshold || !residual_norm.is_finite() {
            break;
        }

        matrix.mul_into(&search_dir, &mut mat_search_dir);
        let search_dir_dot_ap: f32 = search_dir
            .iter()
            .zip(mat_search_dir.iter())
            .map(|(&pi, &api)| pi * api)
            .sum();
        if search_dir_dot_ap.abs() < 1e-30 || !search_dir_dot_ap.is_finite() {
            break;
        }

        let alpha = rz / search_dir_dot_ap;
        for i in 0..dim {
            solution[i] += alpha * search_dir[i];
            residual[i] -= alpha * mat_search_dir[i];
        }

        for i in 0..dim {
            precond_residual[i] = residual[i] * inv_diag[i];
        }
        let rz_new: f32 = residual
            .iter()
            .zip(precond_residual.iter())
            .map(|(&ri, &zi)| ri * zi)
            .sum();
        if rz.abs() < 1e-30 || !rz_new.is_finite() {
            break;
        }
        let beta = rz_new / rz;
        for i in 0..dim {
            search_dir[i] = precond_residual[i] + beta * search_dir[i];
        }
        rz = rz_new;
    }

    solution
}

/// Mean edge length over all triangle edges — the natural length scale for the
/// heat method's time step.
fn mean_edge_length(mesh: &GeodesicMesh) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for face in &mesh.faces {
        let [a, b, c] = *face;
        total += mesh.edge_length(a, b) + mesh.edge_length(b, c) + mesh.edge_length(c, a);
        count += 3;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Validate `time_step` (or derive the standard `t = mean_edge_length²`
/// heuristic) for the heat method.
fn resolve_time_step(mesh: &GeodesicMesh, time_step: Option<f32>) -> Result<f32, GeodesicError> {
    let h = mean_edge_length(mesh);
    if h <= 0.0 || !h.is_finite() {
        return Err(GeodesicError::NumericalError(
            "mesh has zero extent: every edge is degenerate, so distance is undefined".into(),
        ));
    }

    match time_step {
        Some(t) if t > 0.0 && t.is_finite() => Ok(t),
        Some(t) => Err(GeodesicError::InvalidConfig(format!(
            "time_step must be positive and finite, got {t}"
        ))),
        None => Ok(h * h),
    }
}

/// Step 1: solve `(M + t·Lc) u = δ_source` for the heat distribution `u`,
/// returning it together with the cotangent Laplacian and lumped mass matrix
/// (both reused by steps 2 and 3).
fn solve_heat_diffusion(
    mesh: &GeodesicMesh,
    sources: &[usize],
    t: f32,
    n_iter: usize,
) -> Result<(Vec<f32>, SparseSym, Vec<f32>), GeodesicError> {
    let n = mesh.n_vertices();

    // A small multiple of the mean weight keeps both systems positive definite
    // despite the constant nullspace, without perceptibly biasing the solution.
    let (laplacian, mass) = build_cotangent_operators(mesh, 1e-8);

    let total_area: f32 = mass.iter().sum();
    if total_area <= 0.0 || !total_area.is_finite() {
        return Err(GeodesicError::NumericalError(
            "mesh has zero total area: every face is degenerate, so distance is undefined".into(),
        ));
    }

    // `build_cotangent_operators` returns Lc in the positive-semidefinite
    // convention, so the backward-Euler operator is M + t·Lc.  Both sides are
    // divided through by t — giving (Lc + M/t) u = δ/t — because with the
    // standard t = h² choice that puts Lc (dimensionless cotangents) and M/t
    // (also O(1)) on the same scale, which is far better conditioned for the
    // conjugate-gradient solve.  The solution is identical, and step 2
    // normalises away its magnitude in any case.
    let heat_op = laplacian.plus_scaled_diagonal(1.0 / t, &mass);
    let mut rhs = vec![0.0f32; n];
    for &s in sources {
        rhs[s] += 1.0 / t;
    }
    let u = solve_cg(&heat_op, &rhs, n_iter, 1e-8);

    Ok((u, laplacian, mass))
}

/// Step 2: `X = −∇u / ‖∇u‖`, per face.
///
/// For a linear function on triangle (i, j, k) with unit normal N and area A,
///   `∇u = ( u_i (N × e_i) + u_j (N × e_j) + u_k (N × e_k) ) / (2A)`,
/// where `e_i` is the edge opposite vertex `i`.
fn compute_face_gradient_field(mesh: &GeodesicMesh, u: &[f32]) -> Vec<[f32; 3]> {
    let mut face_field: Vec<[f32; 3]> = Vec::with_capacity(mesh.n_faces());
    for face in &mesh.faces {
        let [ia, ib, ic] = *face;
        let pa = mesh.vertices[ia];
        let pb = mesh.vertices[ib];
        let pc = mesh.vertices[ic];

        let e_a = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]]; // opposite a
        let e_b = [pa[0] - pc[0], pa[1] - pc[1], pa[2] - pc[2]]; // opposite b
        let e_c = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]]; // opposite c

        let double_a = double_area(pa, pb, pc);
        if double_a < 1e-20 {
            face_field.push([0.0; 3]);
            continue;
        }
        let normal = {
            let u_vec = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let v_vec = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let cross = [
                u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
                u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
                u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0],
            ];
            [
                cross[0] / double_a,
                cross[1] / double_a,
                cross[2] / double_a,
            ]
        };

        let cross_n = |e: [f32; 3]| {
            [
                normal[1] * e[2] - normal[2] * e[1],
                normal[2] * e[0] - normal[0] * e[2],
                normal[0] * e[1] - normal[1] * e[0],
            ]
        };
        let (ga, gb, gc) = (cross_n(e_a), cross_n(e_b), cross_n(e_c));

        let mut grad = [0.0f32; 3];
        for k in 0..3 {
            grad[k] = (u[ia] * ga[k] + u[ib] * gb[k] + u[ic] * gc[k]) / double_a;
        }

        // Normalise and flip.  Heat decays away from the source, so ∇u points
        // *toward* it; distance grows away from it, so ∇φ points *away*.
        // Negating aligns the unit field with ∇φ, which is what step 3
        // integrates.
        let norm = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();
        if norm < 1e-20 || !norm.is_finite() {
            face_field.push([0.0; 3]);
        } else {
            face_field.push([-grad[0] / norm, -grad[1] / norm, -grad[2] / norm]);
        }
    }
    face_field
}

/// Step 3: the integrated divergence of `X` at each vertex.
///
/// The integrated divergence at vertex i, summed over incident faces, is
///   `(∇·X)_i = ½ Σ_f ( cot θ₁ (e₁ · X_f) + cot θ₂ (e₂ · X_f) )`,
/// with `e₁, e₂` the two edges of `f` emanating from `i` and `θ₁, θ₂` the
/// angles opposite them.
fn compute_divergence(mesh: &GeodesicMesh, face_field: &[[f32; 3]]) -> Vec<f32> {
    let mut divergence = vec![0.0f32; mesh.n_vertices()];
    for (f_idx, face) in mesh.faces.iter().enumerate() {
        let x_f = face_field[f_idx];
        if x_f == [0.0; 3] {
            continue;
        }
        let [ia, ib, ic] = *face;
        let pa = mesh.vertices[ia];
        let pb = mesh.vertices[ib];
        let pc = mesh.vertices[ic];

        // Angle at each vertex; the edge from i is weighted by the cotangent of
        // the angle opposite it within this triangle.
        let cot_a = cot_at(pa, pb, pc);
        let cot_b = cot_at(pb, pc, pa);
        let cot_c = cot_at(pc, pa, pb);

        let dot = |e: [f32; 3]| e[0] * x_f[0] + e[1] * x_f[1] + e[2] * x_f[2];

        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let bc = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];
        let ba = [-ab[0], -ab[1], -ab[2]];
        let ca = [-ac[0], -ac[1], -ac[2]];
        let cb = [-bc[0], -bc[1], -bc[2]];

        // At a: edges ab (opposite c) and ac (opposite b).
        divergence[ia] += 0.5 * (cot_c * dot(ab) + cot_b * dot(ac));
        // At b: edges ba (opposite c) and bc (opposite a).
        divergence[ib] += 0.5 * (cot_c * dot(ba) + cot_a * dot(bc));
        // At c: edges ca (opposite b) and cb (opposite a).
        divergence[ic] += 0.5 * (cot_b * dot(ca) + cot_a * dot(cb));
    }
    divergence
}

/// Sign convention: for a piecewise-linear φ with X = ∇φ, an edge dotted with
/// the gradient is just the endpoint difference (`e_ij` · ∇φ = `φ_j` − `φ_i`), so
/// the divergence formula collapses to `Σ_j` `w_ij` (`φ_j` − `φ_i`) — the Laplacian
/// in the *negative*-semidefinite convention.  `laplacian` here is the
/// positive one (diagonal +Σw, off-diagonal −w), i.e. its negation, so the
/// Poisson system to solve is `laplacian · φ = −∇·X`, not `+∇·X`.  Getting
/// this backwards yields φ = −distance, which the non-negativity clamp below
/// then flattens to all zeros.
///
/// This also shifts φ so the nearest source sits at exactly 0, then clamps
/// away solver noise and marks isolated vertices as unreachable.
fn solve_and_normalize_phi(
    laplacian: &SparseSym,
    divergence: &[f32],
    sources: &[usize],
    mass: &[f32],
    n_iter: usize,
) -> Vec<f32> {
    let neg_divergence: Vec<f32> = divergence.iter().map(|&d| -d).collect();
    let mut phi = solve_cg(laplacian, &neg_divergence, n_iter, 1e-8);

    let source_min = sources
        .iter()
        .map(|&s| phi[s])
        .fold(f32::INFINITY, f32::min);
    if source_min.is_finite() {
        for v in &mut phi {
            *v -= source_min;
        }
    }
    // Distance is non-negative by definition; clamp away solver noise near the
    // source, and drop any non-finite value to "unreachable".
    for (v, &m) in phi.iter_mut().zip(mass.iter()) {
        if m <= 0.0 {
            // Isolated vertex: no incident area, so it carries no distance.
            *v = f32::INFINITY;
        } else if v.is_finite() {
            *v = v.max(0.0);
        } else {
            *v = f32::INFINITY;
        }
    }
    for &s in sources {
        phi[s] = 0.0;
    }
    phi
}

/// Run the heat method from `sources`, returning per-vertex geodesic distances.
///
/// `time_step` is the backward-Euler `t`; when `None` the standard heuristic
/// `t = mean_edge_length²` is used. `n_iter` bounds the conjugate-gradient
/// iterations for each of the two linear solves.
///
/// # Errors
///
/// Returns [`GeodesicError::NumericalError`] when the mesh has zero total area
/// (every triangle degenerate), which leaves no metric to measure distance in.
pub(super) fn heat_distances(
    mesh: &GeodesicMesh,
    sources: &[usize],
    time_step: Option<f32>,
    n_iter: usize,
) -> Result<Vec<f32>, GeodesicError> {
    let t = resolve_time_step(mesh, time_step)?;
    let (u, laplacian, mass) = solve_heat_diffusion(mesh, sources, t, n_iter)?;
    let face_field = compute_face_gradient_field(mesh, &u);
    let divergence = compute_divergence(mesh, &face_field);
    let phi = solve_and_normalize_phi(&laplacian, &divergence, sources, &mass, n_iter);
    Ok(phi)
}
