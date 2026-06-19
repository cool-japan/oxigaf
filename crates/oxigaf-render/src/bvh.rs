//! Bounding Volume Hierarchy (BVH) for fast spatial queries over Gaussian primitives.
//!
//! Implements axis-aligned bounding boxes (AABB) and a SAH-based BVH for
//! frustum culling, sphere queries, and ray intersection tests.

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Create an AABB from explicit min/max corners.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Create a degenerate AABB at a single point.
    pub fn from_point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    /// Create an AABB from a center and radius (axis-aligned sphere enclosure).
    pub fn from_center_radius(center: [f32; 3], radius: f32) -> Self {
        Self {
            min: [center[0] - radius, center[1] - radius, center[2] - radius],
            max: [center[0] + radius, center[1] + radius, center[2] + radius],
        }
    }

    /// Return the smallest AABB that contains both self and other.
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Expand this AABB to include point p.
    pub fn expand(&mut self, p: [f32; 3]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.min[2] = self.min[2].min(p[2]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
        self.max[2] = self.max[2].max(p[2]);
    }

    /// Centroid of the AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Size (extent) of the AABB along each axis.
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Surface area of the AABB. Returns 0.0 for degenerate boxes.
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        2.0 * (s[0] * s[1] + s[1] * s[2] + s[2] * s[0])
    }

    /// Index (0=x, 1=y, 2=z) of the longest axis.
    pub fn longest_axis(&self) -> usize {
        let s = self.size();
        if s[0] >= s[1] && s[0] >= s[2] {
            0
        } else if s[1] >= s[2] {
            1
        } else {
            2
        }
    }

    /// True if point p is inside or on the boundary of this AABB.
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }

    /// True if the sphere (center, radius) overlaps this AABB.
    ///
    /// Uses the closest-point-on-AABB distance test.
    pub fn intersects_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        let mut sq_dist = 0.0_f32;
        for ((&v, &mn), &mx) in center.iter().zip(self.min.iter()).zip(self.max.iter()) {
            if v < mn {
                let d = mn - v;
                sq_dist += d * d;
            } else if v > mx {
                let d = v - mx;
                sq_dist += d * d;
            }
        }
        sq_dist <= radius * radius
    }

    /// True if a ray (origin, precomputed inv_dir) intersects this AABB.
    ///
    /// Uses the slab intersection method. Infinite values (from zero direction
    /// components) are handled correctly by IEEE 754 arithmetic.
    pub fn intersects_ray(&self, origin: [f32; 3], inv_dir: [f32; 3]) -> bool {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        for i in 0..3 {
            let t1 = (self.min[i] - origin[i]) * inv_dir[i];
            let t2 = (self.max[i] - origin[i]) * inv_dir[i];
            let t_near = t1.min(t2);
            let t_far = t1.max(t2);
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
        }

        t_min <= t_max && t_max >= 0.0
    }
}

// ─── BVH node ────────────────────────────────────────────────────────────────

const MAX_LEAF_SIZE: usize = 8;
const MAX_DEPTH: usize = 32;
const NUM_SAH_BUCKETS: usize = 16;

/// A node in the BVH tree.
pub enum BvhNode {
    /// Leaf node holding a set of Gaussian indices.
    Leaf {
        aabb: Aabb,
        gaussian_indices: Vec<usize>,
    },
    /// Internal node with two children.
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    /// Return the bounding box of this node.
    fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Internal { aabb, .. } => aabb,
        }
    }

    /// Count total tree nodes recursively.
    fn count_nodes(&self) -> usize {
        match self {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Internal { left, right, .. } => 1 + left.count_nodes() + right.count_nodes(),
        }
    }

    /// Maximum depth of the subtree rooted at this node.
    fn max_depth(&self) -> usize {
        match self {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Internal { left, right, .. } => 1 + left.max_depth().max(right.max_depth()),
        }
    }
}

// ─── GaussianBvh ─────────────────────────────────────────────────────────────

/// BVH over a set of Gaussians.
pub struct GaussianBvh {
    root: Option<BvhNode>,
    /// Per-Gaussian AABBs stored alongside the tree for per-primitive leaf tests.
    primitive_aabbs: Vec<Aabb>,
    num_gaussians: usize,
}

impl GaussianBvh {
    /// Build a BVH over Gaussians given their positions and log-space scales.
    ///
    /// For each Gaussian the AABB is `center ± max_scale` where
    /// `max_scale = exp(scale[i]).max()`.
    pub fn build(positions: &[[f32; 3]], scales: &[[f32; 3]]) -> Self {
        let n = positions.len();
        if n == 0 || scales.len() != n {
            return Self {
                root: None,
                primitive_aabbs: Vec::new(),
                num_gaussians: n,
            };
        }

        // Compute per-Gaussian AABBs.
        let aabbs: Vec<Aabb> = positions
            .iter()
            .zip(scales.iter())
            .map(|(pos, scale)| {
                let max_scale = scale[0].exp().max(scale[1].exp()).max(scale[2].exp());
                Aabb::from_center_radius(*pos, max_scale)
            })
            .collect();

        // Indices to recurse on.
        let mut indices: Vec<usize> = (0..n).collect();
        let root = build_recursive(&aabbs, &mut indices, 0);

        Self {
            root: Some(root),
            primitive_aabbs: aabbs,
            num_gaussians: n,
        }
    }

    /// Frustum cull: return Gaussian indices whose AABB overlaps the view frustum.
    ///
    /// `planes` is an array of 6 half-space planes `[a, b, c, d]` where the
    /// inside half-space satisfies `ax + by + cz + d >= 0`.
    pub fn frustum_cull(&self, planes: &[[f32; 4]; 6]) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            frustum_cull_recursive(root, planes, &self.primitive_aabbs, &mut result);
        }
        result
    }

    /// Sphere query: return all Gaussian indices whose AABB intersects the sphere.
    pub fn sphere_query(&self, center: [f32; 3], radius: f32) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            sphere_query_recursive(root, center, radius, &self.primitive_aabbs, &mut result);
        }
        result
    }

    /// Ray query: return all Gaussian indices hit by the ray (AABB-approximate).
    pub fn ray_query(&self, origin: [f32; 3], direction: [f32; 3]) -> Vec<usize> {
        let inv_dir = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            ray_query_recursive(root, origin, inv_dir, &self.primitive_aabbs, &mut result);
        }
        result
    }

    /// Total number of nodes in the BVH tree.
    pub fn num_nodes(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.count_nodes())
    }

    /// Maximum depth of the BVH tree.
    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.max_depth())
    }

    /// Number of Gaussians this BVH was built over.
    pub fn num_gaussians(&self) -> usize {
        self.num_gaussians
    }
}

// ─── Build helpers ────────────────────────────────────────────────────────────

/// Recursively build a BVH node from a subset of Gaussians (identified by
/// their indices into the original AABB array).
fn build_recursive(aabbs: &[Aabb], indices: &mut [usize], depth: usize) -> BvhNode {
    let n = indices.len();

    // Compute the bounding box of all primitives in this subset.
    let node_aabb = compute_union_aabb(aabbs, indices);

    // Terminal conditions: leaf if few primitives or max depth reached.
    if n <= MAX_LEAF_SIZE || depth >= MAX_DEPTH {
        return BvhNode::Leaf {
            aabb: node_aabb,
            gaussian_indices: indices.to_vec(),
        };
    }

    // Try SAH split; fall back to midpoint split.
    let split_pos = find_sah_split(aabbs, indices, &node_aabb);

    // Guard: if SAH produced a degenerate partition, try midpoint.
    let split_pos = if split_pos == 0 || split_pos == n {
        midpoint_split(aabbs, indices, &node_aabb)
    } else {
        split_pos
    };

    // If still degenerate, force leaf.
    if split_pos == 0 || split_pos == n {
        return BvhNode::Leaf {
            aabb: node_aabb,
            gaussian_indices: indices.to_vec(),
        };
    }

    let (left_indices, right_indices) = indices.split_at_mut(split_pos);
    let left = Box::new(build_recursive(aabbs, left_indices, depth + 1));
    let right = Box::new(build_recursive(aabbs, right_indices, depth + 1));

    BvhNode::Internal {
        aabb: node_aabb,
        left,
        right,
    }
}

/// Compute union AABB over a set of indices into the aabb slice.
fn compute_union_aabb(aabbs: &[Aabb], indices: &[usize]) -> Aabb {
    let first_idx = indices[0];
    let mut result = aabbs[first_idx];
    for &idx in &indices[1..] {
        result = result.union(&aabbs[idx]);
    }
    result
}

/// Find the best SAH split position, returning the partition index into `indices`.
///
/// Sorts `indices` by primitive centroid along the chosen axis so that
/// `indices[..split_pos]` is the left partition.
fn find_sah_split(aabbs: &[Aabb], indices: &mut [usize], parent_aabb: &Aabb) -> usize {
    let n = indices.len();
    let axis = parent_aabb.longest_axis();

    // Sort primitives by centroid along the chosen axis.
    indices.sort_by(|&a, &b| {
        let ca = aabbs[a].center()[axis];
        let cb = aabbs[b].center()[axis];
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build left-to-right prefix union of AABBs.
    let mut left_aabbs: Vec<Aabb> = Vec::with_capacity(n);
    {
        let mut running = aabbs[indices[0]];
        left_aabbs.push(running);
        for &idx in &indices[1..] {
            running = running.union(&aabbs[idx]);
            left_aabbs.push(running);
        }
    }

    // Build right-to-left suffix union of AABBs.
    let mut right_aabbs: Vec<Aabb> = vec![aabbs[indices[0]]; n]; // placeholder
    {
        let mut running = aabbs[indices[n - 1]];
        right_aabbs[n - 1] = running;
        for i in (0..n - 1).rev() {
            running = running.union(&aabbs[indices[i]]);
            right_aabbs[i] = running;
        }
    }

    // Evaluate NUM_SAH_BUCKETS candidate split positions.
    let bucket_size = n.max(NUM_SAH_BUCKETS) / NUM_SAH_BUCKETS;
    let mut best_cost = f32::INFINITY;
    let mut best_split = n / 2; // default midpoint

    let parent_area = parent_aabb.surface_area();
    let area_denom = if parent_area > 0.0 { parent_area } else { 1.0 };

    for bucket in 1..NUM_SAH_BUCKETS {
        let split = (bucket * bucket_size).min(n - 1);
        if split == 0 || split >= n {
            continue;
        }
        let left_area = left_aabbs[split - 1].surface_area();
        let right_area = right_aabbs[split].surface_area();
        let cost = (left_area * split as f32 + right_area * (n - split) as f32) / area_denom;
        if cost < best_cost {
            best_cost = cost;
            best_split = split;
        }
    }

    best_split
}

/// Fall back: split at median along longest axis.
fn midpoint_split(aabbs: &[Aabb], indices: &mut [usize], parent_aabb: &Aabb) -> usize {
    let n = indices.len();
    let axis = parent_aabb.longest_axis();

    indices.sort_by(|&a, &b| {
        let ca = aabbs[a].center()[axis];
        let cb = aabbs[b].center()[axis];
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    n / 2
}

// ─── Query helpers ────────────────────────────────────────────────────────────

/// Test whether an AABB is fully outside any of the 6 frustum planes.
///
/// Uses the p-vertex (positive vertex) optimization: for each plane, the
/// vertex of the AABB furthest in the plane-normal direction is chosen.
/// If that vertex is outside the plane, the whole box is outside.
fn aabb_outside_frustum(aabb: &Aabb, planes: &[[f32; 4]; 6]) -> bool {
    for plane in planes {
        let [a, b, c, d] = *plane;
        // P-vertex: component-wise max projection onto plane normal.
        let px = if a >= 0.0 { aabb.max[0] } else { aabb.min[0] };
        let py = if b >= 0.0 { aabb.max[1] } else { aabb.min[1] };
        let pz = if c >= 0.0 { aabb.max[2] } else { aabb.min[2] };
        if a * px + b * py + c * pz + d < 0.0 {
            return true; // Fully outside this plane.
        }
    }
    false
}

fn frustum_cull_recursive(
    node: &BvhNode,
    planes: &[[f32; 4]; 6],
    primitive_aabbs: &[Aabb],
    result: &mut Vec<usize>,
) {
    let aabb = node.aabb();
    if aabb_outside_frustum(aabb, planes) {
        return;
    }
    match node {
        BvhNode::Leaf {
            gaussian_indices, ..
        } => {
            // Per-primitive test to avoid false positives from the node envelope.
            for &idx in gaussian_indices {
                if let Some(prim_aabb) = primitive_aabbs.get(idx) {
                    if !aabb_outside_frustum(prim_aabb, planes) {
                        result.push(idx);
                    }
                }
            }
        }
        BvhNode::Internal { left, right, .. } => {
            frustum_cull_recursive(left, planes, primitive_aabbs, result);
            frustum_cull_recursive(right, planes, primitive_aabbs, result);
        }
    }
}

fn sphere_query_recursive(
    node: &BvhNode,
    center: [f32; 3],
    radius: f32,
    primitive_aabbs: &[Aabb],
    result: &mut Vec<usize>,
) {
    if !node.aabb().intersects_sphere(center, radius) {
        return;
    }
    match node {
        BvhNode::Leaf {
            gaussian_indices, ..
        } => {
            // Per-primitive test to avoid false positives from the node envelope.
            for &idx in gaussian_indices {
                if let Some(prim_aabb) = primitive_aabbs.get(idx) {
                    if prim_aabb.intersects_sphere(center, radius) {
                        result.push(idx);
                    }
                }
            }
        }
        BvhNode::Internal { left, right, .. } => {
            sphere_query_recursive(left, center, radius, primitive_aabbs, result);
            sphere_query_recursive(right, center, radius, primitive_aabbs, result);
        }
    }
}

fn ray_query_recursive(
    node: &BvhNode,
    origin: [f32; 3],
    inv_dir: [f32; 3],
    primitive_aabbs: &[Aabb],
    result: &mut Vec<usize>,
) {
    if !node.aabb().intersects_ray(origin, inv_dir) {
        return;
    }
    match node {
        BvhNode::Leaf {
            gaussian_indices, ..
        } => {
            // Per-primitive test to avoid false positives from the node envelope.
            for &idx in gaussian_indices {
                if let Some(prim_aabb) = primitive_aabbs.get(idx) {
                    if prim_aabb.intersects_ray(origin, inv_dir) {
                        result.push(idx);
                    }
                }
            }
        }
        BvhNode::Internal { left, right, .. } => {
            ray_query_recursive(left, origin, inv_dir, primitive_aabbs, result);
            ray_query_recursive(right, origin, inv_dir, primitive_aabbs, result);
        }
    }
}

// ─── Frustum plane extraction ─────────────────────────────────────────────────

/// Extract 6 view-frustum planes from a column-major 4×4 view-projection matrix.
///
/// Uses the Gribb-Hartmann method. Each plane `[a, b, c, d]` defines the
/// inside half-space as `ax + by + cz + d >= 0`. Planes are normalized so that
/// `(a, b, c)` is a unit normal.
///
/// Column-major layout: `m[col][row]` or equivalently the array is stored as
/// `[col0_row0, col0_row1, col0_row2, col0_row3, col1_row0, ...]`.
pub fn extract_frustum_planes(view_proj: &[[f32; 4]; 4]) -> [[f32; 4]; 6] {
    // view_proj is column-major: view_proj[col][row].
    // Row i of the matrix is: view_proj[0][i], view_proj[1][i], view_proj[2][i], view_proj[3][i].
    let row = |r: usize| -> [f32; 4] {
        [
            view_proj[0][r],
            view_proj[1][r],
            view_proj[2][r],
            view_proj[3][r],
        ]
    };

    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    // Gribb-Hartmann: planes derived from row combinations.
    let left = add_planes(r3, r0); // row3 + row0
    let right = sub_planes(r3, r0); // row3 - row0
    let bottom = add_planes(r3, r1); // row3 + row1
    let top = sub_planes(r3, r1); // row3 - row1
    let near = add_planes(r3, r2); // row3 + row2
    let far = sub_planes(r3, r2); // row3 - row2

    [
        normalize_plane(left),
        normalize_plane(right),
        normalize_plane(bottom),
        normalize_plane(top),
        normalize_plane(near),
        normalize_plane(far),
    ]
}

fn add_planes(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn sub_planes(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

fn normalize_plane(plane: [f32; 4]) -> [f32; 4] {
    let [a, b, c, d] = plane;
    let len = (a * a + b * b + c * c).sqrt();
    if len > 0.0 {
        [a / len, b / len, c / len, d / len]
    } else {
        plane
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Aabb tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_aabb_new() {
        let aabb = Aabb::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(aabb.min, [1.0, 2.0, 3.0]);
        assert_eq!(aabb.max, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_aabb_from_center_radius() {
        let aabb = Aabb::from_center_radius([0.0, 0.0, 0.0], 1.0);
        assert_eq!(aabb.min, [-1.0, -1.0, -1.0]);
        assert_eq!(aabb.max, [1.0, 1.0, 1.0]);

        let aabb2 = Aabb::from_center_radius([1.0, 2.0, 3.0], 0.5);
        assert!((aabb2.min[0] - 0.5).abs() < 1e-6);
        assert!((aabb2.max[2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([-1.0, 0.5, 0.0], [2.0, 1.5, 3.0]);
        let u = a.union(&b);
        assert_eq!(u.min, [-1.0, 0.0, 0.0]);
        assert_eq!(u.max, [2.0, 1.5, 3.0]);
    }

    #[test]
    fn test_aabb_expand() {
        let mut aabb = Aabb::from_point([0.0, 0.0, 0.0]);
        aabb.expand([1.0, -1.0, 2.0]);
        assert_eq!(aabb.min, [0.0, -1.0, 0.0]);
        assert_eq!(aabb.max, [1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_aabb_surface_area() {
        // Unit cube: 6 * 1^2 = 6
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((aabb.surface_area() - 6.0).abs() < 1e-6);

        // 2x3x4 box: 2*(2*3 + 3*4 + 4*2) = 2*(6+12+8) = 52
        let aabb2 = Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        assert!((aabb2.surface_area() - 52.0).abs() < 1e-6);

        // Degenerate: flat square 2x3x0
        let aabb3 = Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 0.0]);
        assert!((aabb3.surface_area() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_aabb_longest_axis() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [3.0, 1.0, 2.0]);
        assert_eq!(aabb.longest_axis(), 0);

        let aabb2 = Aabb::new([0.0, 0.0, 0.0], [1.0, 5.0, 2.0]);
        assert_eq!(aabb2.longest_axis(), 1);

        let aabb3 = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 4.0]);
        assert_eq!(aabb3.longest_axis(), 2);
    }

    #[test]
    fn test_aabb_contains_point() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(aabb.contains_point([0.5, 0.5, 0.5]));
        assert!(aabb.contains_point([0.0, 0.0, 0.0]));
        assert!(aabb.contains_point([1.0, 1.0, 1.0]));
        assert!(!aabb.contains_point([1.1, 0.5, 0.5]));
        assert!(!aabb.contains_point([-0.1, 0.5, 0.5]));
    }

    #[test]
    fn test_aabb_intersects_sphere_inside() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        // Sphere centered inside the box always intersects.
        assert!(aabb.intersects_sphere([1.0, 1.0, 1.0], 0.1));
        // Sphere centered outside but close enough.
        assert!(aabb.intersects_sphere([3.0, 1.0, 1.0], 1.5));
    }

    #[test]
    fn test_aabb_intersects_sphere_outside() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Sphere far away.
        assert!(!aabb.intersects_sphere([5.0, 0.0, 0.0], 1.0));
        // Sphere just touching the corner — distance = sqrt(3) ≈ 1.732.
        assert!(!aabb.intersects_sphere([3.0, 3.0, 3.0], 1.0));
        // Sphere touching at exactly the corner point (2,2,2) distance = sqrt(3).
        assert!(aabb.intersects_sphere([2.0, 2.0, 2.0], 2.0));
    }

    #[test]
    fn test_aabb_intersects_ray() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Ray along x axis through the center.
        let inv_dir = [1.0, f32::INFINITY, f32::INFINITY];
        assert!(aabb.intersects_ray([-1.0, 0.5, 0.5], inv_dir));
        // Ray from inside.
        assert!(aabb.intersects_ray([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_aabb_ray_miss() {
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Ray parallel to x, missing in y.
        let inv_dir = [1.0, f32::INFINITY, f32::INFINITY];
        assert!(!aabb.intersects_ray([-1.0, 2.0, 0.5], inv_dir));
        // Ray going the wrong direction (origin far past the box).
        assert!(!aabb.intersects_ray([5.0, 0.5, 0.5], [1.0, 0.0, 0.0]));
    }

    // ── GaussianBvh tests ─────────────────────────────────────────────────────

    #[test]
    fn test_bvh_build_empty() {
        let bvh = GaussianBvh::build(&[], &[]);
        assert_eq!(bvh.num_gaussians(), 0);
        assert_eq!(bvh.num_nodes(), 0);
        assert_eq!(bvh.depth(), 0);
    }

    #[test]
    fn test_bvh_build_single() {
        let positions = [[0.0f32, 0.0, 0.0]];
        let scales = [[0.0f32, 0.0, 0.0]]; // exp(0) = 1.0
        let bvh = GaussianBvh::build(&positions, &scales);
        assert_eq!(bvh.num_gaussians(), 1);
        assert_eq!(bvh.num_nodes(), 1);
        assert_eq!(bvh.depth(), 1);

        // Sphere query should find it.
        let hits = bvh.sphere_query([0.0, 0.0, 0.0], 2.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], 0);
    }

    #[test]
    fn test_bvh_build_many() {
        // 100 Gaussians spread out in a grid.
        let n = 100usize;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [i as f32, (i % 10) as f32, (i / 10) as f32])
            .collect();
        let scales: Vec<[f32; 3]> = vec![[0.0f32, 0.0, 0.0]; n]; // scale = 1.0
        let bvh = GaussianBvh::build(&positions, &scales);
        assert_eq!(bvh.num_gaussians(), n);
        assert!(bvh.num_nodes() > 1);

        // Big sphere should contain all of them.
        let hits = bvh.sphere_query([50.0, 5.0, 5.0], 200.0);
        assert_eq!(hits.len(), n);
    }

    #[test]
    fn test_bvh_depth_bounded() {
        // Even with many Gaussians the depth must not exceed MAX_DEPTH + 1.
        let n = 1024usize;
        let positions: Vec<[f32; 3]> = (0..n).map(|i| [i as f32 * 0.001, 0.0, 0.0]).collect();
        let scales: Vec<[f32; 3]> = vec![[-5.0f32, -5.0, -5.0]; n]; // tiny scale
        let bvh = GaussianBvh::build(&positions, &scales);
        assert!(bvh.depth() <= MAX_DEPTH + 1);
    }

    // ── Frustum cull tests ────────────────────────────────────────────────────

    /// Build a simple orthographic frustum in the cube [-1,1]^3.
    fn unit_cube_planes() -> [[f32; 4]; 6] {
        // Left (x >= -1): a=1, d=1 → plane [1,0,0,1]
        // Right (x <=  1): a=-1, d=1 → plane [-1,0,0,1]
        // Bottom (y >= -1): [0,1,0,1]
        // Top (y <=  1): [0,-1,0,1]
        // Near (z >= -1): [0,0,1,1]
        // Far (z <=  1): [0,0,-1,1]
        [
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    }

    #[test]
    fn test_frustum_cull_all_inside() {
        let n = 10usize;
        // All Gaussians inside [-0.5, 0.5]^3.
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32 - 0.45;
                [t, t * 0.5, t * 0.3]
            })
            .collect();
        let scales: Vec<[f32; 3]> = vec![[-4.0f32, -4.0, -4.0]; n]; // scale ≈ 0.018
        let bvh = GaussianBvh::build(&positions, &scales);
        let planes = unit_cube_planes();
        let hits = bvh.frustum_cull(&planes);
        // All should be inside.
        assert_eq!(hits.len(), n);
    }

    #[test]
    fn test_frustum_cull_some_outside() {
        // One Gaussian inside, one far outside.
        let positions = [[0.0f32, 0.0, 0.0], [100.0, 100.0, 100.0]];
        let scales = [[-4.0f32, -4.0, -4.0], [-4.0, -4.0, -4.0]];
        let bvh = GaussianBvh::build(&positions, &scales);
        let planes = unit_cube_planes();
        let hits = bvh.frustum_cull(&planes);
        assert!(hits.contains(&0));
        assert!(!hits.contains(&1));
    }

    // ── Sphere query tests ────────────────────────────────────────────────────

    #[test]
    fn test_sphere_query() {
        let positions = [[0.0f32, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]];
        let scales = [[-4.0f32, -4.0, -4.0]; 3];
        let bvh = GaussianBvh::build(&positions, &scales);

        // Small sphere only around the first Gaussian.
        let hits = bvh.sphere_query([0.0, 0.0, 0.0], 1.0);
        assert!(hits.contains(&0));
        assert!(!hits.contains(&2));

        // Large sphere covers all three.
        let hits_all = bvh.sphere_query([10.0, 0.0, 0.0], 15.0);
        assert_eq!(hits_all.len(), 3);
    }

    // ── Ray query tests ───────────────────────────────────────────────────────

    #[test]
    fn test_ray_query() {
        // Three Gaussians along the x axis.
        let positions = [
            [1.0f32, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 10.0, 0.0], // off-axis
        ];
        let scales = [[-1.0f32, -1.0, -1.0]; 3]; // scale ≈ 0.368
        let bvh = GaussianBvh::build(&positions, &scales);

        // Ray along x axis: should hit indices 0 and 1.
        let hits = bvh.ray_query([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(hits.contains(&0));
        assert!(hits.contains(&1));
        // Index 2 is off-axis; its AABB doesn't intersect the x-axis ray.
        assert!(!hits.contains(&2));
    }
}
