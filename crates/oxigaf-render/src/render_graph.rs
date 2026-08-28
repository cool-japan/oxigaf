//! Render graph: a CPU-side DAG that describes render pass order, dependencies, and resources.
//!
//! A render graph is a Directed Acyclic Graph (DAG) where nodes are render passes and edges are
//! data-flow dependencies (pass B depends on pass A when B reads a resource written by A).
//! Compilation performs topological sorting via Kahn's algorithm and validates resource lifetimes.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when building or compiling a render graph.
#[derive(Debug, Error)]
pub enum RenderGraphError {
    /// A pass with this name already exists.
    #[error("Pass '{0}' already exists in the graph")]
    DuplicatePass(String),

    /// A resource with this name already exists.
    #[error("Resource '{0}' already exists")]
    DuplicateResource(String),

    /// A referenced pass was not found.
    #[error("Pass '{0}' not found")]
    PassNotFound(String),

    /// A referenced resource was not found.
    #[error("Resource '{0}' not found")]
    ResourceNotFound(String),

    /// A cycle was detected during topological sort.
    #[error("Cycle detected in render graph involving pass '{0}'")]
    CycleDetected(String),

    /// A pass reads a resource that is not produced by any prior pass it depends on.
    #[error("Pass '{dep}' depends on '{pass}' but '{pass}' does not produce resource '{res}'")]
    ResourceDependencyMismatch {
        /// Name of the pass that produces the resource.
        pass: String,
        /// Name of the dependent pass.
        dep: String,
        /// Name of the resource.
        res: String,
    },

    /// A resource is listed in both reads and writes for the same pass.
    #[error("Resource '{0}' declared as both read and write in pass '{1}'")]
    ReadWriteConflict(String, String),

    /// Two different passes both write the same resource. `PassDesc.writes`
    /// is documented as an exclusive write, so a second writer is rejected
    /// rather than silently overriding the first in the writer index (which
    /// would point every dependency edge and `ResourceLifetime.written_by`
    /// at the wrong producer).
    #[error("Resource '{res}' is written by both pass '{first}' and pass '{second}'")]
    DuplicateWrite {
        /// Name of the resource written twice.
        res: String,
        /// Name of the pass that wrote it first (execution/registration order).
        first: String,
        /// Name of the pass that wrote it again.
        second: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Resource types
// ─────────────────────────────────────────────────────────────────────────────

/// Pixel format for a render graph resource (texture / buffer).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceFormat {
    /// 8-bit RGBA (4 bytes / pixel).
    Rgba8,
    /// 16-bit half-float RGBA (8 bytes / pixel).
    Rgba16f,
    /// 32-bit float RGBA (16 bytes / pixel).
    Rgba32f,
    /// 32-bit depth (4 bytes / pixel).
    Depth32f,
    /// Single 32-bit float channel (4 bytes / pixel).
    R32f,
    /// Two 16-bit half-float channels (4 bytes / pixel).
    Rg16f,
}

impl ResourceFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            ResourceFormat::Rgba8 => 4,
            ResourceFormat::Rgba16f => 8,
            ResourceFormat::Rgba32f => 16,
            ResourceFormat::Depth32f => 4,
            ResourceFormat::R32f => 4,
            ResourceFormat::Rg16f => 4,
        }
    }
}

/// Description of a resource (texture or buffer) used in the render graph.
#[derive(Debug, Clone)]
pub struct ResourceDesc {
    /// Unique name for this resource.
    pub name: String,
    /// Pixel/element format.
    pub format: ResourceFormat,
    /// Width in pixels / elements.
    pub width: u32,
    /// Height in pixels (1 for 1D buffers).
    pub height: u32,
}

impl ResourceDesc {
    /// Create a new resource description.
    pub fn new(name: impl Into<String>, format: ResourceFormat, width: u32, height: u32) -> Self {
        Self {
            name: name.into(),
            format,
            width,
            height,
        }
    }

    /// Total size in bytes: `width * height * bytes_per_pixel`.
    pub fn size_bytes(&self) -> usize {
        self.width as usize * self.height as usize * self.format.bytes_per_pixel()
    }
}

/// Stable, name-based identifier for a resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceId(String);

impl ResourceId {
    /// Create a new resource ID from a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The name this ID refers to.
    pub fn name(&self) -> &str {
        &self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass description
// ─────────────────────────────────────────────────────────────────────────────

/// Category of work performed by a render pass.
#[derive(Debug, Clone, PartialEq)]
pub enum PassType {
    /// GPU compute shader dispatch.
    Compute,
    /// Rasterization pipeline.
    Rasterize,
    /// Buffer/texture copy or upload.
    Transfer,
    /// Swapchain presentation.
    Present,
}

/// Description of a single render pass within the graph.
#[derive(Debug, Clone)]
pub struct PassDesc {
    /// Unique name for this pass.
    pub name: String,
    /// Category of the pass.
    pub pass_type: PassType,
    /// Resources this pass reads (must have been written by an earlier pass or be external).
    pub reads: Vec<ResourceId>,
    /// Resources this pass exclusively writes.
    pub writes: Vec<ResourceId>,
    /// Estimated GPU execution time in microseconds (scheduling hint).
    pub estimated_gpu_us: u64,
}

impl PassDesc {
    /// Create a new pass description with empty read/write lists.
    pub fn new(name: impl Into<String>, pass_type: PassType) -> Self {
        Self {
            name: name.into(),
            pass_type,
            reads: Vec::new(),
            writes: Vec::new(),
            estimated_gpu_us: 0,
        }
    }

    /// Builder: add a resource to the read list.
    pub fn read(mut self, resource: impl Into<String>) -> Self {
        self.reads.push(ResourceId::new(resource));
        self
    }

    /// Builder: add a resource to the write list.
    pub fn write(mut self, resource: impl Into<String>) -> Self {
        self.writes.push(ResourceId::new(resource));
        self
    }

    /// Builder: set the estimated GPU time.
    pub fn with_gpu_time(mut self, us: u64) -> Self {
        self.estimated_gpu_us = us;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compiled render graph
// ─────────────────────────────────────────────────────────────────────────────

/// Lifetime information for a single resource across the render graph.
#[derive(Debug, Clone)]
pub struct ResourceLifetime {
    /// The resource this entry describes.
    pub resource_id: ResourceId,
    /// Name of the pass that writes this resource (`""` if external / not written in graph).
    pub written_by: String,
    /// Names of passes that read this resource (in execution order).
    pub read_by: Vec<String>,
    /// Whether this resource's memory can be aliased after all readers have finished.
    ///
    /// `true` when no pass after the last reader re-writes the resource, so the
    /// backing allocation can be reused once the resource's last reader completes.
    pub can_alias: bool,
}

/// The result of successfully compiling a [`RenderGraph`].
#[derive(Debug)]
pub struct CompiledRenderGraph {
    /// Pass names in topologically sorted (dependency-respecting) execution order.
    pub execution_order: Vec<String>,
    /// Per-resource lifetime information.
    pub resource_lifetimes: Vec<ResourceLifetime>,
    /// Conservative peak memory estimate (sum of all resource sizes in bytes).
    pub peak_memory_bytes: usize,
    /// Total number of passes.
    pub num_passes: usize,
}

impl CompiledRenderGraph {
    /// Group passes into parallel batches.
    ///
    /// Passes in the same batch have no data-flow dependencies between them and
    /// can therefore be executed concurrently.  The batches are ordered so that
    /// all dependencies of batch *k* are satisfied by batches 0 … k-1.
    ///
    /// The algorithm computes the "level" of each pass (0 for passes with no
    /// predecessors, `max(predecessor_level) + 1` otherwise) and groups passes
    /// by level, preserving the `execution_order` within each group.
    pub fn parallel_passes(&self, graph: &RenderGraph) -> Vec<Vec<String>> {
        // Build a map from pass name → index in execution_order for fast look-up.
        let order_index: HashMap<&str, usize> = self
            .execution_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // Build: resource_name → writer_pass_name (from the graph).
        let mut resource_writer: HashMap<&str, &str> = HashMap::new();
        for pass in &graph.passes {
            for w in &pass.writes {
                resource_writer.insert(w.name(), pass.name.as_str());
            }
        }

        // Compute level for each pass in execution order (forward pass is safe because
        // execution_order is already topologically sorted).
        let mut level: HashMap<&str, usize> = HashMap::new();

        for pass_name in &self.execution_order {
            let pass = match graph.pass(pass_name) {
                Some(p) => p,
                None => {
                    level.insert(pass_name.as_str(), 0);
                    continue;
                }
            };

            let max_pred_level = pass
                .reads
                .iter()
                .filter_map(|r| resource_writer.get(r.name()))
                .filter_map(|writer| level.get(writer))
                .copied()
                .max()
                .unwrap_or(0);

            let this_level = if pass
                .reads
                .iter()
                .any(|r| resource_writer.contains_key(r.name()))
            {
                max_pred_level + 1
            } else {
                0
            };

            level.insert(pass_name.as_str(), this_level);
        }

        // Determine the maximum level.
        let max_level = level.values().copied().max().unwrap_or(0);

        // Group passes by level, preserving execution_order within each group.
        let mut batches: Vec<Vec<String>> = vec![Vec::new(); max_level + 1];
        for pass_name in &self.execution_order {
            let lvl = level.get(pass_name.as_str()).copied().unwrap_or(0);
            batches[lvl].push(pass_name.clone());
        }

        // Remove empty batches (can occur if levels are not contiguous, which shouldn't
        // happen with Kahn's sort but is defensive).
        batches.retain(|b| !b.is_empty());

        // Sort within each batch by execution_order index for determinism.
        for batch in &mut batches {
            batch.sort_by_key(|name| {
                order_index
                    .get(name.as_str())
                    .copied()
                    .unwrap_or(usize::MAX)
            });
        }

        batches
    }

    /// A human-readable summary of the compiled execution plan.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Compiled Render Graph ===\n");
        out.push_str(&format!("Passes: {}\n", self.num_passes));
        out.push_str(&format!(
            "Peak memory: {} bytes ({:.2} MiB)\n",
            self.peak_memory_bytes,
            self.peak_memory_bytes as f64 / (1024.0 * 1024.0)
        ));
        out.push_str("\nExecution order:\n");
        for (i, name) in self.execution_order.iter().enumerate() {
            out.push_str(&format!("  {:2}. {}\n", i + 1, name));
        }
        out.push_str("\nResource lifetimes:\n");
        for lt in &self.resource_lifetimes {
            let writer = if lt.written_by.is_empty() {
                "<external>".to_string()
            } else {
                lt.written_by.clone()
            };
            out.push_str(&format!(
                "  {} | writer: {} | readers: [{}] | alias: {}\n",
                lt.resource_id.name(),
                writer,
                lt.read_by.join(", "),
                lt.can_alias
            ));
        }
        out
    }

    /// All passes (by name) that read the given resource, in execution order.
    pub fn passes_reading(&self, resource: &ResourceId) -> Vec<&str> {
        self.resource_lifetimes
            .iter()
            .find(|lt| &lt.resource_id == resource)
            .map(|lt| lt.read_by.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render graph
// ─────────────────────────────────────────────────────────────────────────────

/// A Directed Acyclic Graph (DAG) describing render passes and the resources that flow between them.
///
/// Build the graph by calling [`add_resource`] and [`add_pass`], then call [`compile`] to
/// validate, topologically sort, and compute resource lifetimes.
///
/// [`add_resource`]: RenderGraph::add_resource
/// [`add_pass`]: RenderGraph::add_pass
/// [`compile`]: RenderGraph::compile
pub struct RenderGraph {
    passes: Vec<PassDesc>,
    resources: Vec<ResourceDesc>,
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    /// Create an empty render graph.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
        }
    }

    // ── Resource management ──────────────────────────────────────────────────

    /// Add a resource description to the graph.
    ///
    /// Returns the stable [`ResourceId`] for the resource, or
    /// [`RenderGraphError::DuplicateResource`] if a resource with that name already exists.
    pub fn add_resource(&mut self, desc: ResourceDesc) -> Result<ResourceId, RenderGraphError> {
        if self.resources.iter().any(|r| r.name == desc.name) {
            return Err(RenderGraphError::DuplicateResource(desc.name));
        }
        let id = ResourceId::new(&desc.name);
        self.resources.push(desc);
        Ok(id)
    }

    /// Look up a resource by its [`ResourceId`].
    pub fn resource(&self, id: &ResourceId) -> Option<&ResourceDesc> {
        self.resources.iter().find(|r| r.name == id.0)
    }

    // ── Pass management ──────────────────────────────────────────────────────

    /// Add a render pass to the graph.
    ///
    /// Returns [`RenderGraphError::DuplicatePass`] if a pass with that name already exists.
    pub fn add_pass(&mut self, desc: PassDesc) -> Result<(), RenderGraphError> {
        if self.passes.iter().any(|p| p.name == desc.name) {
            return Err(RenderGraphError::DuplicatePass(desc.name));
        }
        self.passes.push(desc);
        Ok(())
    }

    /// Look up a pass by name.
    pub fn pass(&self, name: &str) -> Option<&PassDesc> {
        self.passes.iter().find(|p| p.name == name)
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Sum of `estimated_gpu_us` across all passes.
    pub fn estimated_gpu_time_us(&self) -> u64 {
        self.passes.iter().map(|p| p.estimated_gpu_us).sum()
    }

    /// Number of passes registered in the graph.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }

    /// Number of resources registered in the graph.
    pub fn num_resources(&self) -> usize {
        self.resources.len()
    }

    // ── Compilation ──────────────────────────────────────────────────────────

    /// Compile the render graph.
    ///
    /// Steps:
    /// 1. Validate that no pass lists the same resource in both reads and writes.
    /// 2. Build a dependency DAG (B → A if B reads a resource written by A).
    /// 3. Topological sort via Kahn's algorithm.
    /// 4. Detect cycles (remaining nodes after Kahn's).
    /// 5. Compute resource lifetimes and memory estimate.
    pub fn compile(&self) -> Result<CompiledRenderGraph, RenderGraphError> {
        let n = self.passes.len();

        // ── 1. Validate read/write conflicts ────────────────────────────────
        for pass in &self.passes {
            for w in &pass.writes {
                if pass.reads.contains(w) {
                    return Err(RenderGraphError::ReadWriteConflict(
                        w.name().to_owned(),
                        pass.name.clone(),
                    ));
                }
            }
        }

        // ── 2. Build resource → writer index map ────────────────────────────
        // Maps resource name → index of the pass that writes it. Each
        // resource must have exactly one writer (see `PassDesc::writes`'s
        // docs): silently letting a second writer overwrite the map entry
        // would point every dependency edge and the resource's eventual
        // `ResourceLifetime.written_by` at the wrong producer.
        let mut resource_writer: HashMap<&str, usize> = HashMap::new();
        for (idx, pass) in self.passes.iter().enumerate() {
            for w in &pass.writes {
                if let Some(&first_idx) = resource_writer.get(w.name()) {
                    return Err(RenderGraphError::DuplicateWrite {
                        res: w.name().to_owned(),
                        first: self.passes[first_idx].name.clone(),
                        second: pass.name.clone(),
                    });
                }
                resource_writer.insert(w.name(), idx);
            }
        }

        // ── 3. Build adjacency list and in-degree counts ────────────────────
        // Edge A → B means "B must execute after A" (B reads something A writes).
        // adj[a] = list of passes that depend on a.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];

        for (b_idx, pass_b) in self.passes.iter().enumerate() {
            for r in &pass_b.reads {
                if let Some(&a_idx) = resource_writer.get(r.name()) {
                    if a_idx == b_idx {
                        // A pass writing and "reading" the same resource via different paths
                        // is caught by the read/write conflict check above; skip self-loops.
                        continue;
                    }
                    // Avoid duplicate edges.
                    if !adj[a_idx].contains(&b_idx) {
                        adj[a_idx].push(b_idx);
                        in_degree[b_idx] += 1;
                    }
                }
                // If no pass writes this resource it is treated as external → no edge needed.
            }
        }

        // ── 4. Kahn's topological sort ───────────────────────────────────────
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate().take(n) {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut execution_order: Vec<String> = Vec::with_capacity(n);

        while let Some(node) = queue.pop_front() {
            execution_order.push(self.passes[node].name.clone());
            for &succ in &adj[node] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        // ── 5. Cycle detection ──────────────────────────────────────────────
        if execution_order.len() < n {
            // Find the first pass not in the sorted output.
            let sorted_set: std::collections::HashSet<&str> =
                execution_order.iter().map(|s| s.as_str()).collect();
            let cycle_pass = self
                .passes
                .iter()
                .find(|p| !sorted_set.contains(p.name.as_str()))
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(RenderGraphError::CycleDetected(cycle_pass));
        }

        // ── 6. Resource lifetimes ───────────────────────────────────────────
        // Build an index from pass name → position in execution_order.
        let exec_pos: HashMap<&str, usize> = execution_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        let mut resource_lifetimes: Vec<ResourceLifetime> = Vec::new();

        for res in &self.resources {
            let writer_idx = resource_writer.get(res.name.as_str()).copied();
            let written_by = writer_idx
                .map(|i| self.passes[i].name.clone())
                .unwrap_or_default();

            // Collect readers in execution order.
            let mut read_by: Vec<String> = Vec::new();
            for pass_name in &execution_order {
                let pass = match self.pass(pass_name) {
                    Some(p) => p,
                    None => continue,
                };
                if pass.reads.iter().any(|r| r.name() == res.name) {
                    read_by.push(pass_name.clone());
                }
            }

            // can_alias: true when no pass after the last reader re-writes this resource.
            // Since each resource has at most one writer in this model, and the writer
            // precedes all readers (validated by topological sort), the resource memory
            // can always be aliased after the last reader completes.
            let last_reader_pos = read_by
                .iter()
                .filter_map(|name| exec_pos.get(name.as_str()).copied())
                .max();

            let can_alias = match (writer_idx, last_reader_pos) {
                (Some(_w_idx), Some(last_pos)) => {
                    // Check whether any pass after last_pos writes to this resource again.
                    // In our model each resource has a single writer, so this is always false
                    // (the writer is before last_pos), making can_alias = true.
                    let writer_pos = exec_pos
                        .get(written_by.as_str())
                        .copied()
                        .unwrap_or(usize::MAX);
                    writer_pos <= last_pos
                }
                // External resource (no writer) or no readers → conservatively true.
                _ => true,
            };

            resource_lifetimes.push(ResourceLifetime {
                resource_id: ResourceId::new(&res.name),
                written_by,
                read_by,
                can_alias,
            });
        }

        // ── 7. Peak memory estimate ─────────────────────────────────────────
        let peak_memory_bytes: usize = self.resources.iter().map(|r| r.size_bytes()).sum();

        Ok(CompiledRenderGraph {
            execution_order,
            resource_lifetimes,
            peak_memory_bytes,
            num_passes: n,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard 3DGS render graph
// ─────────────────────────────────────────────────────────────────────────────

/// Build an illustrative 3D Gaussian Splatting render graph for avatar rendering.
///
/// This is a simplified, five-pass sketch of the data flow in a 3DGS
/// forward+backward pass, intended as a worked example of the
/// [`RenderGraph`] API (compilation, topological ordering, resource
/// lifetimes) -- it is **not** a literal description of the real GPU
/// pipeline. The actual rasterizer builds its pipelines via
/// `pipeline::RasterPipelines`, which creates substantially more than five
/// GPU pipelines, and nothing in this crate currently drives rendering
/// through a [`CompiledRenderGraph`]. Treat the pass/resource names below as
/// representative, not authoritative.
///
/// The graph contains five passes:
///
/// | # | Name            | Type      | Writes                                    | Reads                                     |
/// |---|-----------------|-----------|-------------------------------------------|-------------------------------------------|
/// | 1 | preprocess      | Compute   | tile_counts, keys, values                 | —                                         |
/// | 2 | sort            | Compute   | sorted_keys, sorted_values, tile_ranges   | tile_counts, keys, values                 |
/// | 3 | rasterize_fwd   | Rasterize | color, depth, transmittance               | sorted_keys, sorted_values, tile_ranges   |
/// | 4 | rasterize_bwd   | Compute   | grad_positions, grad_scales, grad_colors  | color, depth, transmittance               |
/// | 5 | output          | Present   | —                                         | color                                     |
///
/// `tile_counts`, `keys`, `values`, `sorted_keys`, `sorted_values` and the
/// gradient buffers are inherently per-Gaussian (not per-pixel) data, so
/// they are sized by `num_gaussians` rather than `width * height`; only the
/// framebuffer-shaped resources (`color`, `depth`, `transmittance`,
/// `tile_ranges`) are sized by the image dimensions.
///
/// # Errors
///
/// Returns an error if two of the hardcoded resource or pass names below
/// were to collide (this would indicate a bug in this function itself,
/// since every name here is a distinct literal).
pub fn build_standard_3dgs_graph(
    width: u32,
    height: u32,
    num_gaussians: u32,
) -> Result<RenderGraph, RenderGraphError> {
    let mut g = RenderGraph::new();
    let n = num_gaussians.max(1);

    // ── Resources ────────────────────────────────────────────────────────────
    // Per-Gaussian buffers — use R32f as a generic "u32 / f32 data" format,
    // sized as a 1-D buffer of `num_gaussians` elements (not width*height).
    g.add_resource(ResourceDesc::new("tile_counts", ResourceFormat::R32f, n, 1))?;
    g.add_resource(ResourceDesc::new("keys", ResourceFormat::R32f, n, 1))?;
    g.add_resource(ResourceDesc::new("values", ResourceFormat::R32f, n, 1))?;
    g.add_resource(ResourceDesc::new("sorted_keys", ResourceFormat::R32f, n, 1))?;
    g.add_resource(ResourceDesc::new(
        "sorted_values",
        ResourceFormat::R32f,
        n,
        1,
    ))?;

    // Per-tile buffer: sized by the image dimensions (a stand-in for the
    // real tile grid, which would additionally depend on tile size).
    g.add_resource(ResourceDesc::new(
        "tile_ranges",
        ResourceFormat::Rg16f,
        width,
        height,
    ))?;

    // Output textures — genuinely per-pixel.
    g.add_resource(ResourceDesc::new(
        "color",
        ResourceFormat::Rgba8,
        width,
        height,
    ))?;
    g.add_resource(ResourceDesc::new(
        "depth",
        ResourceFormat::Depth32f,
        width,
        height,
    ))?;
    g.add_resource(ResourceDesc::new(
        "transmittance",
        ResourceFormat::R32f,
        width,
        height,
    ))?;

    // Gradient outputs — per-Gaussian (position/scale/colour gradients),
    // not per-pixel.
    g.add_resource(ResourceDesc::new(
        "grad_positions",
        ResourceFormat::Rgba32f,
        n,
        1,
    ))?;
    g.add_resource(ResourceDesc::new(
        "grad_scales",
        ResourceFormat::Rgba32f,
        n,
        1,
    ))?;
    g.add_resource(ResourceDesc::new(
        "grad_colors",
        ResourceFormat::Rgba32f,
        n,
        1,
    ))?;

    // ── Passes ───────────────────────────────────────────────────────────────
    g.add_pass(
        PassDesc::new("preprocess", PassType::Compute)
            .write("tile_counts")
            .write("keys")
            .write("values")
            .with_gpu_time(500),
    )?;
    g.add_pass(
        PassDesc::new("sort", PassType::Compute)
            .read("tile_counts")
            .read("keys")
            .read("values")
            .write("sorted_keys")
            .write("sorted_values")
            .write("tile_ranges")
            .with_gpu_time(2000),
    )?;
    g.add_pass(
        PassDesc::new("rasterize_fwd", PassType::Rasterize)
            .read("sorted_keys")
            .read("sorted_values")
            .read("tile_ranges")
            .write("color")
            .write("depth")
            .write("transmittance")
            .with_gpu_time(3000),
    )?;
    g.add_pass(
        PassDesc::new("rasterize_bwd", PassType::Compute)
            .read("color")
            .read("depth")
            .read("transmittance")
            .write("grad_positions")
            .write("grad_scales")
            .write("grad_colors")
            .with_gpu_time(5000),
    )?;
    g.add_pass(
        PassDesc::new("output", PassType::Present)
            .read("color")
            .with_gpu_time(100),
    )?;

    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Build a simple two-pass chain: pass_a writes buf_a, pass_b reads buf_a and writes buf_b.
    fn simple_graph() -> RenderGraph {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("buf_a", ResourceFormat::Rgba8, 512, 512))
            .unwrap();
        g.add_resource(ResourceDesc::new("buf_b", ResourceFormat::Rgba8, 512, 512))
            .unwrap();
        g.add_pass(PassDesc::new("pass_a", PassType::Compute).write("buf_a"))
            .unwrap();
        g.add_pass(
            PassDesc::new("pass_b", PassType::Compute)
                .read("buf_a")
                .write("buf_b"),
        )
        .unwrap();
        g
    }

    // ── Test 1: new() starts empty ───────────────────────────────────────────
    #[test]
    fn test_new_starts_empty() {
        let g = RenderGraph::new();
        assert_eq!(g.num_passes(), 0);
        assert_eq!(g.num_resources(), 0);
    }

    // ── Test 2: add_resource returns ResourceId ──────────────────────────────
    #[test]
    fn test_add_resource_returns_id() {
        let mut g = RenderGraph::new();
        let id = g
            .add_resource(ResourceDesc::new("buf", ResourceFormat::Rgba8, 64, 64))
            .unwrap();
        assert_eq!(id.name(), "buf");
        assert_eq!(g.num_resources(), 1);
    }

    // ── Test 3: add_resource duplicate → DuplicateResource ──────────────────
    #[test]
    fn test_add_resource_duplicate() {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("buf", ResourceFormat::Rgba8, 64, 64))
            .unwrap();
        let err = g
            .add_resource(ResourceDesc::new("buf", ResourceFormat::Rgba8, 64, 64))
            .unwrap_err();
        assert!(matches!(err, RenderGraphError::DuplicateResource(_)));
    }

    // ── Test 4: add_pass adds pass ───────────────────────────────────────────
    #[test]
    fn test_add_pass() {
        let mut g = RenderGraph::new();
        g.add_pass(PassDesc::new("my_pass", PassType::Compute))
            .unwrap();
        assert_eq!(g.num_passes(), 1);
    }

    // ── Test 5: add_pass duplicate → DuplicatePass ──────────────────────────
    #[test]
    fn test_add_pass_duplicate() {
        let mut g = RenderGraph::new();
        g.add_pass(PassDesc::new("my_pass", PassType::Compute))
            .unwrap();
        let err = g
            .add_pass(PassDesc::new("my_pass", PassType::Compute))
            .unwrap_err();
        assert!(matches!(err, RenderGraphError::DuplicatePass(_)));
    }

    // ── Test 6: compile simple chain → [pass_a, pass_b] ─────────────────────
    #[test]
    fn test_compile_simple_chain() {
        let g = simple_graph();
        let compiled = g.compile().unwrap();
        assert_eq!(compiled.execution_order, vec!["pass_a", "pass_b"]);
    }

    // ── Test 7: compile two independent passes ───────────────────────────────
    #[test]
    fn test_compile_independent_passes() {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("x", ResourceFormat::R32f, 8, 8))
            .unwrap();
        g.add_resource(ResourceDesc::new("y", ResourceFormat::R32f, 8, 8))
            .unwrap();
        g.add_pass(PassDesc::new("pa", PassType::Compute).write("x"))
            .unwrap();
        g.add_pass(PassDesc::new("pb", PassType::Compute).write("y"))
            .unwrap();
        let compiled = g.compile().unwrap();
        // Both passes must appear (order unspecified between them).
        let set: std::collections::HashSet<&str> = compiled
            .execution_order
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(set.contains("pa"));
        assert!(set.contains("pb"));
        assert_eq!(compiled.num_passes, 2);
    }

    // ── Test 8: compile detects cycle ────────────────────────────────────────
    #[test]
    fn test_compile_cycle_detected() {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("x", ResourceFormat::R32f, 8, 8))
            .unwrap();
        g.add_resource(ResourceDesc::new("y", ResourceFormat::R32f, 8, 8))
            .unwrap();
        // A writes x, reads y → depends on B.
        g.add_pass(PassDesc::new("pa", PassType::Compute).write("x").read("y"))
            .unwrap();
        // B writes y, reads x → depends on A.  Cycle: A↔B.
        g.add_pass(PassDesc::new("pb", PassType::Compute).write("y").read("x"))
            .unwrap();
        let err = g.compile().unwrap_err();
        assert!(matches!(err, RenderGraphError::CycleDetected(_)));
    }

    // ── Test 9: compile empty graph ──────────────────────────────────────────
    #[test]
    fn test_compile_empty_graph() {
        let g = RenderGraph::new();
        let compiled = g.compile().unwrap();
        assert!(compiled.execution_order.is_empty());
        assert_eq!(compiled.num_passes, 0);
        assert_eq!(compiled.peak_memory_bytes, 0);
    }

    // ── Test 10: ResourceDesc::size_bytes ────────────────────────────────────
    #[test]
    fn test_resource_desc_size_bytes() {
        let desc = ResourceDesc::new("t", ResourceFormat::Rgba8, 512, 512);
        // 512 * 512 * 4 = 1_048_576
        assert_eq!(desc.size_bytes(), 1_048_576);
    }

    // ── Test 11: ResourceFormat::bytes_per_pixel ─────────────────────────────
    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(ResourceFormat::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(ResourceFormat::Rgba16f.bytes_per_pixel(), 8);
        assert_eq!(ResourceFormat::Rgba32f.bytes_per_pixel(), 16);
        assert_eq!(ResourceFormat::Depth32f.bytes_per_pixel(), 4);
        assert_eq!(ResourceFormat::R32f.bytes_per_pixel(), 4);
        assert_eq!(ResourceFormat::Rg16f.bytes_per_pixel(), 4);
    }

    // ── Test 12: estimated_gpu_time_us sums pass estimates ───────────────────
    #[test]
    fn test_estimated_gpu_time_us() {
        let mut g = RenderGraph::new();
        g.add_pass(PassDesc::new("a", PassType::Compute).with_gpu_time(1000))
            .unwrap();
        g.add_pass(PassDesc::new("b", PassType::Compute).with_gpu_time(2500))
            .unwrap();
        assert_eq!(g.estimated_gpu_time_us(), 3500);
    }

    // ── Test 13: PassDesc builder pattern ────────────────────────────────────
    #[test]
    fn test_pass_desc_builder() {
        let p = PassDesc::new("p", PassType::Rasterize)
            .read("tex_in")
            .write("tex_out")
            .with_gpu_time(42);
        assert_eq!(p.reads.len(), 1);
        assert_eq!(p.reads[0].name(), "tex_in");
        assert_eq!(p.writes.len(), 1);
        assert_eq!(p.writes[0].name(), "tex_out");
        assert_eq!(p.estimated_gpu_us, 42);
    }

    // ── Test 14: format_summary is non-empty ─────────────────────────────────
    #[test]
    fn test_format_summary_non_empty() {
        let g = simple_graph();
        let compiled = g.compile().unwrap();
        let summary = compiled.format_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("pass_a"));
        assert!(summary.contains("pass_b"));
    }

    // ── Test 15: parallel_passes — single chain ───────────────────────────────
    #[test]
    fn test_parallel_passes_single_chain() {
        let g = simple_graph();
        let compiled = g.compile().unwrap();
        let batches = compiled.parallel_passes(&g);
        // pass_a has no predecessors → level 0; pass_b reads pass_a's output → level 1.
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec!["pass_a"]);
        assert_eq!(batches[1], vec!["pass_b"]);
    }

    // ── Test 16: parallel_passes — independent passes in same batch ───────────
    #[test]
    fn test_parallel_passes_independent_same_batch() {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("x", ResourceFormat::R32f, 8, 8))
            .unwrap();
        g.add_resource(ResourceDesc::new("y", ResourceFormat::R32f, 8, 8))
            .unwrap();
        g.add_pass(PassDesc::new("pa", PassType::Compute).write("x"))
            .unwrap();
        g.add_pass(PassDesc::new("pb", PassType::Compute).write("y"))
            .unwrap();
        let compiled = g.compile().unwrap();
        let batches = compiled.parallel_passes(&g);
        // Both passes have no predecessors → same batch.
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
    }

    // ── Test 17: build_standard_3dgs_graph compiles without error ─────────────
    #[test]
    fn test_standard_3dgs_graph_compiles() {
        let g = build_standard_3dgs_graph(1920, 1080, 100_000).expect("graph construction");
        let result = g.compile();
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
    }

    // ── Test 18: build_standard_3dgs_graph has 5 passes ──────────────────────
    #[test]
    fn test_standard_3dgs_graph_five_passes() {
        let g = build_standard_3dgs_graph(1920, 1080, 100_000).expect("graph construction");
        assert_eq!(g.num_passes(), 5);
    }

    // ── Test 18b: per-Gaussian resources are sized by num_gaussians ──────────
    #[test]
    fn test_standard_3dgs_graph_per_gaussian_sizing() {
        let g = build_standard_3dgs_graph(1920, 1080, 12_345).expect("graph construction");
        let keys = g
            .resource(&ResourceId::new("keys"))
            .expect("keys resource must exist");
        assert_eq!(
            keys.width, 12_345,
            "per-Gaussian buffers must be sized by num_gaussians"
        );
        assert_eq!(keys.height, 1);

        let color = g
            .resource(&ResourceId::new("color"))
            .expect("color resource must exist");
        assert_eq!(
            color.width, 1920,
            "per-pixel buffers must still be sized by width/height"
        );
        assert_eq!(color.height, 1080);
    }

    // ── Test 18c: duplicate writes are rejected at compile time ──────────────
    #[test]
    fn test_compile_rejects_duplicate_write() {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("shared", ResourceFormat::Rgba8, 64, 64))
            .unwrap();
        g.add_pass(PassDesc::new("first", PassType::Compute).write("shared"))
            .unwrap();
        g.add_pass(PassDesc::new("second", PassType::Compute).write("shared"))
            .unwrap();

        let err = g.compile().unwrap_err();
        match err {
            RenderGraphError::DuplicateWrite { res, first, second } => {
                assert_eq!(res, "shared");
                assert_eq!(first, "first");
                assert_eq!(second, "second");
            }
            other => panic!("expected DuplicateWrite, got {other:?}"),
        }
    }

    // ── Test 19: passes_reading returns correct passes ────────────────────────
    #[test]
    fn test_passes_reading() {
        let g = simple_graph();
        let compiled = g.compile().unwrap();
        let readers = compiled.passes_reading(&ResourceId::new("buf_a"));
        assert_eq!(readers, vec!["pass_b"]);
    }

    // ── Test 20: num_passes and num_resources ─────────────────────────────────
    #[test]
    fn test_num_passes_and_resources() {
        let g = simple_graph();
        assert_eq!(g.num_passes(), 2);
        assert_eq!(g.num_resources(), 2);
    }

    // ── Test 21: resource() and pass() lookups ────────────────────────────────
    #[test]
    fn test_resource_and_pass_lookups() {
        let g = simple_graph();
        let id = ResourceId::new("buf_a");
        let res = g.resource(&id);
        assert!(res.is_some());
        assert_eq!(res.unwrap().name, "buf_a");

        let pass = g.pass("pass_a");
        assert!(pass.is_some());
        assert_eq!(pass.unwrap().name, "pass_a");

        // Non-existent lookups.
        assert!(g.resource(&ResourceId::new("nonexistent")).is_none());
        assert!(g.pass("nonexistent").is_none());
    }

    // ── Test 22: ResourceId equality and hashing ──────────────────────────────
    #[test]
    fn test_resource_id_equality() {
        let a = ResourceId::new("buf");
        let b = ResourceId::new("buf");
        let c = ResourceId::new("other");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── Test 23: peak_memory_bytes is sum of resource sizes ───────────────────
    #[test]
    fn test_peak_memory_bytes() {
        let g = simple_graph();
        let compiled = g.compile().unwrap();
        // 2 × (512 × 512 × 4) = 2_097_152
        assert_eq!(compiled.peak_memory_bytes, 2 * 512 * 512 * 4);
    }

    // ── Test 24: ReadWriteConflict error ──────────────────────────────────────
    #[test]
    fn test_read_write_conflict() {
        let mut g = RenderGraph::new();
        g.add_resource(ResourceDesc::new("buf", ResourceFormat::R32f, 8, 8))
            .unwrap();
        // A pass that both reads and writes the same resource.
        g.add_pass(
            PassDesc::new("bad_pass", PassType::Compute)
                .read("buf")
                .write("buf"),
        )
        .unwrap();
        let err = g.compile().unwrap_err();
        assert!(matches!(err, RenderGraphError::ReadWriteConflict(_, _)));
    }

    // ── Test 25: 3DGS execution order ─────────────────────────────────────────
    #[test]
    fn test_standard_3dgs_execution_order() {
        let g = build_standard_3dgs_graph(800, 600, 50_000).expect("graph construction");
        let compiled = g.compile().unwrap();
        let order = &compiled.execution_order;
        // preprocess must come before sort, sort before rasterize_fwd, etc.
        let pos = |name: &str| order.iter().position(|n| n == name).unwrap();
        assert!(pos("preprocess") < pos("sort"));
        assert!(pos("sort") < pos("rasterize_fwd"));
        assert!(pos("rasterize_fwd") < pos("rasterize_bwd"));
        assert!(pos("rasterize_fwd") < pos("output"));
    }
}
