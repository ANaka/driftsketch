# Reflections on "Neural Vectorization and Stroke Dynamics" Report

Report: `docs/research/Image-to-Vector Lineart for Plotting.pdf`

## The 2025 Landscape (Report Summary)

The field has split into three lanes:

1. **VLMs generating SVG as code** (OmniSVG, StarVector) -- semantically rich but topologically naive. Great SVGs for screens, bad for plotters because stroke order and continuity are afterthoughts.

2. **Stroke-level reconstruction** (LineDrawer, Single-Line Drawing Vectorization, B-spline abstraction) -- models the *drawing process*, not just the output image. These produce plotter-ready paths with centerline topology, stroke continuity, and mechanical efficiency. LineDrawer's three-stage hierarchy (mind/hand/eye) is the standout.

3. **Hybrid pipelines** (Flux + skeletonization + vpype) -- the pragmatic community standard. Generate a clean raster with a diffusion model, skeletonize it to centerlines, prune the graph, then post-process with vpype for TSP path sorting and line merging. Works today, but semantically blind.

Key enabling tech: **Bezier Splatting** (NeurIPS 2025) replacing DiffVG with 150x faster backward pass for differentiable vector rendering. This unlocks real-time optimization of Bezier curves against pixel-space targets.

## How This Maps to DriftSketch

Our evolution plan (Phase 1-3 in `docs/plans/2026-02-07-image-to-vector-sketch-evolution.md`) is building toward the **stroke reconstruction** camp, which is where the report says the most plotter-relevant work is happening.

### Already aligned

- **Bezier strokes as native representation.** The shift from polylines to parametric curves (Bezier, B-splines) is a consensus direction. Phase 1's move from `(B, 64, 2)` polylines to `(B, 32, 8)` cubic Beziers aligns with LineDrawer, the B-spline abstraction work, and Bezier Splatting.

- **CLIP conditioning (Phase 2) occupies a unique niche.** Neither the pure VLM approach (OmniSVG: generates SVG tokens but with bad topology) nor the pure reconstruction approach (LineDrawer: great topology but requires a raster input pipeline) does what we're planning. DriftSketch as image-conditioned CFM over Bezier control points is a more direct path: image in, plotter-ready strokes out, learned end-to-end. Skips the fragile middle steps of the hybrid pipeline (skeletonization, graph pruning).

- **Phase 3 (pixel-space loss via easydiffvg) is validated by the Bezier Splatting work.** The report highlights DiffVG was too slow for practical training loops. Our plan uses easydiffvg (pure PyTorch reimplementation). Bezier Splatting could be a faster drop-in if we hit training speed issues.

### Gaps to address (future phases)

- **Stroke ordering / TSP.** The report emphasizes that even good stroke geometry needs post-processing for mechanical efficiency (vpype's `linesort`, `linemerge`). Our CFM generates 32 strokes but their order is arbitrary. For plotter output, we'd eventually want either a learned ordering or a vpype post-processing step.

- **Stroke continuity.** LineDrawer's "habitual function" merges sub-strokes into longer coherent paths, reducing pen-up/pen-down operations. Our fixed 32-stroke representation can't express variable numbers of strokes or stroke merging.

- **Curvature smoothness.** The B-spline work adds jerk/snap minimization to the loss. Plotters physically hate sharp curvature changes (servo vibration). A curvature penalty on generated Beziers could be a cheap Phase 3 add-on alongside the pixel loss.

- **Variable stroke width.** LineDrawer's Stage 3 extracts variable-width centerlines for expressive pressure sensitivity. Our model outputs uniform-width strokes. Not critical for basic plotting but matters for fountain pen / brush pen work.

## DriftSketch's Niche

DriftSketch is carving out a position that doesn't quite exist yet -- a *generative* model that natively outputs plotter-friendly Bezier strokes conditioned on images, learned end-to-end. The VLMs can't do topology, the reconstruction methods need a raster pipeline, and the hybrid workflows are glued together with heuristics. If the CFM approach learns good stroke structure directly, it could be a cleaner path than any of the three camps.

## Key Papers to Track

- **OmniSVG** (NeurIPS 2025) -- VLM SVG generation, MMSVG-2M dataset, MMSVGBench
- **LineDrawer** (Computers & Graphics, Aug 2025) -- hierarchical stroke reconstruction (mind/hand/eye)
- **Bezier Splatting** (NeurIPS 2025) -- 150x faster differentiable vector rendering
- **Single-Line Drawing Vectorization** (Pacific Graphics 2025) -- neural intersection classification for continuous paths
- **B-spline Neural Image Abstraction** (Berio et al., 2025) -- curvature-minimizing differentiable splines
- **StarVector** (HuggingFace) -- StarCoder + ViT for structured SVG (icons, diagrams)
- **vpype** -- industry-standard post-processor for mechanical optimization (linemerge, linesort, reloop)
