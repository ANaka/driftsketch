# Vector Sketch Dataset Evaluation for DriftSketch

## Project Requirements

DriftSketch is a Conditional Flow Matching (CFM) model for vector sketch generation. The model requires:

- **Point sequences** of shape `(num_points, 2)` normalized to `[-1, 1]`
- **Integer class labels** for conditioning
- Currently uses synthetic circles and squares (2 classes, 64 points each)
- Training pipeline expects `(points, labels)` tuples from a batch generator

The ideal dataset provides vector/stroke data that can be converted to fixed-length (x, y) point sequences with categorical labels.

---

## 1. Google QuickDraw

**Relevance: 5/5** (Primary recommendation)

### Overview
- **Size:** 50M+ drawings across 345 categories
- **Samples per category:** ~120,000-170,000 (varies)
- **License:** Creative Commons Attribution 4.0 (CC-BY 4.0)
- **Source:** https://github.com/googlecreativelab/quickdraw-dataset

### Data Formats

| Format | File Type | Description |
|--------|-----------|-------------|
| Raw | `.ndjson` | Full strokes with timing: `[[x0,x1,...], [y0,y1,...], [t0,t1,...]]` per stroke |
| Simplified | `.ndjson` | Strokes without timing, scaled to 0-255, RDP-simplified |
| Sketch-RNN | `.npz` | Stroke-3 format: `(dx, dy, pen_state)` as `np.int16` arrays |
| Bitmap | `.npy` | 28x28 grayscale rasters (not useful for vector work) |

### Mapping to DriftSketch Format

**Simplified NDJSON (recommended path):**
1. Each drawing has a `drawing` field: list of strokes, each stroke = `[[x0,x1,...], [y0,y1,...]]`
2. Zip x and y arrays per stroke to get `(x, y)` point tuples
3. Concatenate all strokes (ignoring pen-up transitions, or inserting interpolated points)
4. Resample to fixed `num_points` (e.g., 64 or 128)
5. Normalize from `[0, 255]` to `[-1, 1]`

**Sketch-RNN NPZ (alternative):**
1. Each sample is a variable-length array of `(dx, dy, pen_state)` in stroke-3 format
2. Cumulative sum of `(dx, dy)` gives absolute coordinates
3. Resample to fixed length, normalize to `[-1, 1]`
4. Pre-split into train/valid/test (70K/2.5K/2.5K per category)

### Download Methods
- **Google Cloud Storage:** `gsutil -m cp 'gs://quickdraw_dataset/full/simplified/*.ndjson' .`
- **Hugging Face:** `datasets.load_dataset("google/quickdraw")`
- **Direct URL per category:** `https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson`
- **Sketch-RNN NPZ:** `https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{category}.npz`

### Strengths
- Massive scale -- can select any subset of 345 categories
- Multiple well-documented formats
- Pre-simplified with RDP algorithm
- Already used as the standard benchmark for sketch-rnn, SketchHealer, VQ-SGen, SwiftSketch, and most vector sketch generation research
- CC-BY 4.0 license is very permissive
- Sketch-RNN NPZ format includes train/valid/test splits

### Weaknesses
- Variable-length strokes require resampling to fixed point count
- Simplified format is in pixel space (0-255), needs normalization
- Stroke-3 format uses deltas (offsets), needs cumulative sum for absolute coords
- Many drawings are low quality (rushed doodles from a timed game)
- Some categories have noisy/ambiguous samples

### Recommendation
**Use this as the primary dataset.** Start with 10-20 categories, use simplified NDJSON format. The project already has a download script (`scripts/download_quickdraw.py`) and raw data in `data/raw/quickdraw/` for 10 categories.

---

## 2. TU-Berlin Sketch Dataset

**Relevance: 3/5**

### Overview
- **Size:** 20,000 sketches across 250 categories (80 per category)
- **License:** Research use (academic license)
- **Source:** http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
- **Paper:** "How Do Humans Sketch Objects?" (Eitz, Hays & Alexa, SIGGRAPH 2012)

### Data Format
- Original format: SVG vector files
- Also available as PNG rasters on Kaggle/Hugging Face
- SVG paths contain bezier curves and line segments

### Mapping to DriftSketch Format
1. Parse SVG path commands (`M`, `L`, `C`, `Q`, etc.)
2. Sample points along bezier curves at regular intervals
3. Concatenate all path segments
4. Resample to fixed `num_points`, normalize to `[-1, 1]`

### Download Methods
- Official website (SVG files)
- Hugging Face: `sdiaeyu6n/tu-berlin`
- Kaggle: `zara2099/tu-berlin-hand-sketch-image-dataset` (raster only)

### Strengths
- Curated by researchers, higher quality than QuickDraw
- 250 diverse object categories
- SVG format preserves full vector structure
- Well-established benchmark in sketch recognition

### Weaknesses
- Small dataset (only 80 sketches per category)
- SVG parsing is more complex than QuickDraw's simple coordinate arrays
- Academic license may restrict commercial use
- No pre-split into train/valid/test

### Recommendation
Useful as a secondary validation dataset or for fine-tuning on higher-quality sketches. Too small for primary training. SVG parsing adds pipeline complexity.

---

## 3. Sketchy Database

**Relevance: 3/5**

### Overview
- **Size:** 75,471 sketches of 12,500 objects across 125 categories
- **License:** Research use
- **Source:** https://sketchy.eye.gatech.edu/
- **Paper:** "The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies" (SIGGRAPH 2016)

### Data Format
- Sketches stored as **SVG files** with stroke data
- Paired with corresponding photographs (useful for future image-conditioned generation)
- SVG paths contain temporal stroke ordering

### Mapping to DriftSketch Format
1. Parse SVG path data for each sketch
2. Sample points along paths at regular intervals
3. Resample to fixed length, normalize to `[-1, 1]`

### Download Methods
- Official website: https://sketchy.eye.gatech.edu/
- Download includes sketches (SVG), photos, and annotations

### Strengths
- Sketch-photo pairs enable future image-conditioned generation
- Good category diversity (125 categories)
- Fine-grained object associations
- Larger than TU-Berlin (75K sketches)

### Weaknesses
- SVG parsing required (same complexity as TU-Berlin)
- Academic/research license
- Download requires registration
- Not as widely used for generative models as QuickDraw

### Recommendation
Worth considering for future image-to-sketch generation tasks. The sketch-photo pairing is unique and valuable. Lower priority for initial training due to SVG parsing overhead.

---

## 4. FIGR-8-SVG

**Relevance: 2/5**

### Overview
- **Size:** 1,548,256 images across 17,375 classes
- **License:** Not clearly specified (research use)
- **Source:** https://github.com/marcdemers/FIGR-8-SVG

### Data Format
- Black-and-white vector icons in SVG format
- Also available on Hugging Face in parquet format: `starvector/FIGR-SVG`
- SVG paths with complex commands (curves, arcs, etc.)

### Download Methods
- GitHub repository
- Academic Torrents (1.92 GB)
- Hugging Face: `starvector/FIGR-SVG`
- Google Drive (linked from repo)

### Strengths
- Massive scale (1.5M+ icons)
- Clean, professional vector graphics
- Many categories for diverse conditioning

### Weaknesses
- **Icons are not hand-drawn sketches** -- different visual domain
- Complex SVG paths (machine-generated, not stroke-based)
- Many icons have fills, gradients, and complex shapes that do not map naturally to point sequences
- Very high number of classes (17K) with few samples each (~89 per class)
- License unclear

### Recommendation
**Not recommended** for DriftSketch's current scope. The domain mismatch (professional icons vs. hand-drawn sketches) makes this unsuitable. Could be useful for a separate SVG generation project.

---

## 5. Sketch-RNN Extra Datasets (hardmaru)

**Relevance: 4/5**

### Overview
- **Source:** https://github.com/hardmaru/sketch-rnn-datasets
- **Datasets included:**
  - **Kanji:** 11,100 Japanese characters in stroke-3 format (train: 10,000 / valid: 600 / test: 500)
  - **Aaron's Sheep:** 8,000 AARON-style sheep (train: 7,400 / valid: 300 / test: 300)
  - **Omniglot:** Handwritten characters from 50 alphabets

### Data Format
- `.npz` files in stroke-3 format: `(dx, dy, pen_state)` as `np.int16`
- Same format as QuickDraw Sketch-RNN NPZ
- Pre-split into train/valid/test

### Mapping to DriftSketch Format
- Identical conversion pipeline as QuickDraw Sketch-RNN format
- Cumulative sum of deltas -> absolute coords -> resample -> normalize

### Download Methods
- Direct from GitHub repository (small files, ~1-10 MB each)

### Strengths
- Ready-to-use stroke-3 format (same as QuickDraw NPZ)
- Unique domains: Japanese calligraphy, algorithmic art
- Small download size
- Pre-split datasets

### Weaknesses
- Small datasets
- Niche categories (not general object sketches)
- Single-class datasets (no multi-class conditioning)

### Recommendation
Good for testing the pipeline and for specialized experiments. The kanji dataset is particularly interesting for complex stroke generation. Same processing pipeline as QuickDraw.

---

## 6. Creative Birds / Creative Creatures (DoodlerGAN)

**Relevance: 2/5**

### Overview
- **Size:** Creative Birds: 8,067 sketches; Creative Creatures: 9,097 sketches
- **License:** MIT
- **Source:** https://github.com/facebookresearch/DoodlerGAN
- **Paper:** "Creative Sketch Generation" (ICLR 2021)

### Data Format
- Part-annotated sketches (head, body, wings, legs, etc.)
- Stored as coordinate sequences with part labels
- Raster-oriented pipeline (part images rather than vector strokes)

### Mapping to DriftSketch Format
- Would need to extract and concatenate part coordinates into full sketches
- Part annotations add complexity not needed for basic generation

### Strengths
- MIT license (very permissive)
- Creative/non-standard sketches (not just object recognition)
- Part annotations could enable part-aware generation

### Weaknesses
- Small datasets (~8-9K each)
- Only 2 categories (birds, creatures)
- Part-based format requires extra processing
- Primarily designed for raster generation pipelines

### Recommendation
Low priority for DriftSketch. Limited categories and part-based format add complexity without clear benefit for the current CFM approach.

---

## 7. ControlSketch Dataset (SwiftSketch)

**Relevance: 3/5** (future consideration)

### Overview
- **Size:** 35,000 image-sketch pairs across 100 categories
- **License:** Research (associated with SIGGRAPH 2025 paper)
- **Source:** https://swiftsketch.github.io/
- **Paper:** "SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation" (SIGGRAPH 2025)

### Data Format
- SVG vector sketches paired with images
- Synthetic, high-quality sketches generated from images

### Strengths
- Image-sketch pairs for conditional generation
- 100 categories with consistent quality
- Recent dataset with modern generation standards

### Weaknesses
- Synthetic (not human-drawn)
- SVG format requires parsing
- Availability may be limited (recent paper)

### Recommendation
Worth monitoring for future image-conditioned sketch generation work. Not yet widely available or tested.

---

## Summary Comparison

| Dataset | Samples | Categories | Format | License | Relevance |
|---------|---------|------------|--------|---------|-----------|
| **QuickDraw** | 50M+ | 345 | stroke arrays / stroke-3 npz | CC-BY 4.0 | **5/5** |
| **Sketch-RNN Extra** | 8K-11K each | 1 per file | stroke-3 npz | varies | **4/5** |
| **TU-Berlin** | 20,000 | 250 | SVG | Academic | 3/5 |
| **Sketchy** | 75,471 | 125 | SVG + photos | Academic | 3/5 |
| **ControlSketch** | 35,000 | 100 | SVG + images | Research | 3/5 |
| **FIGR-8-SVG** | 1.5M+ | 17,375 | SVG icons | Unclear | 2/5 |
| **DoodlerGAN** | ~17K | 2 | part-annotated | MIT | 2/5 |

---

## Recommended Implementation Plan

### Phase 1: QuickDraw (immediate)
1. Use the **simplified NDJSON format** (already downloading via `scripts/download_quickdraw.py`)
2. Start with the **10 categories already downloaded**: airplane, bicycle, car, cat, face, fish, flower, guitar, house, tree
3. Processing pipeline:
   - Parse NDJSON, extract `drawing` field (list of strokes)
   - Concatenate stroke points: zip x/y arrays per stroke into (x, y) tuples
   - Resample each drawing to fixed `num_points` (64 or 128) using linear interpolation
   - Normalize from `[0, 255]` to `[-1, 1]`
   - Assign integer class labels (0-9 for 10 categories)
4. Target: ~10,000 samples per category for training (100K total)

### Phase 2: Expand QuickDraw categories
- Scale to 50-100 categories from the full 345 available
- Use Sketch-RNN NPZ format for convenience (pre-split train/valid/test)

### Phase 3: Supplementary datasets (future)
- Add TU-Berlin or Sketchy for higher-quality sketches
- Explore image-conditioned generation with Sketchy or ControlSketch pairs

### Key Conversion Code Pattern

```python
import json
import numpy as np

def load_quickdraw_ndjson(filepath: str, max_samples: int = 10000) -> list[np.ndarray]:
    """Load simplified QuickDraw NDJSON and return list of point arrays."""
    drawings = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            if not data.get("recognized", True):
                continue
            points = []
            for stroke in data["drawing"]:
                xs, ys = stroke[0], stroke[1]
                points.extend(zip(xs, ys))
            drawings.append(np.array(points, dtype=np.float32))
    return drawings

def resample_and_normalize(points: np.ndarray, num_points: int = 64) -> np.ndarray:
    """Resample to fixed length and normalize to [-1, 1]."""
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    if total_length < 1e-6:
        return np.zeros((num_points, 2), dtype=np.float32)
    # Resample at uniform arc-length intervals
    target_lengths = np.linspace(0, total_length, num_points)
    resampled = np.column_stack([
        np.interp(target_lengths, cum_length, points[:, 0]),
        np.interp(target_lengths, cum_length, points[:, 1]),
    ])
    # Normalize from [0, 255] to [-1, 1]
    resampled = (resampled / 255.0) * 2.0 - 1.0
    return resampled.astype(np.float32)
```

---

## References

- QuickDraw Dataset: https://github.com/googlecreativelab/quickdraw-dataset
- QuickDraw on Hugging Face: https://huggingface.co/datasets/google/quickdraw
- Sketch-RNN Paper: https://arxiv.org/abs/1704.03477
- Sketch-RNN Extra Datasets: https://github.com/hardmaru/sketch-rnn-datasets
- TU-Berlin: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
- Sketchy Database: https://sketchy.eye.gatech.edu/
- FIGR-8-SVG: https://github.com/marcdemers/FIGR-8-SVG
- DoodlerGAN: https://github.com/facebookresearch/DoodlerGAN
- SwiftSketch: https://swiftsketch.github.io/
- QuickDraw NDJSON to NPZ converter: https://github.com/hardmaru/quickdraw-ndjson-to-npz
