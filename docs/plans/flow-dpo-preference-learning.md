# Plan: Preference Learning (Flow-DPO) Pipeline for DriftSketch

## Context

DriftSketch trains a CFM model to generate vector sketches (32 cubic Bezier strokes). The model learns to predict velocity fields that transform noise into structured sketches. Currently, training optimizes only MSE on the velocity field against ground-truth data. There's no mechanism to incorporate human judgment about sketch quality.

This plan adds two capabilities: (1) a UI for collecting human preferences on generated sketches, and (2) a Flow-DPO training loop that fine-tunes the base model to favor preferred outputs. The approach adapts Diffusion-DPO (Wallace et al., CVPR 2024) to conditional flow matching, where the DPO objective operates on velocity prediction errors as a proxy for log-likelihood.

## How Flow-DPO Works for CFM

For each preference pair (winner `x_w`, loser `x_l`) sharing the same class condition:

1. Sample shared noise `x0 ~ N(0,I)` and shared time `t ~ U(0,1)`
2. Interpolate: `xt_w = (1-t)*x0 + t*x_w`, `xt_l = (1-t)*x0 + t*x_l`
3. Compute velocity MSE for both current model and frozen reference model on both samples
4. DPO loss: `-log sigmoid(beta * ((ref_loss_w - model_loss_w) - (ref_loss_l - model_loss_l)))`

This pushes the model to assign lower velocity error (higher likelihood) to preferred sketches relative to the reference model baseline. Shared noise/time across pairs is critical for variance reduction.

---

## Step 1: Preference Data Format & Dataset (`src/driftsketch/data/preferences.py`)

**New file.** Storage format: JSONL metadata + .npy tensors.

```
data/preferences/          (gitignored)
  sessions/
    session_YYYYMMDD_HHMMSS.jsonl   # one JSON record per annotation
  beziers/
    pref_..._a.npy                  # (32, 4, 2) float32
    pref_..._b.npy
```

Each JSONL record stores: id, timestamp, comparison_type (`pairwise` or `absolute`), class_label, class_name, paths to bezier .npy files, preference (`a`/`b`/`tie`) or rating (1-5), checkpoint used, annotator.

`PreferenceDataset(Dataset)`:
- Loads all session JSONL files, filters to valid pairwise preferences
- Can synthesize pairs from absolute ratings (any two same-class samples where rating differs by `min_margin`)
- Returns `{winner_beziers: (32, 8), loser_beziers: (32, 8), class_label: int}`

## Step 2: Flow-DPO Loss Module (`src/driftsketch/dpo.py`)

**New file.** Core function:

```python
def flow_dpo_loss(model, ref_model, x1_w, x1_l, class_labels, beta=2000.0):
    # Shared noise + time for variance reduction
    x0 = randn_like(x1_w)
    t = rand(B, 1, 1)

    # Interpolate and get velocity targets for winner and loser
    # Forward through both model (grad) and ref_model (no_grad)
    # Per-sample MSE losses → reward_diff → -log sigmoid(beta * reward_diff)

    return loss, metrics_dict  # metrics: reward_diff, implicit_accuracy, component losses
```

Key details:
- `beta=2000` default (MSE values ~0.01-0.1, so beta*MSE_diff needs to be in ~(-5, 5) for sigmoid gradients)
- Returns detailed metrics dict for wandb logging (implicit accuracy, reward diff distribution)
- Metrics guide beta calibration: `reward_diff_mean` should stay in (-3, 3)

## Step 3: Preference Collection UI (`scripts/collect_preferences.py`)

**New file.** Gradio web app with two modes:

**Pairwise mode** (primary): Side-by-side rendered sketches, buttons for "A is better" / "B is better" / "Skip". Auto-advances to next pair after selection.

**Rating mode** (secondary): Grid of 4-8 sketches with star rating (1-5) per sketch. Pairs synthesized during training.

`PreferenceCollector` class manages:
- Loading a checkpoint and generating samples on-demand via existing `generate_beziers()`
- Rendering via existing `plot_bezier_sketches()` (or inline matplotlib)
- Appending annotations to session JSONL (crash-safe, append-only)
- Session state: count, current pair, category cycling

CLI: `python scripts/collect_preferences.py --checkpoint checkpoints/model.pt`

Dependency: `gradio>=4.0.0` added as optional dep `preference` in pyproject.toml.

## Step 4: DPO Training Script (`src/driftsketch/train_dpo.py`)

**New file.** Entry point: `driftsketch-train-dpo`

Key differences from base `train.py`:
- Loads base checkpoint, creates frozen `ref_model = deepcopy(model)`
- Uses `PreferenceDataset` instead of `ControlSketchDataset`
- Calls `flow_dpo_loss()` instead of MSE
- Lower default LR (`1e-5` vs `1e-4`) — fine-tuning, not training from scratch
- Optional SFT regularization: `--sft-weight` adds standard CFM loss on winners to prevent drift
- Saves both model and ref_model state_dicts in checkpoints (for resumability)
- Final checkpoint compatible with existing `driftsketch-generate` (same `model_state_dict` + `config` format)

CLI args: `--base-checkpoint` (required), `--beta`, `--sft-weight`, `--lr`, `--epochs`, `--batch-size`, `--preference-dir`, `--checkpoint-dir`, `--no-wandb`

## Step 5: Package Integration

- `pyproject.toml`: Add `driftsketch-train-dpo` entry point, add `preference = ["gradio>=4.0.0"]` optional dep
- `data/preferences/` added to `.gitignore`
- No changes needed to `model.py` or `inference.py` — DPO checkpoints are drop-in compatible with existing inference

---

## Files to Create/Modify

| File | Action | Purpose |
|---|---|---|
| `src/driftsketch/dpo.py` | Create | `flow_dpo_loss()` function |
| `src/driftsketch/data/preferences.py` | Create | `PreferenceDataset` class |
| `src/driftsketch/train_dpo.py` | Create | DPO training entry point |
| `scripts/collect_preferences.py` | Create | Gradio preference collection UI |
| `pyproject.toml` | Modify | Entry point + optional dep |
| `.gitignore` | Modify | Add `data/preferences/` |

Existing files (`model.py`, `train.py`, `inference.py`) are **not modified**. DPO checkpoints use the same format and are compatible with existing `driftsketch-generate`.

## Verification

1. **Loss function**: Construct deterministic test — winner = training data, loser = noise. After optimization steps, `implicit_accuracy` should increase toward 1.0
2. **Data pipeline**: Create mock JSONL + .npy, verify `PreferenceDataset` returns correct shapes `(32, 8)`
3. **Collection UI**: Launch Gradio app, collect 10 pairs, verify JSONL + .npy files written correctly
4. **End-to-end**: Collect ~50 pairs → `driftsketch-train-dpo --epochs 10` → loss decreases, checkpoint saves → `driftsketch-generate --checkpoint checkpoints/dpo/model_dpo.pt` works
5. **Metrics**: During training, `dpo/implicit_accuracy` rises from ~0.5; `dpo/reward_diff_mean` stays in (-3, 3)

## Data Volume Guidance

- **Minimum viable**: ~200 pairwise preferences (~7 min at 2s/pair)
- **Recommended**: 1000+ preferences (~35 min)
- **Diminishing returns**: Beyond 5000 pairs for offline DPO; online DPO (iterative generate→rate→train) becomes more valuable at that point

## Future Extensions (not in this plan)

- **Online DPO**: Integrate generation into training loop for iterative refinement
- **Image conditioning**: When Phase 2 CLIP lands, extend `flow_dpo_loss` to accept `clip_features` and store image paths in preference records
- **Automated reward model**: Train a reward model on collected preferences, use for reward-weighted fine-tuning or automated filtering
