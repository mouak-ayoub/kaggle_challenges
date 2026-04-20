# GitHub Copilot Instructions

Instructions are organized per challenge. Only follow the rules for the challenge you are currently working on.

---

## Physionet ECG Challenge (`Physionet_ECG_challenge/`)

**Scope:** Only apply these rules when working on files inside `Physionet_ECG_challenge/`

### Required Reading (Start of Session)
**ALWAYS read `Physionet_ECG_challenge/LOCAL_PROJECT_MEMORY.md` first** before answering questions or making changes. This file contains:
- Current project stage and goals
- What has been confirmed/learned
- Active notebook organization
- Latest milestones and next steps

### Read-Only Memory
- **DO NOT modify** `LOCAL_PROJECT_MEMORY.md` - this file is managed by Codex only
- Use its contents to understand context but never write to it

### Key Files to Reference
- `README.md` - Architecture decisions and Stage A/B pipeline design
- `LOCAL_PROJECT_MEMORY.md` - Current working state (READ ONLY)
- `src/` - Reusable code organized by semantic stage

### Code Style for ECG Project
- Prefer `scikit-image`, `opencv` framework implementations
- Use dataclass configs for parameters
- Keep notebooks step-by-step with short markdown
- Separate reusable src code from notebook-specific plotting

### Hough Near-Miss Recovery — Method 3 (family-constrained 1D ρ refinement)

**Problem:** a true border line can sit just below the global threshold (e.g. accumulator=183 vs threshold=184.8).
Lowering the global threshold globally is brittle; it degrades precision on other images.

**Solution implemented in Step 5 of `ecg_hough_lines_page_detection.ipynb`:**

1. **Stage 1 (already done in Step 3):** use the global threshold to find the dominant angle family θ₀.
2. **Stage 2 (Step 5):** fix that angle family and collapse the 2D accumulator to a 1D signal over ρ:
   ```
   s(ρ) = max_{θ ∈ [θ₀-Δ, θ₀+Δ]}  H(ρ, θ)   # or sum
   ```
3. Detect **local maxima with prominence** in `s(ρ)` using `scipy.signal.find_peaks`.
4. A border line that was just below the global 2D threshold can still appear as a clear local peak in the 1D signal, because competition is now only within one angle family.

**Key config variables (notebook-local):**
- `RHO1D_DELTA_THETA_DEG_DOMINANT` / `RHO1D_DELTA_THETA_DEG_PERP` — band half-width (reuse family tolerances)
- `RHO1D_MIN_PROMINENCE` — min relative prominence fraction (default 0.05)

**Why better than lowering global threshold:**
- Targeted relaxation: only inside the correct angle family
- Detects real local maxima, not just anything above an absolute level
- Prints which 1D peaks are BELOW the global threshold (recovery candidates)

---

## Other Challenges

Other challenges in this workspace do not have specific AI instructions yet.
Add a section here when needed for: `BirdCLEF_CHALLENGE/`, `cibmtr_challenge/`, `foot_challenge/`, `Jigsaw_challenge/`

