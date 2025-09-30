# FMR Lab — Overview Notebook (Module‑based)

**Goal:** keep heavy code in Python modules and use a thin, explanatory notebook as the lab UI.

## Files here
- `labio.py` — secure `.txt/.csv` loading and creation of `H` column
- `labfit.py` — unified `fit_model(x, y, model=...)` with derivative line-shapes
- `FMR_Lab_Overview.ipynb` — the thin notebook that calls those modules

> In your repo, move `labio.py` and `labfit.py` into `src/` (e.g., `src/io/loader.py`, `src/analysis/fitting.py`) and just update the import cell in the notebook.

## How to run
1. Open `FMR_Lab_Overview.ipynb` in Jupyter/VS Code.
2. Set `DATA_FOLDER`, `SLOPE_T_PER_A`, `INTERCEPT_T`, `MODEL_NAME` in the config cell.
3. Run cells from top to bottom. Use the optional synthetic generator if you need a quick demo.

## Next additions (recommended)
- `analysis/kittel.py` — fit `f vs H_res` (Kittel) for `γ/2π` and `M_s`.
- `analysis/damping.py` — fit `ΔH vs f` for `α` and `ΔH₀`.
- `plot/` helpers for consistent figures.
