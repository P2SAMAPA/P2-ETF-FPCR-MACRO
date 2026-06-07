# Functional Principal Component Regression (FPCR) with Macro-Weighted Curves

Applies functional data analysis to ETF return curves. Each ETF's return series is smoothed with a B‑spline basis. The score is the derivative (slope) at the last point – a smooth momentum signal. Macro variables are not used in this version but can be integrated as covariates in a future extension.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- B‑spline smoothing with configurable number of basis functions
- Score = instantaneous slope (derivative) at the end of the window
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-fpcr-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n * n_basis))
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- Positive slope → upward curvature / momentum at the end of the window → expected to continue upward.
- Negative slope → downward curvature.

## Requirements

See `requirements.txt`.
