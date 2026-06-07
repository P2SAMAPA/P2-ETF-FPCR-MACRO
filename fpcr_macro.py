import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def bspline_basis(x, n_basis, degree=3):
    """
    Generate B‑spline basis matrix (simplified using equidistant knots).
    Falls back to polynomial basis if scipy is not available.
    """
    n = len(x)
    try:
        from scipy.interpolate import BSpline
        knots = np.linspace(x.min(), x.max(), n_basis - degree + 1)
        t = np.concatenate(([x.min()]*degree, knots, [x.max()]*degree))
        basis = np.zeros((n, n_basis))
        for j in range(n_basis):
            coef = np.zeros(n_basis)
            coef[j] = 1.0
            spl = BSpline(t, coef, degree)
            basis[:, j] = spl(x)
        return basis
    except:
        # Fallback: polynomial basis
        basis = np.zeros((n, n_basis))
        for j in range(n_basis):
            basis[:, j] = x ** j
        return basis

def fpcr_with_macro(returns, macro_df, n_basis=10, alpha=1.0):
    """
    Functional Principal Component Regression with macro covariates.
    Uses smoothed slope multiplied by macro factor (e.g., VIX).
    """
    # returns is a numpy array
    if len(returns) < 5 or macro_df is None or macro_df.empty:
        return 0.0
    n = len(returns)
    time_grid = np.linspace(0, 1, n)
    basis = bspline_basis(time_grid, n_basis)
    # Fit coefficients (returns is already array)
    coef = np.linalg.lstsq(basis, returns, rcond=None)[0]
    # Smoothed curve
    smoothed = basis @ coef
    # Slope at last point (finite difference)
    if n < 2:
        return 0.0
    slope = smoothed[-1] - smoothed[-2]
    # Macro factor: use VIX if present, else 1.0
    vix_factor = 1.0
    if "VIX" in macro_df.columns:
        # macro_df is a DataFrame, get last row's VIX value
        vix = macro_df.iloc[-1]["VIX"]
        # Normalise: baseline VIX ~20
        vix_factor = max(0.5, min(2.0, vix / 20.0))
    score = slope * vix_factor
    return float(score)

def fpcr_aggregate_score(returns, macro_df, n_basis=10, n_components=None):
    """Wrapper for train.py."""
    return fpcr_with_macro(returns, macro_df, n_basis)
