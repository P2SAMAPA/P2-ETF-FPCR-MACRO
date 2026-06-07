import numpy as np
from scipy.linalg import svd

def bspline_basis(x, n_basis, degree=3):
    """
    Generate B‑spline basis matrix (simplified using equidistant knots).
    """
    n = len(x)
    knots = np.linspace(x.min(), x.max(), n_basis - degree + 1)
    # Use scipy's BSpline; if not available, fallback to polynomial basis
    try:
        from scipy.interpolate import BSpline
        # Create knot vector with multiplicity at endpoints
        t = np.r_[tuple([x.min()] * degree), knots, tuple([x.max()] * degree)]
        basis = np.zeros((n, n_basis))
        for j in range(n_basis):
            coef = np.zeros(n_basis)
            coef[j] = 1.0
            spl = BSpline(t, coef, degree)
            basis[:, j] = spl(x)
        return basis
    except:
        # Fallback: use polynomial basis (degree = n_basis-1)
        basis = np.zeros((n, n_basis))
        for j in range(n_basis):
            basis[:, j] = x ** j
        return basis

def functional_pca(returns_matrix, n_basis=10, n_components=3):
    """
    Perform functional PCA on a set of return curves (curves × time).
    returns_matrix: (n_curves, n_timepoints) – each row is a curve (e.g., ETF returns over time)
    Returns:
        scores: (n_curves, n_components) – FPC scores for each curve
        components: (n_timepoints, n_components) – principal functions evaluated on time grid
        explained_variance: array
    """
    n_curves, n_time = returns_matrix.shape
    # Define time grid
    time_grid = np.linspace(0, 1, n_time)
    # Compute basis expansion for each curve
    basis = bspline_basis(time_grid, n_basis)
    # Coefficients for each curve: least squares fit to basis
    coefs = np.zeros((n_curves, n_basis))
    for i in range(n_curves):
        coefs[i, :] = np.linalg.lstsq(basis, returns_matrix[i, :], rcond=None)[0]
    # PCA on coefficients (weighted by basis inner product)
    dt = time_grid[1] - time_grid[0]
    G = basis.T @ basis * dt
    try:
        sqrtG = np.linalg.cholesky(G)
        inv_sqrtG = np.linalg.inv(sqrtG)
        X = coefs @ inv_sqrtG.T
        U, s, Vt = svd(X, full_matrices=False)
        scores = X @ Vt[:n_components, :].T
        explained_variance = s[:n_components]**2 / (s**2).sum()
        phi = basis @ (inv_sqrtG.T @ Vt[:n_components, :].T)
        # Normalise phi to have unit norm
        for j in range(phi.shape[1]):
            phi[:, j] = phi[:, j] / np.sqrt(np.trapz(phi[:, j]**2, time_grid))
    except:
        # Fallback: ordinary PCA on raw curves
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(returns_matrix)
        explained_variance = pca.explained_variance_ratio_
        phi = pca.components_.T
    return scores, phi, explained_variance

def fpcr_score(returns_curve, macro_df, n_basis=10, n_components=3):
    """
    Simplified FPCR: smooth the return curve and compute the derivative at the last point.
    """
    if len(returns_curve) < 5:
        return 0.0
    n = len(returns_curve)
    time_grid = np.linspace(0, 1, n)
    basis = bspline_basis(time_grid, n_basis)
    # Fit coefficients
    coef = np.linalg.lstsq(basis, returns_curve, rcond=None)[0]
    # Evaluate smoothed curve at all time points
    smoothed = basis @ coef
    # Compute derivative at the last point using finite differences on the smoothed curve
    if n < 2:
        return 0.0
    # Use backward difference
    slope = smoothed[-1] - smoothed[-2]
    return slope

def fpcr_aggregate_score(returns, macro_df, n_basis=10, n_components=3):
    """
    For a single ETF (returns vector), compute the smoothed slope as the score.
    """
    if len(returns) < 5:
        return 0.0
    slope = fpcr_score(returns, macro_df, n_basis, n_components)
    return float(slope)
