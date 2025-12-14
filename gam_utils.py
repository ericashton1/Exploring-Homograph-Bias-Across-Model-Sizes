import numpy as np
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit

# Reference: pyGAM docs on LinearGAM regularization and penalty tuning patterns.
# https://pygam.readthedocs.io/en/stable/api/gam.html


def fit_regularized_gam(
    X,
    y,
    lam_values=None,
    n_splines=10,
    n_splits=5,
    test_size=0.2,
    random_state=42,
    verbose=True,
):
    """
    Fit a linear GAM with a single smooth term while tuning the smoothing penalty (lam)
    via lightweight repeated holdout validation. Returns the fitted GAM, the best lam,
    and diagnostics for each lam that was tested.
    """
    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(y).ravel()

    if lam_values is None:
        lam_values = np.logspace(-3, 2, 8)
    lam_values = list(lam_values)

    splitter = ShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    splits = list(splitter.split(X))

    diagnostics = []
    for lam in lam_values:
        fold_mse = []
        for train_idx, val_idx in splits:
            gam = LinearGAM(s(0, n_splines=n_splines), lam=lam).fit(
                X[train_idx], y[train_idx]
            )
            preds = gam.predict(X[val_idx])
            fold_mse.append(mean_squared_error(y[val_idx], preds))
        diagnostics.append(
            {
                "lam": float(lam),
                "mse": float(np.mean(fold_mse)),
                "std": float(np.std(fold_mse)),
            }
        )

    best = min(diagnostics, key=lambda item: item["mse"])
    best_gam = LinearGAM(s(0, n_splines=n_splines), lam=best["lam"]).fit(X, y)

    if verbose:
        print(
            f"[GAM] selected lam={best['lam']:.4f} "
            f"(mean MSE={best['mse']:.4f}, std={best['std']:.4f})"
        )

    return best_gam, best["lam"], diagnostics

