"""
Wine Quality Project (UCI) - Regression-focused

- Downloads UCI winequality-red.csv / winequality-white.csv if missing
- Loads red, white, or combined dataset
- Runs EDA + saves plots
- Trains baseline + regularized + tree-based models
- Reports CV + holdout test metrics
- Saves outputs to ./outputs/

Dataset: UCI Wine Quality (Cortez et al., 2009) includes red/white vinho verde variants,
physicochemical inputs, and sensory quality score (0-10). (See UCI dataset page.)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---- UCI download URLs (classic location) ----
# These are the standard file endpoints referenced by UCI mirrors/packages. [3](https://rdrr.io/github/coatless/ucidata/man/wine.html)
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
RED_URL = UCI_BASE + "winequality-red.csv"
WHITE_URL = UCI_BASE + "winequality-white.csv"


def ensure_data(data_dir: Path) -> None:
    """Download UCI CSVs if they don't exist locally."""
    data_dir.mkdir(parents=True, exist_ok=True)

    red_path = data_dir / "winequality-red.csv"
    white_path = data_dir / "winequality-white.csv"

    if not red_path.exists():
        print(f"Downloading {RED_URL} -> {red_path}")
        urllib.request.urlretrieve(RED_URL, red_path)

    if not white_path.exists():
        print(f"Downloading {WHITE_URL} -> {white_path}")
        urllib.request.urlretrieve(WHITE_URL, white_path)


def load_data(data_dir: Path, which: str = "both") -> pd.DataFrame:
    """
    Load red, white, or combined wine quality dataset.
    UCI files are semicolon-separated. [1](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)[2](https://archive.ics.uci.edu/dataset/186/wine+quality)
    """
    red_path = data_dir / "winequality-red.csv"
    white_path = data_dir / "winequality-white.csv"

    if which not in {"red", "white", "both"}:
        raise ValueError("which must be one of: red, white, both")

    if which in {"red", "both"}:
        red = pd.read_csv(red_path, sep=";")
        red["wine_type"] = "red"
    else:
        red = None

    if which in {"white", "both"}:
        white = pd.read_csv(white_path, sep=";")
        white["wine_type"] = "white"
    else:
        white = None

    if which == "red":
        return red
    if which == "white":
        return white

    df = pd.concat([red, white], axis=0, ignore_index=True)
    return df


def make_outputs_dir(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)


def save_basic_eda(df: pd.DataFrame, outputs_dir: Path) -> None:
    """Save a few basic EDA artifacts to outputs/."""
    # Summary stats
    summary = df.describe(include="all").transpose()
    summary.to_csv(outputs_dir / "eda_summary.csv")

    # Missingness
    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv(outputs_dir / "missing_values.csv")

    # Quality distribution
    plt.figure()
    df["quality"].value_counts().sort_index().plot(kind="bar")
    plt.title("Quality Score Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outputs_dir / "quality_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Save correlation table
    corr.to_csv(outputs_dir / "correlation_matrix.csv")


def stratified_bins(y: pd.Series, n_bins: int = 6) -> np.ndarray:
    """
    Create bins for stratified split in regression:
    quality is an integer-ish ordinal target, so binning helps preserve distribution.
    """
    # Use quantile bins with fallback to unique bins if needed
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        return bins.astype(str).to_numpy()
    except ValueError:
        # fallback: use the raw integer values
        return y.astype(str).to_numpy()


def build_models(random_state: int = 42):
    """
    Return model specs with (name, estimator, param_grid, needs_scaling).
    """
    models = []

    # Baseline linear regression
    models.append((
        "LinearRegression",
        LinearRegression(),
        None,
        True
    ))

    # Regularized linear models
    models.append((
        "Ridge",
        Ridge(random_state=random_state),
        {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        True
    ))

    models.append((
        "Lasso",
        Lasso(random_state=random_state, max_iter=10000),
        {"model__alpha": [0.001, 0.01, 0.1, 1.0, 5.0]},
        True
    ))

    # Tree-based models (no scaling needed)
    models.append((
        "RandomForest",
        RandomForestRegressor(
            random_state=random_state,
            n_estimators=400,
            n_jobs=-1
        ),
        {
            "model__max_depth": [None, 6, 10, 16],
            "model__min_samples_split": [2, 5, 10]
        },
        False
    ))

    models.append((
        "GradientBoosting",
        GradientBoostingRegressor(random_state=random_state),
        {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3]
        },
        False
    ))

    return models


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test, outputs_dir: Path, do_grid: bool, param_grid=None):
    """
    Cross-validate + (optionally) grid search + test evaluation.
    Returns dict with metrics and best params.
    """
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Optional GridSearchCV
    best_estimator = pipeline
    best_params = {}

    if do_grid and param_grid:
        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_estimator = gs.best_estimator_
        best_params = gs.best_params_
    else:
        best_estimator.fit(X_train, y_train)

    # Cross-validation on training using the best estimator
    cv_results = cross_validate(
        best_estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    cv_rmse = -np.mean(cv_results["test_rmse"])
    cv_mae = -np.mean(cv_results["test_mae"])
    cv_r2 = np.mean(cv_results["test_r2"])

    # Test metrics
    y_pred = best_estimator.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # Save predicted vs actual plot
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.35)
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title(f"Predicted vs Actual - {name}")
    # reference line
    minv, maxv = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv])
    plt.tight_layout()
    plt.savefig(outputs_dir / f"pred_vs_actual_{name}.png", dpi=150)
    plt.close()

    # Save residual plot
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.35)
    plt.axhline(0)
    plt.xlabel("Predicted Quality")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residual Plot - {name}")
    plt.tight_layout()
    plt.savefig(outputs_dir / f"residuals_{name}.png", dpi=150)
    plt.close()

    # Feature importance / coefficients
    feature_report_path = outputs_dir / f"feature_effects_{name}.csv"
    try:
        model = best_estimator.named_steps["model"]
        if hasattr(model, "coef_"):
            effects = pd.Series(model.coef_, index=X_train.columns).sort_values(key=np.abs, ascending=False)
            effects.to_csv(feature_report_path, header=["coefficient"])
        elif hasattr(model, "feature_importances_"):
            effects = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            effects.to_csv(feature_report_path, header=["importance"])
    except Exception:
        pass

    return {
        "model": name,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "cv_r2": cv_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "best_params": best_params
    }


def main():
    parser = argparse.ArgumentParser(description="Wine Quality Project (UCI) - Regression")
    parser.add_argument("--which", choices=["red", "white", "both"], default="both",
                        help="Which dataset variant to use.")
    parser.add_argument("--data_dir", default="data", help="Directory to store/load dataset CSV files.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory to write results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_grid", action="store_true", help="Skip grid search (faster).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)
    make_outputs_dir(outputs_dir)

    # Get data
    ensure_data(data_dir)
    df = load_data(data_dir, which=args.which)

    # Save EDA outputs
    save_basic_eda(df, outputs_dir)

    # Features/target
    target = "quality"
    drop_cols = [target]
    # For now, keep wine_type only if 'both'. We'll one-hot encode it manually for simplicity.
    if "wine_type" in df.columns and args.which == "both":
        # One-hot encode wine_type
        df = pd.get_dummies(df, columns=["wine_type"], drop_first=True)

    X = df.drop(columns=drop_cols)
    y = df[target].astype(float)

    # Train/test split with stratified bins
    strat = stratified_bins(y, n_bins=6)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=strat
    )

    # Build and evaluate models
    results = []
    models = build_models(random_state=args.seed)

    for name, estimator, grid, needs_scaling in models:
        steps = []
        if needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))

        pipe = Pipeline(steps=steps)
        print(f"\n=== Training: {name} ===")
        res = evaluate_model(
            name=name,
            pipeline=pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            outputs_dir=outputs_dir,
            do_grid=(not args.no_grid),
            param_grid=grid
        )
        results.append(res)

    # Save results table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="test_rmse", ascending=True)
    results_df.to_csv(outputs_dir / "model_results.csv", index=False)

    print("\nDone. Top models by test RMSE:")
    print(results_df[["model", "test_rmse", "test_mae", "test_r2"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()