"""
Deep Diagnostic Evaluation Suite
=================================
Goes beyond classic metrics (Accuracy, ROC, F1) to expose *why* models behave
like a coin flip on the test set.

Tiers:
  1. Statistical Robustness  — MCC, Cohen's Kappa, Log Loss, Brier Score
  2. Probability Calibration — Calibration Curve, ECE, Confidence Distribution
  3. Discriminative Power    — KS Statistic, Gini, Cumulative Gains / Lift
  4. Distribution Shift      — PSI per feature, train vs. test density plots
  5. Decision-Theoretic      — Decision Curve Analysis, Optimal Threshold (Youden's J)

Usage:
    python -m src.deep_evaluation
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss,
    roc_curve, auc, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

import src.config as config
from src.dataset import load_data, get_dataloader, DataProcessor
from src.model import ChurnModel

warnings.filterwarnings('ignore')

ASSETS_DIR = os.path.join(config.BASE_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: get predictions from both models
# ---------------------------------------------------------------------------

def _get_pytorch_predictions(X_test, y_test, processor):
    """Load PyTorch model and return (labels, probs, preds)."""
    device = torch.device('cpu')
    input_dim = processor.get_feature_dim()
    model = ChurnModel(input_dim=input_dim, hidden_units=config.HIDDEN_UNITS, dropout_rate=0.0).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    loader = get_dataloader(X_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            probs = model.predict_proba(bx.to(device))
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(by.numpy().flatten())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    return labels, probs, preds


def _get_tf_predictions(X_test):
    """Load TF model and return probs or None."""
    try:
        from tensorflow import keras
        from src.train_tf import TF_MODEL_PATH
        if not os.path.exists(TF_MODEL_PATH):
            return None
        tf_model = keras.models.load_model(TF_MODEL_PATH)
        return tf_model.predict(X_test, verbose=0).flatten()
    except Exception:
        return None


def _get_xgb_predictions(X_test):
    """Load XGBoost model and return probs or None."""
    try:
        import xgboost as xgb
        from src.train_xgb import XGB_MODEL_PATH
        if not os.path.exists(XGB_MODEL_PATH):
            return None
        clf = xgb.XGBClassifier()
        clf.load_model(XGB_MODEL_PATH)
        return clf.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"  Could not load XGBoost: {e}")
        return None


def _get_lgb_predictions(X_test):
    """Load LightGBM model and return probs or None."""
    try:
        import lightgbm as lgb
        from src.train_lgb import LGB_MODEL_PATH
        if not os.path.exists(LGB_MODEL_PATH):
            return None
        booster = lgb.Booster(model_file=LGB_MODEL_PATH)
        return booster.predict(X_test)
    except Exception as e:
        print(f"  Could not load LightGBM: {e}")
        return None


def _get_cat_predictions(X_test):
    """Load CatBoost model and return probs or None."""
    try:
        from catboost import CatBoostClassifier
        from src.train_cat import CAT_MODEL_PATH
        if not os.path.exists(CAT_MODEL_PATH):
            return None
        clf = CatBoostClassifier()
        clf.load_model(CAT_MODEL_PATH)
        return clf.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"  Could not load CatBoost: {e}")
        return None


# ---------------------------------------------------------------------------
# TIER 1 — Statistical Robustness
# ---------------------------------------------------------------------------

def tier1_statistical_robustness(labels, probs, preds, model_name="PyTorch MLP"):
    """MCC, Cohen's Kappa, Log Loss, Brier Score."""
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    ll = log_loss(labels, probs)
    brier = brier_score_loss(labels, probs)

    results = {
        'Model': model_name,
        'MCC': mcc,
        "Cohen's Kappa": kappa,
        'Log Loss': ll,
        'Brier Score': brier,
    }
    return results


# ---------------------------------------------------------------------------
# TIER 2 — Probability Calibration
# ---------------------------------------------------------------------------

def _expected_calibration_error(labels, probs, n_bins=10):
    """Compute ECE (Expected Calibration Error)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(labels)


def tier2_calibration_plots(labels, probs_dict):
    """
    Calibration curve + ECE + Confidence histogram for each model.
    probs_dict: {'PyTorch MLP': probs_array, 'TF/Keras MLP': probs_array, ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'PyTorch MLP': 'darkorange', 'TF/Keras MLP': 'green',
              'XGBoost': 'crimson', 'LightGBM': 'royalblue', 'CatBoost': 'purple'}

    # --- Calibration Curve ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly Calibrated')
    ece_results = {}
    for name, probs in probs_dict.items():
        fraction_pos, mean_predicted = calibration_curve(labels, probs, n_bins=10, strategy='uniform')
        ece = _expected_calibration_error(labels, probs)
        ece_results[name] = ece
        ax.plot(mean_predicted, fraction_pos, 's-', color=colors.get(name, 'gray'),
                label=f'{name} (ECE = {ece:.4f})', markersize=6)

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    # --- Confidence Distribution ---
    ax2 = axes[1]
    for name, probs in probs_dict.items():
        ax2.hist(probs, bins=50, alpha=0.5, color=colors.get(name, 'gray'),
                 label=name, edgecolor='black', linewidth=0.3)
    ax2.axvline(x=0.5, color='red', linestyle='--', lw=1.5, label='Decision Boundary (0.5)')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, 'calibration_and_confidence.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return ece_results


# ---------------------------------------------------------------------------
# TIER 3 — Discriminative Power & Separation
# ---------------------------------------------------------------------------

def _ks_statistic(labels, probs):
    """Kolmogorov-Smirnov statistic and optimal threshold."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    ks_values = tpr - fpr
    ks_stat = ks_values.max()
    ks_threshold = thresholds[ks_values.argmax()] if len(thresholds) > 0 else 0.5
    return ks_stat, ks_threshold, fpr, tpr, ks_values


def tier3_discrimination_plots(labels, probs_dict):
    """KS plot + Cumulative Gains + Lift chart."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    colors = {'PyTorch MLP': 'darkorange', 'TF/Keras MLP': 'green',
              'XGBoost': 'crimson', 'LightGBM': 'royalblue', 'CatBoost': 'purple'}

    ks_results = {}

    n_models = len(probs_dict)
    bar_width = 0.8 / max(n_models, 1)

    for model_idx, (name, probs) in enumerate(probs_dict.items()):
        color = colors.get(name, 'gray')

        # --- KS Chart ---
        ks_stat, ks_thresh, fpr, tpr, ks_vals = _ks_statistic(labels, probs)
        ks_results[name] = {'KS Statistic': ks_stat, 'KS Threshold': ks_thresh,
                            'Gini': 2 * auc(fpr, tpr) - 1}
        ax = axes[0]
        thresholds_plot = np.linspace(0, 1, len(tpr))
        ax.plot(thresholds_plot, tpr, color=color, lw=2, label=f'{name} TPR')
        ax.plot(thresholds_plot, fpr, color=color, lw=2, linestyle='--', alpha=0.6)

        # --- Cumulative Gains ---
        ax2 = axes[1]
        sorted_idx = np.argsort(probs)[::-1]
        sorted_labels = labels[sorted_idx]
        cum_gains = np.cumsum(sorted_labels) / sorted_labels.sum()
        percentile = np.arange(1, len(cum_gains) + 1) / len(cum_gains)
        ax2.plot(percentile, cum_gains, color=color, lw=2, label=name)

        # --- Lift Chart ---
        ax3 = axes[2]
        n_bins = 10
        bin_size = len(probs) // n_bins
        lifts = []
        baseline_rate = labels.mean()
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(sorted_labels)
            bin_rate = sorted_labels[start:end].mean()
            lifts.append(bin_rate / baseline_rate if baseline_rate > 0 else 1.0)
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        ax3.bar(np.arange(1, n_bins + 1) + offset,
                lifts, width=bar_width, color=color, alpha=0.8, label=name, edgecolor='black', linewidth=0.3)

    # Finalize KS plot
    ax = axes[0]
    ax.set_xlabel('Normalized Threshold', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('KS Chart (TPR vs FPR Separation)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Finalize Cumulative Gains
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Model')
    ax2.set_xlabel('% of Population Contacted', fontsize=12)
    ax2.set_ylabel('% of Churners Captured', fontsize=12)
    ax2.set_title('Cumulative Gains Chart', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Finalize Lift
    ax3 = axes[2]
    ax3.axhline(y=1.0, color='red', linestyle='--', lw=1.5, label='Baseline (No Model)')
    ax3.set_xlabel('Decile (1 = Highest Risk)', fontsize=12)
    ax3.set_ylabel('Lift', fontsize=12)
    ax3.set_title('Lift Chart by Decile', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, 'ks_gains_lift.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return ks_results


# ---------------------------------------------------------------------------
# TIER 4 — Distribution Shift (PSI)
# ---------------------------------------------------------------------------

def _psi(expected, actual, bins=10):
    """Population Stability Index between two arrays."""
    eps = 1e-4
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), bins + 1)
    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected) + eps
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual) + eps
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value


def tier4_distribution_shift(train_df, test_df):
    """PSI per numerical feature + density comparison plots."""
    numerical_cols = config.NUMERICAL_COLS
    n = len(numerical_cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()

    psi_results = {}
    for i, col in enumerate(numerical_cols):
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_vals = train_df[col].dropna().values.astype(float)
        test_vals = test_df[col].dropna().values.astype(float)

        psi_val = _psi(train_vals, test_vals, bins=20)
        psi_results[col] = psi_val

        ax = axes[i]
        ax.hist(train_vals, bins=40, alpha=0.5, density=True, color='steelblue',
                label='Train', edgecolor='black', linewidth=0.3)
        ax.hist(test_vals, bins=40, alpha=0.5, density=True, color='salmon',
                label='Test', edgecolor='black', linewidth=0.3)

        # PSI badge color
        badge_color = 'green' if psi_val < 0.1 else ('orange' if psi_val < 0.25 else 'red')
        ax.set_title(f'{col}\nPSI = {psi_val:.4f}', fontsize=11, fontweight='bold', color=badge_color)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distribution Shift: Train vs. Test (PSI)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, 'psi_distribution_shift.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return psi_results


# ---------------------------------------------------------------------------
# TIER 5 — Decision-Theoretic Analysis
# ---------------------------------------------------------------------------

def tier5_decision_curve(labels, probs_dict):
    """Decision Curve Analysis + Optimal Threshold (Youden's J)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'PyTorch MLP': 'darkorange', 'TF/Keras MLP': 'green',
              'XGBoost': 'crimson', 'LightGBM': 'royalblue', 'CatBoost': 'purple'}

    youden_results = {}

    # --- Decision Curve ---
    ax = axes[0]
    thresholds = np.linspace(0.01, 0.99, 200)
    prevalence = labels.mean()

    # "Treat All" baseline
    treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, treat_all_nb, 'k--', lw=1, label='Treat All')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1, label='Treat None')

    for name, probs in probs_dict.items():
        net_benefits = []
        for t in thresholds:
            pred_pos = (probs >= t).astype(int)
            tp = ((pred_pos == 1) & (labels == 1)).sum()
            fp = ((pred_pos == 1) & (labels == 0)).sum()
            n = len(labels)
            nb = (tp / n) - (fp / n) * (t / (1 - t))
            net_benefits.append(nb)
        ax.plot(thresholds, net_benefits, color=colors.get(name, 'gray'), lw=2, label=name)

    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('Decision Curve Analysis (DCA)', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, max(0.5, prevalence + 0.1)])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Youden's J / Optimal Threshold ---
    ax2 = axes[1]
    for name, probs in probs_dict.items():
        fpr, tpr, thresh = roc_curve(labels, probs)
        j_scores = tpr - fpr
        optimal_idx = j_scores.argmax()
        optimal_thresh = thresh[optimal_idx] if optimal_idx < len(thresh) else 0.5
        youden_results[name] = {'Youden J': j_scores[optimal_idx], 'Optimal Threshold': optimal_thresh}

        ax2.plot(thresh, tpr[:len(thresh)], color=colors.get(name, 'gray'), lw=2, label=f'{name} - TPR')
        ax2.plot(thresh, fpr[:len(thresh)], color=colors.get(name, 'gray'), lw=2, linestyle='--',
                 alpha=0.6, label=f'{name} - FPR')
        ax2.axvline(x=optimal_thresh, color=colors.get(name, 'gray'), linestyle=':',
                    label=f'Optimal T = {optimal_thresh:.3f}')

    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title("Youden's J — Optimal Threshold", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, 'dca_youden.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return youden_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_deep_evaluation():
    print("=" * 70)
    print("  DEEP DIAGNOSTIC EVALUATION SUITE")
    print("=" * 70)

    # Load data
    processor = DataProcessor.load(config.PROCESSOR_SAVE_PATH)
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)

    # Get predictions from ALL models
    print("\n[1/6] Loading model predictions...")
    labels, pt_probs, pt_preds = _get_pytorch_predictions(X_test, y_test, processor)

    probs_dict = {'PyTorch MLP': pt_probs}

    tf_probs = _get_tf_predictions(X_test)
    if tf_probs is not None:
        probs_dict['TF/Keras MLP'] = tf_probs

    xgb_probs = _get_xgb_predictions(X_test)
    if xgb_probs is not None:
        probs_dict['XGBoost'] = xgb_probs
        print("  Loaded XGBoost")

    lgb_probs = _get_lgb_predictions(X_test)
    if lgb_probs is not None:
        probs_dict['LightGBM'] = lgb_probs
        print("  Loaded LightGBM")

    cat_probs = _get_cat_predictions(X_test)
    if cat_probs is not None:
        probs_dict['CatBoost'] = cat_probs
        print("  Loaded CatBoost")

    print(f"  Total models loaded: {len(probs_dict)}")

    # ===========================================================
    # TIER 1 — Statistical Robustness
    # ===========================================================
    print("\n[2/6] TIER 1 — Statistical Robustness Indicators")
    tier1_rows = []
    for name, probs in probs_dict.items():
        preds = (probs >= 0.5).astype(int)
        row = tier1_statistical_robustness(labels, probs, preds, name)
        tier1_rows.append(row)

    t1_df = pd.DataFrame(tier1_rows).set_index('Model')
    print(t1_df.to_string())
    print(f"\n  Interpretation:")
    print(f"    MCC = 0 -> random coin flip   |   Cohen's K = 0 -> no agreement beyond chance")
    print(f"    Brier = 0.25 -> coin flip     |   Log Loss = 0.693 -> coin flip (ln(2))")

    # ===========================================================
    # TIER 2 — Calibration
    # ===========================================================
    print("\n[3/6] TIER 2 — Probability Calibration Diagnostics")
    ece_results = tier2_calibration_plots(labels, probs_dict)
    for name, ece in ece_results.items():
        print(f"  {name} ECE: {ece:.4f}")

    # ===========================================================
    # TIER 3 — Discriminative Power
    # ===========================================================
    print("\n[4/6] TIER 3 — Discriminative Power & Separation")
    ks_results = tier3_discrimination_plots(labels, probs_dict)
    for name, vals in ks_results.items():
        print(f"  {name}: KS = {vals['KS Statistic']:.4f}, "
              f"Gini = {vals['Gini']:.4f}, KS Threshold = {vals['KS Threshold']:.4f}")
    print(f"\n  Interpretation:")
    print(f"    KS = 0 -> no class separation  |  Gini = 0 -> no discriminative power")

    # ===========================================================
    # TIER 4 — Distribution Shift (PSI)
    # ===========================================================
    print("\n[5/6] TIER 4 — Distribution Shift Quantification (PSI)")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    psi_results = tier4_distribution_shift(train_df, test_df)

    print(f"\n  {'Feature':<20} {'PSI':>8}   Interpretation")
    print(f"  {'-'*20} {'-'*8}   {'-'*30}")
    for feat, psi_val in sorted(psi_results.items(), key=lambda x: -x[1]):
        if psi_val >= 0.25:
            interp = "[!!] MAJOR SHIFT -- distribution fundamentally different"
        elif psi_val >= 0.10:
            interp = "[!]  MODERATE SHIFT -- noticeable drift"
        else:
            interp = "[OK] STABLE -- no significant drift"
        print(f"  {feat:<20} {psi_val:>8.4f}   {interp}")

    print(f"\n  PSI Thresholds: < 0.10 = stable | 0.10-0.25 = moderate | > 0.25 = major shift")

    # ===========================================================
    # TIER 5 — Decision-Theoretic
    # ===========================================================
    print("\n[6/6] TIER 5 — Decision-Theoretic Analysis")
    youden_results = tier5_decision_curve(labels, probs_dict)
    for name, vals in youden_results.items():
        print(f"  {name}: Youden's J = {vals['Youden J']:.4f}, "
              f"Optimal Threshold = {vals['Optimal Threshold']:.4f}")

    # ===========================================================
    # COMPREHENSIVE SUMMARY TABLE
    # ===========================================================
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE DIAGNOSTIC SUMMARY")
    print("=" * 70)

    summary_rows = []
    for name, probs in probs_dict.items():
        preds = (probs >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc_val = auc(fpr, tpr)
        row = {
            'Model': name,
            'Accuracy': f"{accuracy_score(labels, preds):.4f}",
            'Precision': f"{precision_score(labels, preds):.4f}",
            'Recall': f"{recall_score(labels, preds):.4f}",
            'F1-Score': f"{f1_score(labels, preds):.4f}",
            'ROC AUC': f"{roc_auc_val:.4f}",
            'MCC': f"{matthews_corrcoef(labels, preds):.4f}",
            "Cohen's K": f"{cohen_kappa_score(labels, preds):.4f}",
            'Brier': f"{brier_score_loss(labels, probs):.4f}",
            'Log Loss': f"{log_loss(labels, probs):.4f}",
            'KS Stat': f"{ks_results[name]['KS Statistic']:.4f}",
            'Gini': f"{ks_results[name]['Gini']:.4f}",
            'ECE': f"{ece_results[name]:.4f}",
            "Youden's J": f"{youden_results[name]['Youden J']:.4f}",
            'Thresh*': f"{youden_results[name]['Optimal Threshold']:.4f}",
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index('Model')
    print(summary_df.to_string())

    # Coin-flip baseline
    print(f"\n  Coin-Flip Baseline Reference:")
    print(f"    Accuracy=0.50 | ROC AUC=0.50 | MCC=0.00 | K=0.00")
    print(f"    Brier=0.25    | Log Loss=0.693 | KS=0.00 | Gini=0.00")

    print(f"\n  Generated Diagnostic Plots:")
    print(f"    -> assets/calibration_and_confidence.png")
    print(f"    -> assets/ks_gains_lift.png")
    print(f"    -> assets/psi_distribution_shift.png")
    print(f"    -> assets/dca_youden.png")
    print("\n" + "=" * 70)

    return summary_df, psi_results


if __name__ == "__main__":
    run_deep_evaluation()
