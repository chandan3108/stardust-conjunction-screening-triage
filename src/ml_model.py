"""
ml_model.py — ML Pre-Filter Training & Inference

Trains LightGBM and XGBoost models with asymmetric loss functions
optimized for ultra-high recall in conjunction screening.

Key design:
- Asymmetric loss: c_FN = 50 × c_FP (missing a threat costs 50x more)
- Neyman-Pearson threshold: fix Recall ≥ 99.9%, maximize Precision
- Safety-critical: zero missed threats is the #1 priority
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, classification_report,
    fbeta_score, average_precision_score, confusion_matrix,
    roc_auc_score
)
from typing import Tuple, Dict, Optional

from config import (
    FEATURE_COLUMNS, ML_TARGET_RECALL,
    ML_C_FN, ML_C_FP, ML_DEFAULT_THRESHOLD, MODEL_DIR
)


# ============================================================
# LightGBM Training
# ============================================================

def train_lightgbm(
    df: pd.DataFrame,
    target_recall: float = ML_TARGET_RECALL,
    c_fn: float = ML_C_FN,
    c_fp: float = ML_C_FP,
) -> Tuple[lgb.Booster, float]:
    """
    Train LightGBM with asymmetric loss for conjunction screening.

    Args:
        df: Training DataFrame with FEATURE_COLUMNS + 'label'
        target_recall: Minimum required recall (default 0.999)
        c_fn: False negative cost multiplier (default 50)
        c_fp: False positive cost multiplier (default 1)

    Returns:
        (trained_model, optimal_threshold)
    """
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df['label'].values.astype(np.float64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=FEATURE_COLUMNS)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                       feature_name=FEATURE_COLUMNS)

    # Custom asymmetric loss
    def asymmetric_objective(preds, train_data):
        labels = train_data.get_label()
        probs = 1.0 / (1.0 + np.exp(-preds))
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        grad = c_fp * (1.0 - labels) * probs - c_fn * labels * (1.0 - probs)
        hess = (c_fp * (1.0 - labels) + c_fn * labels) * probs * (1.0 - probs)
        return grad, hess

    def asymmetric_eval(preds, train_data):
        labels = train_data.get_label()
        probs = 1.0 / (1.0 + np.exp(-preds))
        preds_bin = (probs >= 0.5).astype(int)
        fn = np.sum((labels == 1) & (preds_bin == 0))
        fp = np.sum((labels == 0) & (preds_bin == 1))
        cost = c_fn * fn + c_fp * fp
        return 'asym_cost', cost, False

    params = {
        'objective': asymmetric_objective,
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    print("\n[ML Model] Training LightGBM with asymmetric loss...")
    print(f"  c_FN = {c_fn}, c_FP = {c_fp} (ratio: {c_fn/c_fp:.0f}:1)")
    print(f"  Target recall: {target_recall*100:.1f}%")

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        feval=asymmetric_eval,
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(100),
        ]
    )

    # Predict on validation set
    raw_preds = model.predict(X_val)
    val_probs = 1.0 / (1.0 + np.exp(-raw_preds))

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(
        y_val, val_probs, target_recall
    )

    # Print evaluation
    print_evaluation_report(y_val, val_probs, optimal_threshold)

    # Save model and config
    save_model(model, optimal_threshold, target_recall, c_fn, c_fp)

    return model, optimal_threshold


# ============================================================
# XGBoost Training (Comparison Baseline)
# ============================================================

def train_xgboost(
    df: pd.DataFrame,
    target_recall: float = ML_TARGET_RECALL,
    c_fn: float = ML_C_FN,
    c_fp: float = ML_C_FP,
) -> Tuple[xgb.Booster, float]:
    """Train XGBoost with asymmetric loss as comparison baseline."""
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df['label'].values.astype(np.float64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train,
                         feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val,
                       feature_names=FEATURE_COLUMNS)

    def xgb_asymmetric_obj(preds, dtrain):
        labels = dtrain.get_label()
        probs = 1.0 / (1.0 + np.exp(-preds))
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        grad = c_fp * (1.0 - labels) * probs - c_fn * labels * (1.0 - probs)
        hess = (c_fp * (1.0 - labels) + c_fn * labels) * probs * (1.0 - probs)
        return grad, hess

    params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'max_delta_step': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'eval_metric': 'aucpr',
    }

    print("\n[ML Model] Training XGBoost baseline...")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        obj=xgb_asymmetric_obj,
        early_stopping_rounds=30,
        verbose_eval=100,
    )

    raw_preds = model.predict(dval, output_margin=True)
    val_probs = 1.0 / (1.0 + np.exp(-raw_preds))

    optimal_threshold = find_optimal_threshold(
        y_val, val_probs, target_recall
    )

    print_evaluation_report(y_val, val_probs, optimal_threshold, name="XGBoost")

    # Save XGBoost model
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "stardust_xgb.json"))

    return model, optimal_threshold


# ============================================================
# Threshold Optimization (Neyman-Pearson Criterion)
# ============================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_recall: float = ML_TARGET_RECALL
) -> float:
    """
    Find the highest threshold that achieves >= target_recall.
    Maximizes precision subject to the recall constraint.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Find all thresholds where recall >= target
    valid = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid) > 0:
        best_idx = valid[np.argmax(precisions[valid])]
        return float(thresholds[best_idx])
    else:
        print(f"  WARNING: Could not achieve {target_recall*100}% recall. "
              f"Max recall: {recalls.max()*100:.1f}%")
        # Return threshold that gives highest recall
        best_recall_idx = np.argmax(recalls[:-1])
        return float(thresholds[best_recall_idx])


# ============================================================
# Evaluation
# ============================================================

def print_evaluation_report(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    name: str = "LightGBM"
):
    """Print comprehensive evaluation metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n{'=' * 60}")
    print(f"STARDUST {name} MODEL EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"Decision Threshold: {threshold:.6f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Safe', 'Threat'])}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix:")
    print(f"  True Positives  (threats caught):  {tp}")
    print(f"  False Positives (false alarms):    {fp}")
    print(f"  True Negatives  (safe confirmed):  {tn}")
    print(f"  False Negatives (MISSED THREATS):  {fn}")

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f2 = fbeta_score(y_true, y_pred, beta=2)
    f5 = fbeta_score(y_true, y_pred, beta=5)
    pr_auc = average_precision_score(y_true, y_probs)

    try:
        roc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc = 0.0

    print(f"\nKey Metrics:")
    print(f"  Recall (sensitivity):    {recall*100:.3f}%")
    print(f"  Precision:               {precision*100:.3f}%")
    print(f"  F2 Score (2x recall):    {f2:.4f}")
    print(f"  F5 Score (5x recall):    {f5:.4f}")
    print(f"  PR-AUC:                  {pr_auc:.4f}")
    print(f"  ROC-AUC:                 {roc:.4f}")

    if fn > 0:
        print(f"\n  ⚠️  WARNING: {fn} THREATS MISSED!")
    else:
        print(f"\n  ✅ ZERO MISSED THREATS — all positives caught")
    print(f"{'=' * 60}")


# ============================================================
# Save / Load
# ============================================================

def save_model(
    model: lgb.Booster,
    threshold: float,
    target_recall: float,
    c_fn: float,
    c_fp: float,
):
    """Save model, threshold, and feature names."""
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_dir / "stardust_lgbm.json"))

    with open(model_dir / "threshold.json", 'w') as f:
        json.dump({
            "optimal_threshold": threshold,
            "target_recall": target_recall,
            "c_fn": c_fn,
            "c_fp": c_fp,
        }, f, indent=2)

    with open(model_dir / "feature_names.json", 'w') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"\n[ML Model] Saved model to {model_dir}/")


def load_model(
    model_path: str = None,
    threshold_path: str = None
) -> Tuple[lgb.Booster, float]:
    """
    Load trained model and threshold config.

    Returns:
        (model, threshold)
    """
    model_dir = Path(MODEL_DIR)
    if model_path is None:
        model_path = str(model_dir / "stardust_lgbm.json")
    if threshold_path is None:
        threshold_path = str(model_dir / "threshold.json")

    model = lgb.Booster(model_file=model_path)

    with open(threshold_path, 'r') as f:
        config = json.load(f)
    threshold = config['optimal_threshold']

    return model, threshold


def predict(
    model: lgb.Booster,
    features: np.ndarray,
    threshold: float = ML_DEFAULT_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on new features.

    Args:
        model: Trained LightGBM Booster
        features: Feature array, shape (n_samples, 28)
        threshold: Decision threshold

    Returns:
        (probabilities, binary_predictions)
    """
    raw_preds = model.predict(features)
    probs = 1.0 / (1.0 + np.exp(-raw_preds))
    predictions = (probs >= threshold).astype(int)
    return probs, predictions


def get_feature_importance(model: lgb.Booster) -> pd.DataFrame:
    """Get feature importance from trained model."""
    importance = model.feature_importance(importance_type='gain')
    feat_names = model.feature_name()
    df = pd.DataFrame({
        'feature': feat_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
    return df
