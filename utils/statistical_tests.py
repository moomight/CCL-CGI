#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.utils import resample

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if len(np.unique(y_pred)) < 2:
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if metric == 'f1':
            if precision + recall == 0:
                score = 0
            else:
                score = 2 * precision * recall / (precision + recall)
        elif metric == 'precision':
            score = precision
        elif metric == 'recall':
            score = recall
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_score, best_precision, best_recall

def calculate_brier_score(y_true, y_pred_proba):
    
    return brier_score_loss(y_true, y_pred_proba)

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins[1:-1])

    ece = 0.0
    bin_info = []

    for i in range(n_bins):
        mask = bin_indices == i

        if np.sum(mask) == 0:
            continue

        bin_confidence = np.mean(y_pred_proba[mask])

        bin_accuracy = np.mean(y_true[mask])

        bin_count = np.sum(mask)

        ece += (bin_count / len(y_true)) * abs(bin_confidence - bin_accuracy)

        bin_info.append({
            'bin_id': i,
            'confidence': bin_confidence,
            'accuracy': bin_accuracy,
            'count': bin_count,
            'lower': bins[i],
            'upper': bins[i + 1]
        })

    return ece, bin_info

def calculate_calibration_metrics(y_true, y_pred_proba, n_bins=10):
    
    brier = calculate_brier_score(y_true, y_pred_proba)
    ece, bin_info = calculate_ece(y_true, y_pred_proba, n_bins)

    return {
        'brier_score': brier,
        'ece': ece,
        'ece_bins': bin_info
    }

def delong_test_auc_ci(y_true, y_pred_proba, alpha=0.05):
    
    from scipy.stats import norm

    auc = roc_auc_score(y_true, y_pred_proba)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    pos_indices = y_true == 1
    neg_indices = y_true == 0

    n_pos = np.sum(pos_indices)
    n_neg = np.sum(neg_indices)

    if n_pos == 0 or n_neg == 0:
        return auc, auc, auc, 0.0

    pos_scores = y_pred_proba[pos_indices]
    neg_scores = y_pred_proba[neg_indices]

    V10 = np.zeros(n_pos)
    for i, pos_score in enumerate(pos_scores):
        V10[i] = np.mean(pos_score > neg_scores) + 0.5 * np.mean(pos_score == neg_scores)

    V01 = np.zeros(n_neg)
    for i, neg_score in enumerate(neg_scores):
        V01[i] = np.mean(pos_scores > neg_score) + 0.5 * np.mean(pos_scores == neg_score)

    S10 = np.var(V10, ddof=1) if n_pos > 1 else 0
    S01 = np.var(V01, ddof=1) if n_neg > 1 else 0

    se = np.sqrt((S10 / n_pos) + (S01 / n_neg))

    if se == 0:
        return auc, auc, auc, 0.0

    z_critical = norm.ppf(1 - alpha / 2)
    ci_lower = auc - z_critical * se
    ci_upper = auc + z_critical * se

    ci_lower = max(0, min(1, ci_lower))
    ci_upper = max(0, min(1, ci_upper))

    return auc, ci_lower, ci_upper, se

def delong_test(y_true_1, y_pred_1, y_true_2, y_pred_2):
    
    from scipy.stats import norm

    auc1 = roc_auc_score(y_true_1, y_pred_1)
    auc2 = roc_auc_score(y_true_2, y_pred_2)

    n1 = len(y_true_1)
    n2 = len(y_true_2)

    var1 = auc1 * (1 - auc1) / n1
    var2 = auc2 * (1 - auc2) / n2

    se = np.sqrt(var1 + var2)

    if se == 0:
        return 0, 1.0, auc1 - auc2

    z_score = (auc1 - auc2) / se
    p_value = 2 * (1 - norm.cdf(abs(z_score))) 

    return z_score, p_value, auc1 - auc2

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95, metric_name='metric'):
    
    bootstrapped_scores = []

    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))

        if len(np.unique(y_true[indices])) < 2:
            continue

        try:
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except:
            continue

    bootstrapped_scores = np.array(bootstrapped_scores)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrapped_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
    mean = np.mean(bootstrapped_scores)

    return mean, ci_lower, ci_upper, bootstrapped_scores

def calculate_metrics_with_ci(y_true, y_pred_proba, y_pred_class=None, n_bootstraps=1000):
    
    if y_pred_class is None:
        y_pred_class = (y_pred_proba > 0.5).astype(int)

    results = {}

    # AUC
    auc_mean, auc_lower, auc_upper, auc_scores = bootstrap_ci(
        y_true, y_pred_proba, roc_auc_score, n_bootstraps, metric_name='AUC'
    )
    results['auc'] = {
        'mean': auc_mean,
        'ci_lower': auc_lower,
        'ci_upper': auc_upper,
        'scores': auc_scores
    }

    # AUPR
    aupr_mean, aupr_lower, aupr_upper, aupr_scores = bootstrap_ci(
        y_true, y_pred_proba, average_precision_score, n_bootstraps, metric_name='AUPR'
    )
    results['aupr'] = {
        'mean': aupr_mean,
        'ci_lower': aupr_lower,
        'ci_upper': aupr_upper,
        'scores': aupr_scores
    }

    # Accuracy
    acc_mean, acc_lower, acc_upper, acc_scores = bootstrap_ci(
        y_true, y_pred_class, accuracy_score, n_bootstraps, metric_name='Accuracy'
    )
    results['acc'] = {
        'mean': acc_mean,
        'ci_lower': acc_lower,
        'ci_upper': acc_upper,
        'scores': acc_scores
    }

    return results

def format_metric_with_ci(mean, ci_lower, ci_upper, precision=3):
    
    fmt = f"{{:.{precision}f}}"
    mean_str = fmt.format(mean)
    lower_str = fmt.format(ci_lower)
    upper_str = fmt.format(ci_upper)

    return f"{mean_str} (95% CI: [{lower_str}, {upper_str}])"

def compute_cv_metrics_with_ci(all_y_true, all_y_pred_proba, all_y_pred_class=None, n_bootstraps=1000):
    
    n_folds = len(all_y_true)

    fold_aucs = []
    fold_auprs = []
    fold_accs = []

    for i in range(n_folds):
        y_true = all_y_true[i]
        y_pred_proba = all_y_pred_proba[i]
        y_pred_class = all_y_pred_class[i] if all_y_pred_class else (y_pred_proba > 0.5).astype(int)

        fold_aucs.append(roc_auc_score(y_true, y_pred_proba))
        fold_auprs.append(average_precision_score(y_true, y_pred_proba))
        fold_accs.append(accuracy_score(y_true, y_pred_class))

    auc_mean = np.mean(fold_aucs)
    auc_std = np.std(fold_aucs)
    aupr_mean = np.mean(fold_auprs)
    aupr_std = np.std(fold_auprs)
    acc_mean = np.mean(fold_accs)
    acc_std = np.std(fold_accs)

    from scipy.stats import t
    confidence = 0.95
    n = n_folds

    t_crit = t.ppf((1 + confidence) / 2, n - 1)

    auc_ci_lower = auc_mean - t_crit * auc_std / np.sqrt(n)
    auc_ci_upper = auc_mean + t_crit * auc_std / np.sqrt(n)

    aupr_ci_lower = aupr_mean - t_crit * aupr_std / np.sqrt(n)
    aupr_ci_upper = aupr_mean + t_crit * aupr_std / np.sqrt(n)

    acc_ci_lower = acc_mean - t_crit * acc_std / np.sqrt(n)
    acc_ci_upper = acc_mean + t_crit * acc_std / np.sqrt(n)

    summary = {
        'auc': {
            'mean': auc_mean,
            'std': auc_std,
            'ci_lower': auc_ci_lower,
            'ci_upper': auc_ci_upper,
            'formatted': format_metric_with_ci(auc_mean, auc_ci_lower, auc_ci_upper)
        },
        'aupr': {
            'mean': aupr_mean,
            'std': aupr_std,
            'ci_lower': aupr_ci_lower,
            'ci_upper': aupr_ci_upper,
            'formatted': format_metric_with_ci(aupr_mean, aupr_ci_lower, aupr_ci_upper)
        },
        'acc': {
            'mean': acc_mean,
            'std': acc_std,
            'ci_lower': acc_ci_lower,
            'ci_upper': acc_ci_upper,
            'formatted': format_metric_with_ci(acc_mean, acc_ci_lower, acc_ci_upper)
        }
    }

    return summary

def compute_cv_metrics_with_delong_and_bootstrap(all_y_true, all_y_pred_proba, n_bootstraps=1000):
    
    y_true_all = np.concatenate(all_y_true)
    y_pred_proba_all = np.concatenate(all_y_pred_proba)
    y_pred_class_all = (y_pred_proba_all > 0.5).astype(int)

    # ===== 1. AUC - DeLong =====
    auc, auc_ci_lower, auc_ci_upper, auc_se = delong_test_auc_ci(
        y_true_all, y_pred_proba_all, alpha=0.05
    )

    # ===== 2. AUPR - Bootstrap =====
    # AUPR
    aupr_mean, aupr_ci_lower, aupr_ci_upper, _ = bootstrap_ci(
        y_true_all, y_pred_proba_all,
        average_precision_score,
        n_bootstraps=n_bootstraps,
        metric_name='AUPR'
    )

    # Accuracy
    acc_mean, acc_ci_lower, acc_ci_upper, _ = bootstrap_ci(
        y_true_all, y_pred_class_all,
        accuracy_score,
        n_bootstraps=n_bootstraps,
        metric_name='Accuracy'
    )

    # Precision
    from sklearn.metrics import matthews_corrcoef
    def safe_precision(y_t, y_p):
        return precision_score(y_t, y_p, zero_division=0)

    prec_mean, prec_ci_lower, prec_ci_upper, _ = bootstrap_ci(
        y_true_all, y_pred_class_all,
        safe_precision,
        n_bootstraps=n_bootstraps,
        metric_name='Precision'
    )

    # Recall
    def safe_recall(y_t, y_p):
        return recall_score(y_t, y_p, zero_division=0)

    rec_mean, rec_ci_lower, rec_ci_upper, _ = bootstrap_ci(
        y_true_all, y_pred_class_all,
        safe_recall,
        n_bootstraps=n_bootstraps,
        metric_name='Recall'
    )

    # F1 Score
    def safe_f1(y_t, y_p):
        return f1_score(y_t, y_p, zero_division=0)

    f1_mean, f1_ci_lower, f1_ci_upper, _ = bootstrap_ci(
        y_true_all, y_pred_class_all,
        safe_f1,
        n_bootstraps=n_bootstraps,
        metric_name='F1'
    )

    # MCC
    def safe_mcc(y_t, y_p):
        try:
            return matthews_corrcoef(y_t, y_p)
        except:
            return 0.0

    mcc_mean, mcc_ci_lower, mcc_ci_upper, _ = bootstrap_ci(
        y_true_all, y_pred_class_all,
        safe_mcc,
        n_bootstraps=n_bootstraps,
        metric_name='MCC'
    )

    metrics = {
        'auc': {
            'mean': auc,
            'ci_lower': auc_ci_lower,
            'ci_upper': auc_ci_upper,
            'se': auc_se,
            'method': 'DeLong',
            'formatted': format_metric_with_ci(auc, auc_ci_lower, auc_ci_upper)
        },
        'aupr': {
            'mean': aupr_mean,
            'ci_lower': aupr_ci_lower,
            'ci_upper': aupr_ci_upper,
            'method': f'Bootstrap (n={n_bootstraps})',
            'formatted': format_metric_with_ci(aupr_mean, aupr_ci_lower, aupr_ci_upper)
        },
        'acc': {
            'mean': acc_mean,
            'ci_lower': acc_ci_lower,
            'ci_upper': acc_ci_upper,
            'method': f'Bootstrap (n={n_bootstraps})',
            'formatted': format_metric_with_ci(acc_mean, acc_ci_lower, acc_ci_upper)
        },
        'precision': {
            'mean': prec_mean,
            'ci_lower': prec_ci_lower,
            'ci_upper': prec_ci_upper,
            'method': f'Bootstrap (n={n_bootstraps})',
            'formatted': format_metric_with_ci(prec_mean, prec_ci_lower, prec_ci_upper)
        },
        'recall': {
            'mean': rec_mean,
            'ci_lower': rec_ci_lower,
            'ci_upper': rec_ci_upper,
            'method': f'Bootstrap (n={n_bootstraps})',
            'formatted': format_metric_with_ci(rec_mean, rec_ci_lower, rec_ci_upper)
        },
        'f1': {
            'mean': f1_mean,
            'ci_lower': f1_ci_lower,
            'ci_upper': f1_ci_upper,
            'method': f'Bootstrap (n={n_bootstraps})',
            'formatted': format_metric_with_ci(f1_mean, f1_ci_lower, f1_ci_upper)
        },
        'mcc': {
            'mean': mcc_mean,
            'ci_lower': mcc_ci_lower,
            'ci_upper': mcc_ci_upper,
            'method': f'Bootstrap (n={n_bootstraps})',
            'formatted': format_metric_with_ci(mcc_mean, mcc_ci_lower, mcc_ci_upper)
        }
    }

    return metrics

if __name__ == "__main__":
    print("=" * 80)

    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_pred1 = np.random.rand(n) * 0.6 + y_true * 0.3 
    y_pred2 = np.random.rand(n) * 0.5 + y_true * 0.4  

    # 1. Bootstrap
    print("\n1. Bootstrap :")
    results = calculate_metrics_with_ci(y_true, y_pred1)
    print(f"   AUC: {format_metric_with_ci(results['auc']['mean'], results['auc']['ci_lower'], results['auc']['ci_upper'])}")
    print(f"   AUPR: {format_metric_with_ci(results['aupr']['mean'], results['aupr']['ci_lower'], results['aupr']['ci_upper'])}")

    # 2. DeLong 
    print("\n2. DeLong:")
    z, p, diff = delong_test(y_true, y_pred1, y_true, y_pred2)
    print(f"   AUC: {diff:.4f}")
    print(f"   Z: {z:.4f}")
    print(f"   P: {p:.4f}")
    print(f"   Significant: {'Yes' if p < 0.05 else 'No'}")
