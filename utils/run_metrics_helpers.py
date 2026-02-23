import os

import numpy as np
import pandas as pd
from scipy.stats import t

from utils.io import format_filename
from utils.statistical_tests import (
    compute_cv_metrics_with_delong_and_bootstrap,
    format_metric_with_ci,
)


def init_metric_containers() -> tuple[dict, dict]:
    temp = {
        "dataset": "ALL",
        "avg_auc": 0.0,
        "avg_acc": 0.0,
        "avg_aupr": 0.0,
        "avg_mcc": 0.0,
        "avg_brier": 0.0,
        "avg_ece": 0.0,
        "avg_optimal_threshold": 0.0,
        "avg_optimal_precision": 0.0,
        "avg_optimal_recall": 0.0,
        "avg_optimal_f1": 0.0,
    }
    results = {
        "auc": [],
        "aupr": [],
        "acc": [],
        "mcc": [],
        "brier": [],
        "ece": [],
        "optimal_threshold": [],
        "optimal_precision": [],
        "optimal_recall": [],
        "optimal_f1": [],
        "y_true_folds": [],
        "y_pred_proba_folds": [],
    }
    return temp, results


def collect_fold_metrics(results: dict, test_metrics: dict | None) -> None:
    if not test_metrics:
        return

    if "metrics" in test_metrics:
        metrics = test_metrics["metrics"]
        if test_metrics.get("y_true") is not None:
            results["y_true_folds"].append(test_metrics["y_true"])
        if test_metrics.get("y_pred_proba") is not None:
            results["y_pred_proba_folds"].append(test_metrics["y_pred_proba"])
    else:
        metrics = test_metrics

    results["auc"].append(metrics.get("auc", 0))
    results["aupr"].append(metrics.get("aupr", 0))
    results["acc"].append(metrics.get("acc", 0))
    results["mcc"].append(metrics.get("mcc", 0))

    if metrics.get("brier") is not None:
        results["brier"].append(metrics.get("brier"))
    if metrics.get("ece") is not None:
        results["ece"].append(metrics.get("ece"))

    if metrics.get("optimal_threshold") is not None:
        results["optimal_threshold"].append(metrics.get("optimal_threshold"))
    if metrics.get("optimal_precision") is not None:
        results["optimal_precision"].append(metrics.get("optimal_precision"))
    if metrics.get("optimal_recall") is not None:
        results["optimal_recall"].append(metrics.get("optimal_recall"))
    if metrics.get("optimal_f1") is not None:
        results["optimal_f1"].append(metrics.get("optimal_f1"))


def summarize_metric_results(results: dict, temp: dict) -> tuple[dict, dict, dict]:
    macro_avg = {
        "auc": np.mean(results["auc"]) if results["auc"] else None,
        "aupr": np.mean(results["aupr"]) if results["aupr"] else None,
        "acc": np.mean(results["acc"]) if results["acc"] else None,
        "mcc": np.mean(results["mcc"]) if results["mcc"] else None,
    }
    macro_std = {
        "auc": np.std(results["auc"], ddof=1) if len(results["auc"]) > 1 else 0.0,
        "aupr": np.std(results["aupr"], ddof=1) if len(results["aupr"]) > 1 else 0.0,
        "acc": np.std(results["acc"], ddof=1) if len(results["acc"]) > 1 else 0.0,
        "mcc": np.std(results["mcc"], ddof=1) if len(results["mcc"]) > 1 else 0.0,
    }

    if len(results["y_true_folds"]) > 0 and len(results["y_pred_proba_folds"]) > 0:
        print("\n📊 Computing confidence intervals (DeLong for AUC, Bootstrap n=1000 for others)...")

        statistical_results = compute_cv_metrics_with_delong_and_bootstrap(
            all_y_true=results["y_true_folds"],
            all_y_pred_proba=results["y_pred_proba_folds"],
        )

        temp["avg_auc"] = statistical_results["auc"]["mean"]
        temp["auc_ci_lower"] = statistical_results["auc"]["ci_lower"]
        temp["auc_ci_upper"] = statistical_results["auc"]["ci_upper"]
        temp["auc_ci_method"] = "DeLong"

        temp["avg_aupr"] = statistical_results["aupr"]["mean"]
        temp["aupr_ci_lower"] = statistical_results["aupr"]["ci_lower"]
        temp["aupr_ci_upper"] = statistical_results["aupr"]["ci_upper"]
        temp["aupr_ci_method"] = "Bootstrap"

        temp["avg_acc"] = statistical_results["acc"]["mean"]
        temp["acc_ci_lower"] = statistical_results["acc"]["ci_lower"]
        temp["acc_ci_upper"] = statistical_results["acc"]["ci_upper"]

        temp["avg_mcc"] = statistical_results["mcc"]["mean"]
        temp["mcc_ci_lower"] = statistical_results["mcc"]["ci_lower"]
        temp["mcc_ci_upper"] = statistical_results["mcc"]["ci_upper"]

        temp["avg_precision"] = statistical_results["precision"]["mean"]
        temp["precision_ci_lower"] = statistical_results["precision"]["ci_lower"]
        temp["precision_ci_upper"] = statistical_results["precision"]["ci_upper"]

        temp["avg_recall"] = statistical_results["recall"]["mean"]
        temp["recall_ci_lower"] = statistical_results["recall"]["ci_lower"]
        temp["recall_ci_upper"] = statistical_results["recall"]["ci_upper"]

        temp["avg_f1"] = statistical_results["f1"]["mean"]
        temp["f1_ci_lower"] = statistical_results["f1"]["ci_lower"]
        temp["f1_ci_upper"] = statistical_results["f1"]["ci_upper"]

        print("✅ Confidence interval computation completed")
        print(
            f"   AUC: {temp['avg_auc']:.4f} "
            f"(95% CI: {temp['auc_ci_lower']:.4f}-{temp['auc_ci_upper']:.4f}, DeLong)"
        )
        print(
            f"   AUPR: {temp['avg_aupr']:.4f} "
            f"(95% CI: {temp['aupr_ci_lower']:.4f}-{temp['aupr_ci_upper']:.4f}, Bootstrap)"
        )
    else:
        print("\n⚠️  Raw prediction data not collected; using traditional t-distribution confidence intervals")
        temp["avg_auc"] = np.mean(results["auc"])
        temp["avg_acc"] = np.mean(results["acc"])
        temp["avg_aupr"] = np.mean(results["aupr"])
        temp["avg_mcc"] = np.mean(results["mcc"])
        auc_std = np.std(results["auc"])
        aupr_std = np.std(results["aupr"])

        confidence = 0.95
        n = len(results["auc"])
        t_crit = t.ppf((1 + confidence) / 2, n - 1)

        temp["auc_ci_lower"] = temp["avg_auc"] - t_crit * auc_std / np.sqrt(n)
        temp["auc_ci_upper"] = temp["avg_auc"] + t_crit * auc_std / np.sqrt(n)
        temp["auc_ci_method"] = "t-distribution"

        temp["aupr_ci_lower"] = temp["avg_aupr"] - t_crit * aupr_std / np.sqrt(n)
        temp["aupr_ci_upper"] = temp["avg_aupr"] + t_crit * aupr_std / np.sqrt(n)
        temp["aupr_ci_method"] = "t-distribution"

    if len(results["brier"]) > 0:
        temp["avg_brier"] = np.mean(results["brier"])
        temp["brier_std"] = np.std(results["brier"])
    if len(results["ece"]) > 0:
        temp["avg_ece"] = np.mean(results["ece"])
        temp["ece_std"] = np.std(results["ece"])

    if len(results["optimal_threshold"]) > 0:
        temp["avg_optimal_threshold"] = np.mean(results["optimal_threshold"])
        temp["avg_optimal_precision"] = np.mean(results["optimal_precision"])
        temp["avg_optimal_recall"] = np.mean(results["optimal_recall"])
        temp["avg_optimal_f1"] = np.mean(results["optimal_f1"])
        temp["optimal_precision_std"] = np.std(results["optimal_precision"])
        temp["optimal_recall_std"] = np.std(results["optimal_recall"])

    return macro_avg, macro_std, temp


def print_metric_report(
    temp: dict,
    results: dict,
    macro_avg: dict,
    macro_std: dict,
    *,
    subset_mode: bool,
    fold_indices,
    selected_folds: list[int],
    actual_folds: int,
    cv_folds: int,
) -> None:
    print(f"\n{'='*80}")
    if subset_mode:
        if fold_indices is not None:
            print(f"🔧 Subset Run Results - folds {selected_folds} (NOT final results):")
        else:
            print(f"🔧 TEST RUN Results - first {actual_folds} fold(s) (NOT final results):")
    else:
        print(f"Logging Info - {cv_folds}-fold Cross-Validation Results:")
    print(f"{'='*80}")

    if "auc_ci_method" in temp:
        print(
            f"AUC:  {format_metric_with_ci(temp['avg_auc'], temp['auc_ci_lower'], temp['auc_ci_upper'])} "
            f"({temp.get('auc_ci_method', 'unknown')})"
        )
    else:
        print(f"AUC:  {temp['avg_auc']:.4f}")

    if "aupr_ci_method" in temp:
        print(
            f"AUPR: {format_metric_with_ci(temp['avg_aupr'], temp['aupr_ci_lower'], temp['aupr_ci_upper'])} "
            f"({temp.get('aupr_ci_method', 'unknown')})"
        )
    else:
        print(f"AUPR: {temp['avg_aupr']:.4f}")

    if "acc_ci_lower" in temp:
        print(f"ACC:  {format_metric_with_ci(temp['avg_acc'], temp['acc_ci_lower'], temp['acc_ci_upper'])} (Bootstrap)")
    else:
        print(f"ACC:  {temp['avg_acc']:.4f}")

    if "mcc_ci_lower" in temp:
        print(f"MCC:  {format_metric_with_ci(temp['avg_mcc'], temp['mcc_ci_lower'], temp['mcc_ci_upper'])} (Bootstrap)")
    else:
        print(f"MCC:  {temp['avg_mcc']:.4f}")

    if "precision_ci_lower" in temp:
        print(
            f"Precision: {format_metric_with_ci(temp['avg_precision'], temp['precision_ci_lower'], temp['precision_ci_upper'])} "
            f"(Bootstrap)"
        )
    if "recall_ci_lower" in temp:
        print(
            f"Recall:    {format_metric_with_ci(temp['avg_recall'], temp['recall_ci_lower'], temp['recall_ci_upper'])} "
            f"(Bootstrap)"
        )
    if "f1_ci_lower" in temp:
        print(f"F1:        {format_metric_with_ci(temp['avg_f1'], temp['f1_ci_lower'], temp['f1_ci_upper'])} (Bootstrap)")

    if "avg_brier" in temp and temp["avg_brier"] > 0:
        print("\nCalibration Metrics:")
        if "brier_ci_lower" in temp:
            print(f"Brier Score: {format_metric_with_ci(temp['avg_brier'], temp['brier_ci_lower'], temp['brier_ci_upper'])}")
        else:
            print(f"Brier Score: {temp['avg_brier']:.4f}")

        if "ece_ci_lower" in temp:
            print(f"ECE:         {format_metric_with_ci(temp['avg_ece'], temp['ece_ci_lower'], temp['ece_ci_upper'])}")
        else:
            print(f"ECE:         {temp['avg_ece']:.4f}")

    if "avg_optimal_threshold" in temp and temp["avg_optimal_threshold"] > 0:
        print("\nOptimal Operating Point (F1-optimized on test):")
        print(f"Threshold: {temp['avg_optimal_threshold']:.3f} ± {np.std(results['optimal_threshold']):.3f}")
        print(f"Precision: {temp['avg_optimal_precision']:.4f} ± {temp.get('optimal_precision_std', 0):.4f}")
        print(f"Recall:    {temp['avg_optimal_recall']:.4f} ± {temp.get('optimal_recall_std', 0):.4f}")
        print(f"F1-Score:  {temp['avg_optimal_f1']:.4f}")

    if subset_mode:
        print(f"\n{'='*80}")
        if fold_indices is not None:
            print(f"⚠️  This was a subset run for folds {selected_folds}.")
            print(f"    Run all {cv_folds} folds (or unset fold_idx) for final results.")
        else:
            print(f"⚠️  This was a TEST RUN with only {actual_folds} fold(s).")
            print(f"    For final results, set test_run=None to run all {cv_folds} folds.")
        print(f"{'='*80}")

    print(f"{'='*80}\n")

    legacy_macro = ""
    if macro_avg["auc"] is not None:
        legacy_macro = (
            f"(macro avg±std) auc: {macro_avg['auc']:.4f}±{macro_std['auc']:.4f}, "
            f"acc: {macro_avg['acc']:.4f}±{macro_std['acc']:.4f}, "
            f"aupr: {macro_avg['aupr']:.4f}±{macro_std['aupr']:.4f}, "
            f"mcc: {macro_avg['mcc']:.4f}±{macro_std['mcc']:.4f}"
        )
    print(
        f"[Legacy Format] avg_auc: {temp['avg_auc']:.4f}, avg_acc: {temp['avg_acc']:.4f}, "
        f"avg_aupr: {temp['avg_aupr']:.4f}, avg_mcc: {temp['avg_mcc']:.4f} {legacy_macro}"
    )


def build_performance_log(
    *,
    context: dict,
    temp: dict,
    results: dict,
    macro_avg: dict,
    macro_std: dict,
) -> dict:
    return {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": context["model_name"],
        "is_test_run": context["subset_mode"],
        "n_folds": context["actual_folds"],
        "cv_folds_total": context["cv_folds"],
        "base_n_cell_types": context["base_n_cell_types"],
        "effective_n_cell_types": context["n_cell_types"],
        "excluded_cell_types": ",".join(context["excluded_cell_types"]) if context["excluded_cell_types"] else "",
        "n_graphs": context["n_graphs"],
        "n_neighbors": context["n_neighbors"],
        "n_layers": context["n_layers"],
        "d_model": context["embed_dim"],
        "dff": context["dff"],
        "num_heads": context["num_heads"],
        "d_sp_enc": context["d_sp_enc"],
        "p": context["p"],
        "q": context["q"],
        "learning_rate": context["lr"],
        "dropout": context["dropout"],
        "loss_mul": context["loss_mul"],
        "batch_size": context["bz"],
        "l2_weight": context["l2_weights"],
        "optimizer": "adam",
        "max_epochs": 200,
        "spatial_encoding": context["spatial"],
        "use_64d_features": context["use_64d_features"],
        "normalize_state_features": context["normalize_state_features"] if context["use_64d_features"] else False,
        "sanitize_features": context["sanitize_features"],
        "use_cancer_ppi": context["use_cancer_ppi"],
        "feature_dim": context["embed_dim"],
        "feature_subset": context["feature_subset"],
        "auc_mean": temp["avg_auc"],
        "aupr_mean": temp["avg_aupr"],
        "acc_mean": temp["avg_acc"],
        "mcc_mean": temp["avg_mcc"],
        "auc_fold_mean": macro_avg["auc"],
        "aupr_fold_mean": macro_avg["aupr"],
        "acc_fold_mean": macro_avg["acc"],
        "mcc_fold_mean": macro_avg["mcc"],
        "auc_fold_std": macro_std["auc"],
        "aupr_fold_std": macro_std["aupr"],
        "acc_fold_std": macro_std["acc"],
        "mcc_fold_std": macro_std["mcc"],
        "auc_ci_lower": temp.get("auc_ci_lower", None),
        "auc_ci_upper": temp.get("auc_ci_upper", None),
        "aupr_ci_lower": temp.get("aupr_ci_lower", None),
        "aupr_ci_upper": temp.get("aupr_ci_upper", None),
        "acc_ci_lower": temp.get("acc_ci_lower", None),
        "acc_ci_upper": temp.get("acc_ci_upper", None),
        "mcc_ci_lower": temp.get("mcc_ci_lower", None),
        "mcc_ci_upper": temp.get("mcc_ci_upper", None),
        "brier_mean": temp.get("avg_brier", None),
        "brier_std": temp.get("brier_std", None),
        "brier_ci_lower": temp.get("brier_ci_lower", None),
        "brier_ci_upper": temp.get("brier_ci_upper", None),
        "ece_mean": temp.get("avg_ece", None),
        "ece_std": temp.get("ece_std", None),
        "ece_ci_lower": temp.get("ece_ci_lower", None),
        "ece_ci_upper": temp.get("ece_ci_upper", None),
        "optimal_threshold_mean": temp.get("avg_optimal_threshold", None),
        "optimal_threshold_std": np.std(results["optimal_threshold"]) if len(results["optimal_threshold"]) > 0 else None,
        "optimal_precision_mean": temp.get("avg_optimal_precision", None),
        "optimal_precision_std": temp.get("optimal_precision_std", None),
        "optimal_recall_mean": temp.get("avg_optimal_recall", None),
        "optimal_recall_std": temp.get("optimal_recall_std", None),
        "optimal_f1_mean": temp.get("avg_optimal_f1", None),
    }


def write_performance_csv(performance_log: dict, *, model_name: str, log_dir: str) -> str:
    performance_csv = format_filename(log_dir, f"CCL_CGI_performance_{model_name}.csv")
    df = pd.DataFrame([performance_log])

    if os.path.exists(performance_csv):
        df.to_csv(performance_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(performance_csv, mode="w", header=True, index=False)

    print(f"\n✅ Performance log written to: {performance_csv}")
    return performance_csv
