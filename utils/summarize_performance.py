#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize CCL-CGI performance CSVs (Python 3.6+).

Usage:
  python utils/summarize_performance.py --models MODEL_A MODEL_B
  python utils/summarize_performance.py --latest 10
  python utils/summarize_performance.py --pattern R14_CCL_
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LOG_DIR = Path("log")
PREFIX = "CCL_CGI_performance_"

def _to_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "none" or value.lower() == "nan":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "none" or value.lower() == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_bool(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    if value in ("", "none", "nan"):
        return None
    return None


def _latest_row(path):
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
        return last


def _strip_prefix_suffix(filename, prefix, suffix):
    if filename.startswith(prefix):
        filename = filename[len(prefix) :]
    if filename.endswith(suffix):
        filename = filename[: -len(suffix)]
    return filename


def summarize_file(path):
    row = _latest_row(path)
    if not row:
        return None
    model_name = row.get("model_name") or _strip_prefix_suffix(path.name, PREFIX, ".csv")
    return {
        "path": path,
        "model_name": model_name,
        "n_folds": _to_int(row.get("n_folds")),
        "aupr_mean": _to_float(row.get("aupr_mean")),
        "aupr_std": _to_float(row.get("aupr_std")),
        "acc_mean": _to_float(row.get("acc_mean")),
        "acc_std": _to_float(row.get("acc_std")),
        "mcc_mean": _to_float(row.get("mcc_mean")),
        "mcc_std": _to_float(row.get("mcc_std")),
        "auc_mean": _to_float(row.get("auc_mean")),
        "auc_std": _to_float(row.get("auc_std")),
        "use_state_features": _to_bool(row.get("use_state_features")),
        "use_cancer_ppi": _to_bool(row.get("use_cancer_ppi")),
        "normalize_state_features": _to_bool(row.get("normalize_state_features")),
        "n_cell_types": _to_int(row.get("n_cell_types")),
        "num_heads": _to_int(row.get("num_heads")),
    }


def iter_perf_files(log_dir):
    if not log_dir.exists():
        return []
    for p in log_dir.glob(f"{PREFIX}*.csv"):
        if p.is_file():
            yield p


def pick_latest(files, n):
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def format_opt(x):
    return "NA" if x is None else f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="Model names (maps to log/CCL_CGI_performance_<model>.csv).")
    parser.add_argument("--pattern", type=str, default=None, help="Substring filter over filename/model_name.")
    parser.add_argument("--latest", type=int, default=0, help="Show N latest performance files.")
    parser.add_argument("--log_dir", type=str, default=str(LOG_DIR), help="Log directory (default: log).")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    selected = []
    if args.models:
        for model in args.models:
            selected.append(log_dir / f"{PREFIX}{model}.csv")
    else:
        all_files = list(iter_perf_files(log_dir))
        if args.pattern:
            pat = args.pattern
            all_files = [p for p in all_files if pat in p.name]
        if args.latest and args.latest > 0:
            selected = pick_latest(all_files, args.latest)
        else:
            selected = pick_latest(all_files, 20)

    summaries = []
    for p in selected:
        s = summarize_file(p)
        if s is not None:
            if args.pattern and args.models:
                if args.pattern not in s["model_name"] and args.pattern not in p.name:
                    continue
            summaries.append(s)

    if not summaries:
        print("No performance summaries found.")
        return

    headers = [
        "model_name",
        "n_folds",
        "aupr_mean±std",
        "acc_mean±std",
        "mcc_mean±std",
        "auc_mean±std",
        "state",
        "cancer_ppi",
        "norm_state",
        "n_cell_types",
        "num_heads",
        "file",
    ]
    print("\t".join(headers))
    for s in summaries:
        print(
            "\t".join(
                [
                    s["model_name"],
                    str(s["n_folds"]) if s["n_folds"] is not None else "NA",
                    f"{format_opt(s['aupr_mean'])}±{format_opt(s['aupr_std'])}",
                    f"{format_opt(s['acc_mean'])}±{format_opt(s['acc_std'])}",
                    f"{format_opt(s['mcc_mean'])}±{format_opt(s['mcc_std'])}",
                    f"{format_opt(s['auc_mean'])}±{format_opt(s['auc_std'])}",
                    "1" if s["use_state_features"] else "0" if s["use_state_features"] is not None else "NA",
                    "1" if s["use_cancer_ppi"] else "0" if s["use_cancer_ppi"] is not None else "NA",
                    "1" if s["normalize_state_features"] else "0" if s["normalize_state_features"] is not None else "NA",
                    str(s["n_cell_types"]) if s["n_cell_types"] is not None else "NA",
                    str(s["num_heads"]) if s["num_heads"] is not None else "NA",
                    str(s["path"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
