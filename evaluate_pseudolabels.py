#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_pseudolabels.py

- Évalue des pseudo-labels ternaires (oui / non / inconnu) vs un gold.
- INNER JOIN sur l'identifiant => ignore automatiquement les prédictions sans gold.
- Exporte CSVs + affiche un résumé console (macro + tops labels),
  incluant sensibilité et spécificité.

Nouveautés:
- Bootstrap (IC) sur l'unité "report" (resampling des lignes du merged):
  --bootstrap_n, --bootstrap_seed, --bootstrap_alpha
  => ajoute des colonnes *__ci_low / *__ci_high dans per_label et summary.
- Option --max_errors_per_label pour limiter le CSV d'erreurs.

Usage:
python evaluate_pseudolabels.py \
  --split val \
  --gold_val annotations/gold_clean/gold_val.csv \
  --pred pseudolabels/output_ministral3_3B.csv \
  --out_dir results/ministral3_3B_eval \
  --bootstrap_n 2000 --bootstrap_seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Normalisation des valeurs
# -----------------------------

YES_SET = {"oui", "yes", "y", "true", "1", 1, True}
NO_SET = {"non", "no", "n", "false", "0", 0, False}
UNK_SET = {"inconnu", "unknown", "unk", "na", "n/a", "", None}


def _is_nan(x) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def normalize_ternary(x) -> str:
    """Normalise vers {'oui','non','inconnu'}."""
    if _is_nan(x):
        return "inconnu"

    if isinstance(x, str):
        s = x.strip().lower()
        if s in YES_SET:
            return "oui"
        if s in NO_SET:
            return "non"
        if s in UNK_SET:
            return "inconnu"
        # sorties fréquentes LLM
        if s in {"positive", "pos"}:
            return "oui"
        if s in {"negative", "neg"}:
            return "non"
        if s in {"?", "unsure", "uncertain"}:
            return "inconnu"
        if s.isdigit():
            return "oui" if int(s) == 1 else "non"
        return "inconnu"

    if x in YES_SET:
        return "oui"
    if x in NO_SET:
        return "non"
    return "inconnu"


def to_bin_int(x: str) -> int:
    """Map 'oui'->1, 'non'->0. (Ne doit jamais recevoir 'inconnu')."""
    if x == "oui":
        return 1
    if x == "non":
        return 0
    raise ValueError(f"to_bin_int expects 'oui'/'non', got {x!r}")


def ternary_to_code(x: str) -> int:
    """
    Code compact:
      oui -> 1
      non -> 0
      inconnu -> -1
    """
    if x == "oui":
        return 1
    if x == "non":
        return 0
    return -1


# -----------------------------
# I/O + détection colonnes
# -----------------------------

COMMON_ID_COLS = ["report_id", "study_id", "exam_id", "accession", "accession_number", "id"]


def detect_id_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in COMMON_ID_COLS:
        if c in df.columns:
            return c
    raise ValueError(
        f"Impossible de trouver la colonne ID. "
        f"Essayé preferred={preferred!r} et {COMMON_ID_COLS}. "
        f"Colonnes disponibles (extrait): {list(df.columns)[:30]}"
    )


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def list_pred_files(pred_path: Path, pred_glob: Optional[str]) -> List[Path]:
    if pred_path.is_file():
        return [pred_path]
    if pred_path.is_dir():
        pattern = pred_glob or "*.csv"
        return sorted(pred_path.glob(pattern))
    raise ValueError(f"--pred doit être un fichier ou un dossier: {pred_path}")


def parse_labels_arg(labels_arg: Optional[str]) -> Optional[List[str]]:
    """
    --labels peut être:
    - None (auto)
    - une liste séparée par des virgules: "a,b,c"
    - un fichier texte: path.txt (1 label par ligne)
    """
    if labels_arg is None:
        return None
    p = Path(labels_arg)
    if p.exists() and p.is_file():
        labels = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                labels.append(s)
        return labels
    return [x.strip() for x in labels_arg.split(",") if x.strip()]


def infer_labels(gold_df: pd.DataFrame, id_col: str, labels_arg: Optional[List[str]]) -> List[str]:
    if labels_arg is not None and len(labels_arg) > 0:
        return labels_arg
    return [c for c in gold_df.columns if c != id_col]


def find_pred_col(pred_df: pd.DataFrame, label: str) -> Optional[str]:
    candidates = [
        label,
        f"{label}_pred",
        f"pred_{label}",
        f"prediction_{label}",
        f"{label}__pred",
        f"{label}.pred",
    ]
    for c in candidates:
        if c in pred_df.columns:
            return c
    return None


def prepare_merged(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col_gold: str,
    id_col_pred: str,
    labels: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Construit un DataFrame merged ne contenant que:
    - report_id
    - gold__{label}
    - pred__{label}

    INNER JOIN => ignore automatiquement les pseudo-labels sans gold.
    """
    pred_col_map: Dict[str, str] = {}
    kept_labels: List[str] = []

    for lab in labels:
        c = find_pred_col(pred_df, lab)
        if c is not None:
            pred_col_map[lab] = c
            kept_labels.append(lab)

    if len(kept_labels) == 0:
        raise ValueError(
            "Aucun label n'a été trouvé dans le CSV de prédictions. "
            "Soit les noms ne correspondent pas, soit il faut passer --labels."
        )

    missing_in_pred = [lab for lab in labels if lab not in pred_col_map]
    if missing_in_pred:
        # non bloquant, mais utile en debug
        pass

    gold_small = gold_df[[id_col_gold] + list(labels)].copy()
    pred_small = pred_df[[id_col_pred] + [pred_col_map[l] for l in kept_labels]].copy()

    gold_small = gold_small.rename(columns={id_col_gold: "report_id"})
    pred_small = pred_small.rename(columns={id_col_pred: "report_id"})

    gold_small = gold_small.rename(columns={lab: f"gold__{lab}" for lab in labels})
    pred_small = pred_small.rename(columns={pred_col_map[lab]: f"pred__{lab}" for lab in kept_labels})

    merged = gold_small.merge(pred_small, on="report_id", how="inner")
    return merged, kept_labels, pred_col_map


# -----------------------------
# Métriques
# -----------------------------

def safe_div(n: float, d: float) -> float:
    if d is None:
        return float("nan")
    try:
        if float(d) == 0.0:
            return float("nan")
    except Exception:
        return float("nan")
    return float(n / d)


def compute_binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    bal_acc = np.nanmean([sens, spec]) if not (np.isnan(sens) and np.isnan(spec)) else np.nan

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "acc": float(acc), "sens": float(sens), "spec": float(spec),
        "ppv": float(ppv), "npv": float(npv),
        "f1": float(f1), "bal_acc": float(bal_acc),
    }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return compute_binary_metrics_from_counts(tp, tn, fp, fn)


# -----------------------------
# Bootstrap CI
# -----------------------------

METRICS_FOR_CI_PER_LABEL = [
    "sens", "spec", "bal_acc", "f1",
    "acc", "ppv", "npv",
    "coverage_pred_answered_over_total",
    "coverage_answered_over_gold_known",
    "abstention_P_inconnu_given_gold_oui",
    "abstention_P_inconnu_given_gold_non",
]

METRICS_FOR_CI_SUMMARY = [
    "macro_sens", "macro_spec", "macro_bal_acc", "macro_f1",
    "mean_coverage_pred_answered_over_total",
    "mean_coverage_answered_over_gold_known",
    "mean_abstention_given_gold_oui",
    "mean_abstention_given_gold_non",
]


def bootstrap_percentile_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Percentile CI, nan-aware.
    Retourne (low, high). Si tout nan => (nan, nan).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    low = float(np.quantile(x, alpha / 2.0))
    high = float(np.quantile(x, 1.0 - alpha / 2.0))
    return low, high


def _compute_metrics_from_codes(g_code: np.ndarray, p_code: np.ndarray) -> Dict[str, float]:
    """
    g_code, p_code: arrays length n_total with codes:
      1=yes, 0=no, -1=unk
    Applique la même logique que ton script:
      - eval set = gold connu & pred répondu (oui/non)
      - metrics calculées sur eval set
      - coverage & abstention calculées sur total / conditionnel gold.
    """
    n_total = int(g_code.shape[0])

    gold_known = (g_code >= 0)
    pred_answered = (p_code >= 0)
    answered_and_gold_known = gold_known & pred_answered

    n_gold_known = int(gold_known.sum())
    n_answered = int(pred_answered.sum())
    n_eval = int(answered_and_gold_known.sum())

    # abstentions conditionnelles
    n_gold_yes = int((g_code == 1).sum())
    n_gold_no = int((g_code == 0).sum())
    abst_yes = safe_div(int(((g_code == 1) & (p_code == -1)).sum()), n_gold_yes)
    abst_no = safe_div(int(((g_code == 0) & (p_code == -1)).sum()), n_gold_no)

    out = {
        "n_total_merged": n_total,
        "n_gold_known": n_gold_known,
        "n_pred_answered": n_answered,
        "n_eval_answered_and_gold_known": n_eval,
        "coverage_pred_answered_over_total": safe_div(n_answered, n_total),
        "coverage_answered_over_gold_known": safe_div(n_eval, n_gold_known),
        "abstention_P_inconnu_given_gold_oui": float(abst_yes),
        "abstention_P_inconnu_given_gold_non": float(abst_no),
    }

    if n_eval <= 0:
        out.update({k: float("nan") for k in ["tp", "tn", "fp", "fn", "acc", "sens", "spec", "ppv", "npv", "f1", "bal_acc"]})
        return out

    g_eval = g_code[answered_and_gold_known]
    p_eval = p_code[answered_and_gold_known]

    tp = int(((g_eval == 1) & (p_eval == 1)).sum())
    tn = int(((g_eval == 0) & (p_eval == 0)).sum())
    fp = int(((g_eval == 0) & (p_eval == 1)).sum())
    fn = int(((g_eval == 1) & (p_eval == 0)).sum())

    out.update(compute_binary_metrics_from_counts(tp, tn, fp, fn))
    return out


def add_bootstrap_cis(
    per_label_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    merged: pd.DataFrame,
    kept_labels: Sequence[str],
    split_name: str,
    bootstrap_n: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcule IC bootstrap et ajoute des colonnes:
      metric__ci_low / metric__ci_high
    aux dfs per_label_df et summary_df.
    """
    if bootstrap_n <= 0:
        return per_label_df, summary_df

    rng = np.random.default_rng(bootstrap_seed)
    n_total = int(len(merged))

    # Pré-calcul des codes pour chaque label (sur dataset merged complet)
    gold_codes: Dict[str, np.ndarray] = {}
    pred_codes: Dict[str, np.ndarray] = {}

    for lab in kept_labels:
        g = merged[f"gold__{lab}"].map(normalize_ternary).map(ternary_to_code).to_numpy(dtype=np.int8)
        p = merged[f"pred__{lab}"].map(normalize_ternary).map(ternary_to_code).to_numpy(dtype=np.int8)
        gold_codes[lab] = g
        pred_codes[lab] = p

    # Containers: distributions bootstrap
    # per-label: dict[lab][metric] -> list
    dist_per_label: Dict[str, Dict[str, List[float]]] = {lab: {m: [] for m in METRICS_FOR_CI_PER_LABEL} for lab in kept_labels}
    # summary: metric -> list
    dist_summary: Dict[str, List[float]] = {m: [] for m in METRICS_FOR_CI_SUMMARY}

    for _ in range(bootstrap_n):
        idx = rng.integers(0, n_total, size=n_total, endpoint=False)

        # per label metrics
        per_label_metrics_this_boot = []
        for lab in kept_labels:
            g_b = gold_codes[lab][idx]
            p_b = pred_codes[lab][idx]
            m = _compute_metrics_from_codes(g_b, p_b)

            for mm in METRICS_FOR_CI_PER_LABEL:
                dist_per_label[lab][mm].append(float(m.get(mm, float("nan"))))

            per_label_metrics_this_boot.append(m)

        # macro summary over labels (nanmean)
        # Important: on ne refait pas les "labels_evaluated" etc, seulement les métriques macro/means.
        def _nanmean_key(key: str) -> float:
            arr = np.array([float(d.get(key, np.nan)) for d in per_label_metrics_this_boot], dtype=float)
            return float(np.nanmean(arr))

        dist_summary["macro_sens"].append(_nanmean_key("sens"))
        dist_summary["macro_spec"].append(_nanmean_key("spec"))
        dist_summary["macro_bal_acc"].append(_nanmean_key("bal_acc"))
        dist_summary["macro_f1"].append(_nanmean_key("f1"))
        dist_summary["mean_coverage_pred_answered_over_total"].append(_nanmean_key("coverage_pred_answered_over_total"))
        dist_summary["mean_coverage_answered_over_gold_known"].append(_nanmean_key("coverage_answered_over_gold_known"))
        dist_summary["mean_abstention_given_gold_oui"].append(_nanmean_key("abstention_P_inconnu_given_gold_oui"))
        dist_summary["mean_abstention_given_gold_non"].append(_nanmean_key("abstention_P_inconnu_given_gold_non"))

    # Ajout CI dans per_label_df
    per_label_df = per_label_df.copy()
    for lab in kept_labels:
        mask = (per_label_df["split"] == split_name) & (per_label_df["label"] == lab)
        if not mask.any():
            continue
        for mm in METRICS_FOR_CI_PER_LABEL:
            arr = np.array(dist_per_label[lab][mm], dtype=float)
            lo, hi = bootstrap_percentile_ci(arr, alpha=bootstrap_alpha)
            per_label_df.loc[mask, f"{mm}__ci_low"] = lo
            per_label_df.loc[mask, f"{mm}__ci_high"] = hi

    # Ajout CI dans summary_df
    summary_df = summary_df.copy()
    for mm in METRICS_FOR_CI_SUMMARY:
        arr = np.array(dist_summary[mm], dtype=float)
        lo, hi = bootstrap_percentile_ci(arr, alpha=bootstrap_alpha)
        summary_df.loc[0, f"{mm}__ci_low"] = lo
        summary_df.loc[0, f"{mm}__ci_high"] = hi

    # Meta info bootstrap dans summary
    summary_df.loc[0, "bootstrap_n"] = int(bootstrap_n)
    summary_df.loc[0, "bootstrap_alpha"] = float(bootstrap_alpha)
    summary_df.loc[0, "bootstrap_seed"] = int(bootstrap_seed)

    return per_label_df, summary_df


# -----------------------------
# Prévalence gold
# -----------------------------

def compute_gold_prevalence(
    gold_df: pd.DataFrame,
    split_name: str,
    labels: Sequence[str],
    id_col_preferred: str = "report_id",
) -> pd.DataFrame:
    """
    Prévalences calculées sur le gold du split (AVANT merge avec pred).
    """
    id_col_gold = detect_id_col(gold_df, id_col_preferred)

    rows = []
    n_reports = int(len(gold_df))

    for lab in labels:
        if lab not in gold_df.columns:
            continue

        g = gold_df[lab].map(normalize_ternary)

        n_yes = int((g == "oui").sum())
        n_no = int((g == "non").sum())
        n_unk = int((g == "inconnu").sum())
        n_known = n_yes + n_no

        rows.append(
            {
                "split": split_name,
                "label": lab,
                "n_reports_gold": n_reports,
                "n_gold_oui": n_yes,
                "n_gold_non": n_no,
                "n_gold_inconnu": n_unk,
                "n_gold_known": n_known,
                "prev_oui_sur_total": safe_div(n_yes, n_reports),
                "prev_oui_sur_known": safe_div(n_yes, n_known),
                "rate_inconnu_sur_total": safe_div(n_unk, n_reports),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "label"]).reset_index(drop=True)


# -----------------------------
# Évaluation d'un split
# -----------------------------

def evaluate_one_split(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    split_name: str,
    labels: Sequence[str],
    id_col_preferred: str = "report_id",
    save_merged_debug: bool = True,
    max_errors_per_label: int = 0,
    bootstrap_n: int = 0,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    id_col_gold = detect_id_col(gold_df, id_col_preferred)
    id_col_pred = detect_id_col(pred_df, id_col_preferred)

    merged, kept_labels, pred_col_map = prepare_merged(
        gold_df=gold_df,
        pred_df=pred_df,
        id_col_gold=id_col_gold,
        id_col_pred=id_col_pred,
        labels=labels,
    )

    if merged.empty:
        raise ValueError(
            f"[{split_name}] Après INNER JOIN sur l'ID, 0 lignes. "
            f"Vérifie que gold et pred partagent les mêmes report_id."
        )

    per_label_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for lab in kept_labels:
        gcol = f"gold__{lab}"
        pcol = f"pred__{lab}"

        g = merged[gcol].map(normalize_ternary)
        p = merged[pcol].map(normalize_ternary)

        n_total = int(len(merged))
        gold_known = g.isin(["oui", "non"])
        pred_answered = p.isin(["oui", "non"])
        answered_and_gold_known = gold_known & pred_answered

        n_gold_known = int(gold_known.sum())
        n_answered = int(pred_answered.sum())
        n_eval = int(answered_and_gold_known.sum())

        abst_yes = safe_div(int(((g == "oui") & (p == "inconnu")).sum()), int((g == "oui").sum()))
        abst_no = safe_div(int(((g == "non") & (p == "inconnu")).sum()), int((g == "non").sum()))

        if n_eval > 0:
            y_true = g[answered_and_gold_known].map(to_bin_int).to_numpy()
            y_pred = p[answered_and_gold_known].map(to_bin_int).to_numpy()
            m = compute_binary_metrics(y_true, y_pred)
        else:
            m = {k: np.nan for k in ["tp", "tn", "fp", "fn", "acc", "sens", "spec", "ppv", "npv", "f1", "bal_acc"]}

        per_label_rows.append(
            {
                "split": split_name,
                "label": lab,
                "pred_col_used": pred_col_map.get(lab, ""),
                "n_total_merged": n_total,
                "n_gold_known": n_gold_known,
                "n_pred_answered": n_answered,
                "n_eval_answered_and_gold_known": n_eval,
                "coverage_pred_answered_over_total": safe_div(n_answered, n_total),
                "coverage_answered_over_gold_known": safe_div(n_eval, n_gold_known),
                "abstention_P_inconnu_given_gold_oui": abst_yes,
                "abstention_P_inconnu_given_gold_non": abst_no,
                **m,
            }
        )

        # FP/FN sur eval set + abstentions (optionnellement limité)
        if max_errors_per_label is None:
            max_errors_per_label = 0
        do_limit = (max_errors_per_label > 0)

        # FP/FN
        if n_eval > 0:
            g_eval = g[answered_and_gold_known]
            p_eval = p[answered_and_gold_known]

            fp_mask = (g_eval == "non") & (p_eval == "oui")
            fn_mask = (g_eval == "oui") & (p_eval == "non")

            fp_indices = merged.index[answered_and_gold_known][fp_mask]
            fn_indices = merged.index[answered_and_gold_known][fn_mask]

            if do_limit:
                fp_indices = fp_indices[:max_errors_per_label]
                fn_indices = fn_indices[:max_errors_per_label]

            for idx in fp_indices:
                error_rows.append(
                    {
                        "split": split_name,
                        "label": lab,
                        "report_id": merged.at[idx, "report_id"],
                        "error_type": "FP",
                        "gold": normalize_ternary(merged.at[idx, gcol]),
                        "pred": normalize_ternary(merged.at[idx, pcol]),
                    }
                )
            for idx in fn_indices:
                error_rows.append(
                    {
                        "split": split_name,
                        "label": lab,
                        "report_id": merged.at[idx, "report_id"],
                        "error_type": "FN",
                        "gold": normalize_ternary(merged.at[idx, gcol]),
                        "pred": normalize_ternary(merged.at[idx, pcol]),
                    }
                )

        # abstentions: pred inconnu alors que gold connu
        abst_mask = gold_known & (p == "inconnu")
        abst_indices = merged.index[abst_mask]
        if do_limit:
            abst_indices = abst_indices[:max_errors_per_label]

        for idx in abst_indices:
            error_rows.append(
                {
                    "split": split_name,
                    "label": lab,
                    "report_id": merged.at[idx, "report_id"],
                    "error_type": "ABSTENTION",
                    "gold": normalize_ternary(merged.at[idx, gcol]),
                    "pred": "inconnu",
                }
            )

    per_label_df = pd.DataFrame(per_label_rows).sort_values(["split", "label"]).reset_index(drop=True)
    errors_df = pd.DataFrame(error_rows).sort_values(["split", "label", "error_type", "report_id"]).reset_index(drop=True)

    # Macro (moyenne par label) - nanmean pour ignorer labels non évaluables
    summary = {
        "split": split_name,
        "n_reports_merged": int(len(merged)),
        "n_labels_evaluated": int(len(per_label_df)),
        "macro_sens": float(np.nanmean(pd.to_numeric(per_label_df["sens"], errors="coerce").to_numpy())),
        "macro_spec": float(np.nanmean(pd.to_numeric(per_label_df["spec"], errors="coerce").to_numpy())),
        "macro_bal_acc": float(np.nanmean(pd.to_numeric(per_label_df["bal_acc"], errors="coerce").to_numpy())),
        "macro_f1": float(np.nanmean(pd.to_numeric(per_label_df["f1"], errors="coerce").to_numpy())),
        "mean_coverage_pred_answered_over_total": float(np.nanmean(pd.to_numeric(per_label_df["coverage_pred_answered_over_total"], errors="coerce").to_numpy())),
        "mean_coverage_answered_over_gold_known": float(np.nanmean(pd.to_numeric(per_label_df["coverage_answered_over_gold_known"], errors="coerce").to_numpy())),
        "mean_abstention_given_gold_oui": float(np.nanmean(pd.to_numeric(per_label_df["abstention_P_inconnu_given_gold_oui"], errors="coerce").to_numpy())),
        "mean_abstention_given_gold_non": float(np.nanmean(pd.to_numeric(per_label_df["abstention_P_inconnu_given_gold_non"], errors="coerce").to_numpy())),
    }
    summary_df = pd.DataFrame([summary])

    # Bootstrap CI (optionnel)
    if bootstrap_n and bootstrap_n > 0:
        per_label_df, summary_df = add_bootstrap_cis(
            per_label_df=per_label_df,
            summary_df=summary_df,
            merged=merged,
            kept_labels=kept_labels,
            split_name=split_name,
            bootstrap_n=bootstrap_n,
            bootstrap_alpha=bootstrap_alpha,
            bootstrap_seed=bootstrap_seed,
        )

    merged_debug_df = None
    if save_merged_debug:
        dbg_cols = ["report_id"]
        for lab in [r["label"] for r in per_label_rows]:
            dbg_cols += [f"gold__{lab}", f"pred__{lab}"]
        merged_debug_df = merged[dbg_cols].copy()

    return per_label_df, summary_df, errors_df, merged_debug_df


# -----------------------------
# Outputs
# -----------------------------

def save_outputs(
    out_dir: Path,
    split_name: str,
    per_label_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    merged_debug_df: Optional[pd.DataFrame],
    gold_prev_df: Optional[pd.DataFrame] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_label_df.to_csv(out_dir / f"per_label_metrics_{split_name}.csv", index=False)
    summary_df.to_csv(out_dir / f"summary_{split_name}.csv", index=False)
    errors_df.to_csv(out_dir / f"errors_{split_name}.csv", index=False)

    if gold_prev_df is not None:
        gold_prev_df.to_csv(out_dir / f"gold_prevalence_{split_name}.csv", index=False)

    if merged_debug_df is not None:
        merged_debug_df.to_csv(out_dir / f"merged_used_{split_name}.csv", index=False)


# -----------------------------
# Affichage console
# -----------------------------

def _fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"
    return str(x)


def print_gold_prevalence(split_name: str, gold_prev_df: pd.DataFrame) -> None:
    if gold_prev_df is None or gold_prev_df.empty:
        return

    df = gold_prev_df.copy()
    for c in ["prev_oui_sur_total", "prev_oui_sur_known", "rate_inconnu_sur_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cols = [
        "label",
        "n_reports_gold",
        "n_gold_oui",
        "n_gold_non",
        "n_gold_inconnu",
        "prev_oui_sur_total",
        "prev_oui_sur_known",
        "rate_inconnu_sur_total",
    ]

    print("\nGold prevalence (computed on GOLD split, before merge):")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_console_summary(split_name: str, per_label_df: pd.DataFrame, summary_df: pd.DataFrame, top_k: int = 8) -> None:
    s = summary_df.iloc[0].to_dict()

    print("\n" + "=" * 100)
    print(f"[{split_name}] SUMMARY")
    print("-" * 100)
    print(f"reports_merged: {s['n_reports_merged']}")
    print(f"labels_evaluated: {s['n_labels_evaluated']}")
    print(
        f"macro_sens: {_fmt(s['macro_sens'])} | macro_spec: {_fmt(s['macro_spec'])} | "
        f"macro_bal_acc: {_fmt(s['macro_bal_acc'])} | macro_f1: {_fmt(s['macro_f1'])}"
    )
    print(
        f"mean_coverage(pred answered / total): {_fmt(s['mean_coverage_pred_answered_over_total'])} | "
        f"mean_coverage(answered / gold known): {_fmt(s['mean_coverage_answered_over_gold_known'])}"
    )
    print(
        f"mean_abstention P(inconnu|gold=oui): {_fmt(s['mean_abstention_given_gold_oui'])} | "
        f"P(inconnu|gold=non): {_fmt(s['mean_abstention_given_gold_non'])}"
    )

    if "bootstrap_n" in summary_df.columns and pd.notna(summary_df.loc[0, "bootstrap_n"]):
        bn = int(summary_df.loc[0, "bootstrap_n"])
        alpha = float(summary_df.loc[0, "bootstrap_alpha"])
        print(f"bootstrap: n={bn} | CI={(1.0 - alpha) * 100:.1f}% (percentile)")

    df = per_label_df.copy()
    for c in ["sens", "spec", "bal_acc", "f1", "coverage_pred_answered_over_total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    top_bal = df.sort_values("bal_acc", ascending=False).head(top_k)
    top_f1 = df.sort_values("f1", ascending=False).head(top_k)

    cols = [
        "label",
        "n_eval_answered_and_gold_known",
        "coverage_pred_answered_over_total",
        "sens",
        "spec",
        "bal_acc",
        "f1",
        "fp",
        "fn",
    ]

    print("\nTop labels by balanced accuracy:")
    print(top_bal[cols].to_string(index=False))

    print("\nTop labels by F1:")
    print(top_f1[cols].to_string(index=False))

    print("=" * 100 + "\n")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluate pseudo-labels (oui/non/inconnu) vs gold")

    p.add_argument(
        "--split",
        default="both",
        choices=["val", "test", "both"],
        help="Évaluer val, test, ou les deux.",
    )

    p.add_argument("--gold_val", type=Path, required=False, help="Path to gold_val.csv")
    p.add_argument("--gold_test", type=Path, required=False, help="Path to gold_test.csv")

    p.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Path to prediction CSV, or directory containing many CSVs.",
    )
    p.add_argument(
        "--pred_glob",
        type=str,
        default="*.csv",
        help="If --pred is a directory, glob pattern (default: *.csv)",
    )

    p.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--id_col",
        type=str,
        default="report_id",
        help="Preferred ID column name (auto-detect fallback if absent).",
    )
    p.add_argument(
        "--labels",
        type=str,
        required=False,
        help="Optionnel: 'a,b,c' ou path.txt (1 label/ligne). Sinon auto depuis gold.",
    )
    p.add_argument(
        "--no_merged_debug",
        action="store_true",
        help="Ne pas exporter merged_used_{split}.csv (debug).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Nombre de labels à afficher dans les tops (bal_acc / f1).",
    )

    # Bootstrap
    p.add_argument(
        "--bootstrap_n",
        type=int,
        default=0,
        help="Nombre de bootstraps (0 => désactivé). Exemple: 2000.",
    )
    p.add_argument(
        "--bootstrap_alpha",
        type=float,
        default=0.05,
        help="Alpha pour IC percentile (0.05 => IC 95%).",
    )
    p.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Seed bootstrap pour reproductibilité.",
    )

    # Errors limiting
    p.add_argument(
        "--max_errors_per_label",
        type=int,
        default=0,
        help="Limite le nombre d'erreurs exportées par label et type (FP/FN/ABSTENTION). 0 => illimité.",
    )

    args = p.parse_args()

    if args.split in ("val", "both") and args.gold_val is None:
        p.error("--gold_val est requis quand --split vaut val ou both")
    if args.split in ("test", "both") and args.gold_test is None:
        p.error("--gold_test est requis quand --split vaut test ou both")

    if args.bootstrap_n < 0:
        p.error("--bootstrap_n doit être >= 0")
    if not (0.0 < args.bootstrap_alpha < 1.0):
        p.error("--bootstrap_alpha doit être entre 0 et 1 (ex: 0.05)")
    if args.max_errors_per_label < 0:
        p.error("--max_errors_per_label doit être >= 0")

    return args


def evaluate_for_pred_file(pred_file: Path, args) -> None:
    pred_df = load_csv(pred_file)

    gold_val_df = load_csv(args.gold_val) if args.split in ("val", "both") else None
    gold_test_df = load_csv(args.gold_test) if args.split in ("test", "both") else None

    labels_arg = parse_labels_arg(args.labels)
    gold_for_labels = gold_val_df if gold_val_df is not None else gold_test_df

    id_col_gold = detect_id_col(gold_for_labels, args.id_col)
    labels = infer_labels(gold_for_labels, id_col_gold, labels_arg)

    multi = args.pred.is_dir()
    out_dir = (args.out_dir / pred_file.stem) if multi else args.out_dir

    save_merged_debug = not args.no_merged_debug

    if args.split in ("val", "both"):
        gold_prev = compute_gold_prevalence(
            gold_df=gold_val_df,
            split_name="val",
            labels=labels,
            id_col_preferred=args.id_col,
        )

        per_label, summary, errors, merged_debug = evaluate_one_split(
            gold_df=gold_val_df,
            pred_df=pred_df,
            split_name="val",
            labels=labels,
            id_col_preferred=args.id_col,
            save_merged_debug=save_merged_debug,
            max_errors_per_label=args.max_errors_per_label,
            bootstrap_n=args.bootstrap_n,
            bootstrap_alpha=args.bootstrap_alpha,
            bootstrap_seed=args.bootstrap_seed,
        )

        save_outputs(out_dir, "val", per_label, summary, errors, merged_debug, gold_prev_df=gold_prev)
        print_console_summary("val", per_label, summary, top_k=args.top_k)
        print_gold_prevalence("val", gold_prev)

    if args.split in ("test", "both"):
        gold_prev = compute_gold_prevalence(
            gold_df=gold_test_df,
            split_name="test",
            labels=labels,
            id_col_preferred=args.id_col,
        )

        per_label, summary, errors, merged_debug = evaluate_one_split(
            gold_df=gold_test_df,
            pred_df=pred_df,
            split_name="test",
            labels=labels,
            id_col_preferred=args.id_col,
            save_merged_debug=save_merged_debug,
            max_errors_per_label=args.max_errors_per_label,
            bootstrap_n=args.bootstrap_n,
            bootstrap_alpha=args.bootstrap_alpha,
            bootstrap_seed=args.bootstrap_seed,
        )

        save_outputs(out_dir, "test", per_label, summary, errors, merged_debug, gold_prev_df=gold_prev)
        print_console_summary("test", per_label, summary, top_k=args.top_k)
        print_gold_prevalence("test", gold_prev)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "pred_file": str(pred_file),
        "split": args.split,
        "id_col_preferred": args.id_col,
        "n_labels_requested": int(len(labels)),
        "note": "INNER JOIN on report_id => predictions without matching gold are ignored.",
        "bootstrap_n": int(args.bootstrap_n),
        "bootstrap_alpha": float(args.bootstrap_alpha),
        "bootstrap_seed": int(args.bootstrap_seed),
        "max_errors_per_label": int(args.max_errors_per_label),
    }
    pd.Series(meta).to_json(out_dir / "run_meta.json", force_ascii=False, indent=2)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = list_pred_files(args.pred, args.pred_glob)
    if not pred_files:
        raise ValueError(f"Aucun CSV trouvé: pred={args.pred} glob={args.pred_glob}")

    for f in pred_files:
        evaluate_for_pred_file(f, args)


if __name__ == "__main__":
    main()
