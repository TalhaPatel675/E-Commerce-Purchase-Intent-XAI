"""Two-lens feature importance: SHAP + permutation (PRD §4.4 "more than one method").
TreeExplainer is used explicitly for tree-based models (PRD §4.6 Student hint)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

log = get_logger(__name__)


TREE_MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "DecisionTree")


def _build_explainer(model, X_background: np.ndarray, feature_names: list[str]):
    """Pick the right SHAP explainer per model family (PRD §4.6 "tree models")."""
    import shap

    class_name = type(model).__name__
    if any(n in class_name for n in TREE_MODEL_NAMES):
        try:
            return shap.TreeExplainer(model), "TreeExplainer"
        except Exception as e:
            log.warning("TreeExplainer failed (%s); falling back to shap.Explainer", e)

    try:
        return shap.Explainer(model, X_background, feature_names=feature_names), "Explainer"
    except Exception as e:
        log.warning("shap.Explainer failed (%s); falling back to KernelExplainer", e)
        return shap.KernelExplainer(model.predict_proba, X_background[:50]), "KernelExplainer"


def _as_class1_shap(explainer, X: np.ndarray):
    """Normalise across SHAP API versions."""
    try:
        sv = explainer.shap_values(X)
    except Exception as e:
        log.warning("SHAP value computation failed: %s", e)
        return None
    if isinstance(sv, list):  # binary classification: [neg-class, pos-class]
        return sv[1]
    return sv


def generate_global_shap(
    best_model,
    X_sample: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> dict[str, Any]:
    import matplotlib
    import shap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer, kind = _build_explainer(best_model, X_sample[:100], feature_names)
    log.info("Using SHAP backend: %s", kind)
    sv = _as_class1_shap(explainer, X_sample[:300])
    if sv is None:
        return {"ok": False, "reason": "shap_values returned None"}

    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(sv, X_sample[:300], feature_names=feature_names, show=False)
    except Exception as e:
        log.warning("summary_plot failed: %s", e)
        plt.close()
        plt.figure(figsize=(10, 6))
        shap.bar_plot(np.mean(np.abs(sv), axis=0), feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(sv), axis=0)
    ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True)
    return {"ok": True, "backend": kind, "feature_importance": ranked[:20]}


def generate_permutation_importance(
    best_model,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    n_repeats: int = 10,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Second lens (PRD §4.4). Independent of model family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    # Manual permutation scoring avoids estimator-tag incompatibilities between
    # CatBoost and newer scikit-learn releases while preserving the same metric.
    rng = np.random.default_rng(random_seed)
    baseline = average_precision_score(y_sample, best_model.predict_proba(X_sample)[:, 1])
    drops = np.zeros((n_repeats, X_sample.shape[1]), dtype=float)
    for repeat in range(n_repeats):
        for col in range(X_sample.shape[1]):
            X_perm = X_sample.copy()
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            score = average_precision_score(y_sample, best_model.predict_proba(X_perm)[:, 1])
            drops[repeat, col] = baseline - score
    means = drops.mean(axis=0)
    stds = drops.std(axis=0)
    ranked = sorted(
        zip(feature_names, means.tolist(), stds.tolist()), key=lambda kv: kv[1], reverse=True
    )
    plt.figure(figsize=(10, 6))
    names = [r[0] for r in ranked[:15]]
    means = [r[1] for r in ranked[:15]]
    stds = [r[2] for r in ranked[:15]]
    plt.barh(range(len(names))[::-1], means, xerr=stds)
    plt.yticks(range(len(names))[::-1], names)
    plt.xlabel("Permutation importance (mean ± std) on average_precision")
    plt.title("Permutation importance (permutation_method: independent of model)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

    return {"ok": True, "feature_importance": [(n, float(m), float(s)) for n, m, s in ranked[:20]]}


def generate_local_lime(
    best_model, X_sample: np.ndarray, feature_names: list[str], out_path: Path
) -> dict[str, Any]:
    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            training_data=np.asarray(X_sample),
            feature_names=list(feature_names),
            class_names=["no_purchase", "purchase"],
            discretize_continuous=True,
            random_state=42,
        )
        exp = explainer.explain_instance(
            data_row=X_sample[0],
            predict_fn=best_model.predict_proba,
            num_features=8,
        )
        exp.save_to_file(str(out_path))
        return {"ok": True, "top_features": exp.as_list()}
    except Exception as e:
        log.exception("LIME failed: %s", e)
        return {"ok": False, "reason": str(e)}


def write_model_card(
    out_path: Path,
    best_name: str,
    results: dict[str, Any],
    threshold_info: dict[str, float],
    shap_importance: list[Any],
    perm_importance: list[tuple[str, float, float]],
    ablation_info: dict[str, Any],
    data_dictionary_csv: Path,
) -> Path:
    lines: list[str] = [
        "# Model Card — Purchase Intent Prediction\n",
        "## Intended use",
        "Predict whether an online shopping session will end in a purchase.",
        "Use the probability + value-based decision threshold to plan retention interventions.\n",
        "## Selected model",
        f"`{best_name}`\n",
        "## Decision threshold & business value",
        "",
    ]
    lines.append(
        f"- Decision threshold: **{threshold_info['threshold']:.3f}** (chosen by `value_based_threshold`, not 0.5)"
    )
    lines.append(f"- Expected value at threshold: **{threshold_info['expected_value']:.1f}**")
    lines.append(f"- Precision @ threshold: **{threshold_info['precision']:.3f}**")
    lines.append(f"- Recall @ threshold: **{threshold_info['recall']:.3f}**\n")
    lines.append(
        "## Held-out test set metrics — sorted by PR-AUC (PRD asks for PR-AUC, not accuracy)"
    )
    lines.append(
        "\n| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 | CV PR-AUC (mean ± std) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, m in sorted(results.items(), key=lambda kv: kv[1].get("pr_auc", -1), reverse=True):
        mean = m.get("cv_pr_auc_mean", 0.0)
        sd = m.get("cv_pr_auc_std", 0.0)
        lines.append(
            f"| {name} | {m.get('pr_auc', 0):.3f} | {m.get('roc_auc', 0):.3f} | "
            f"{m.get('recall', 0):.3f} | {m.get('precision', 0):.3f} | "
            f"{m.get('f1', 0):.3f} | {mean:.3f} ± {sd:.3f} |"
        )
    lines.append("\n## Feature importance — lens #1 (SHAP, top 10)")
    if shap_importance:
        lines.append("\n| Rank | Feature | Mean |SHAP| |\n|---|---|---|")
        for i, pair in enumerate(shap_importance[:10], 1):
            try:
                score = float(pair[1])
            except Exception:
                score = 0.0
            lines.append(f"| {i} | {pair[0]} | {score:.4f} |")
    lines.append("\n## Feature importance — lens #2 (scikit-learn permutation, top 10)")
    if perm_importance:
        lines.append("\n| Rank | Feature | Importance | Std |\n|---|---|---|---|")
        for i, (name, mean, std) in enumerate(perm_importance[:10], 1):
            lines.append(f"| {i} | {name} | {mean:.4f} | {std:.4f} |")
    lines.append("\n## PageValues leakage audit (PRD §2.2)")
    if ablation_info:
        m = ablation_info.get("metrics", {})
        lines.append(f"- Dropped features: {ablation_info.get('dropped_features', [])}")
        lines.append(
            f"- Selected model still trained WITHOUT PageValues. PR-AUC: {m.get('pr_auc', 0):.3f}, ROC-AUC: {m.get('roc_auc', 0):.3f}, Recall: {m.get('recall', 0):.3f}"
        )
        lines.append(
            "- Honest take: PageValues is a deployment-time data hazard (only known after the funnel renders). See `docs/pagevalues_ablation.md` for the full write-up.\n"
        )
    lines.append("## Responsible use")
    lines.append(
        "- Intended for aggregate funnel analysis and for choosing which sessions may get a low-cost nudge (for example a discount banner)."
    )
    lines.append(
        "- Do NOT use it to identify or profile a named person, to deny service, to set an individual's price, or for any decision with legal or financial consequences for a person."
    )
    lines.append(
        "- The data is anonymised session behaviour. Do not join it with personally identifying fields."
    )
    lines.append(
        "- PageValues may not be known when a visitor is first seen. Do not score early in a session without re-checking the PageValues audit above.\n"
    )
    lines.append("## Limitations")
    lines.append(
        "- Trained on a single ~12k-session dataset. Re-check distribution shift before any production deploy."
    )
    lines.append("- PageValues is a strong correlative signal; treat it as correlation, not cause.")
    lines.append(
        "- Class imbalance is ~15.7% positives; the primary metric on this dataset must remain PR-AUC, not accuracy."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_explainability_artifacts(
    best_model,
    X_full: np.ndarray,
    y_full: np.ndarray,
    feature_names: list[str],
    reports_dir: Path,
    figures_dir: Path,
    data_dictionary_csv: Path,
    best_name: str,
    results: dict[str, Any],
    threshold_info: dict[str, float],
    ablation_info: dict[str, Any],
    microlens_size: int = 500,
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    idx = rng.choice(X_full.shape[0], size=min(microlens_size, X_full.shape[0]), replace=False)
    X_sample = X_full[idx]
    y_sample = y_full.values[idx] if hasattr(y_full, "values") else y_full[idx]

    shap_info = generate_global_shap(
        best_model, X_sample, feature_names, figures_dir / "shap_global_summary.png"
    )
    perm_info = generate_permutation_importance(
        best_model, X_sample, y_sample, feature_names, figures_dir / "permutation_importance.png"
    )
    local_info = generate_local_lime(
        best_model, X_sample, feature_names, reports_dir / "lime_local_session0.html"
    )

    write_model_card(
        out_path=reports_dir / "MODEL_CARD.md",
        best_name=best_name,
        results=results,
        threshold_info=threshold_info,
        shap_importance=shap_info.get("feature_importance", []),
        perm_importance=perm_info.get("feature_importance", []),
        ablation_info=ablation_info,
        data_dictionary_csv=data_dictionary_csv,
    )

    summary = {
        "selected_model": best_name,
        "threshold": threshold_info["threshold"],
        "expected_value": threshold_info["expected_value"],
        "precision_at_threshold": threshold_info["precision"],
        "recall_at_threshold": threshold_info["recall"],
        "f1_at_threshold": threshold_info["f1"],
        "global_shap": shap_info,
        "permutation_importance": perm_info,
        "local_lime": local_info,
        "pagevalues_ablation": ablation_info,
        "all_results": results,
    }
    (reports_dir / "explainability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("Explainability artifacts written to %s", reports_dir)
    return summary
