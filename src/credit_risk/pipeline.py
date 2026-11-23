"""End-to-end behavioral PD, calibration, expected loss, and monitoring pack.

Protocol, fixed before looking at the holdout:

- Stratified 80/20 split, seed 42. Five-fold stratified CV on the training set.
- Weight of evidence is refit inside every training fold.
- A class-weighted variant is kept only when its CV AUC beats the unweighted
  model by at least 0.005. Probabilities are then isotonic-calibrated on
  out-of-fold training scores either way.
- Gradient boosting becomes the champion only when its CV AUC beats the
  scorecard by at least 0.005. Otherwise the scorecard stays the decision model
  so the cutoff and the adverse-action reasons come from the same equation.
- Sex and marital status are never used as inputs.
- LGD and the undrawn credit-conversion factor are assumptions, not estimates.
"""

from __future__ import annotations

import json
import math
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import (
    APPROVAL_GRID,
    ASSUMED_CCF,
    ASSUMED_LGD,
    EXCEL_PATH,
    FIGURE_DIR,
    GBM_FEATURES,
    LGD_SENSITIVITIES,
    METRICS_PATH,
    N_SPLITS,
    POLICY_APPROVAL_RATE,
    REPORT_DIR,
    SCORECARD_FEATURES,
    SEED,
    TABLE_DIR,
    TEST_SIZE,
)
from credit_risk.data import clean, load_raw
from credit_risk.download import ensure_raw_csv
from credit_risk.expected_loss import (
    assign_risk_grades,
    cutoff_curve,
    exposure_at_default,
    expected_loss,
    grade_summary,
    policy_row,
    portfolio_totals,
    segment_totals,
)
from credit_risk.explain import importance_table, shap_values_matrix
from credit_risk.fairness import (
    acceptance_mask,
    four_fifths_ratio,
    group_outcomes,
    scorecard_reasons,
    shap_reasons,
)
from credit_risk.features import engineer_features, repayment_band, utilization_band
from credit_risk.metrics import (
    brier,
    cumulative_gains,
    decile_table,
    expected_calibration_error,
    population_stability_index,
    ranking_metrics,
    reliability_table,
)
from credit_risk.models import (
    apply_isotonic,
    compare_family,
    cross_validate,
    fit_isotonic,
    fit_scorecard,
    gbm_factory,
    scorecard_factory,
    select_champion,
)
from credit_risk.plotting import (
    plot_approval_by_group,
    plot_calibration,
    plot_correlation,
    plot_cutoff,
    plot_deciles,
    plot_default_rate_by_repayment,
    plot_default_rate_by_utilization,
    plot_expected_loss_by_grade,
    plot_gains,
    plot_importance,
    plot_limit_by_default,
    plot_roc,
    plot_score_distribution,
)
from credit_risk.sql_analysis import run_queries


def run_all(save: bool = True, verbose: bool = True) -> dict:
    log = _logger(verbose)
    ensure_raw_csv()
    raw = load_raw()
    log(f"Loaded {len(raw):,} rows from the UCI workbook")
    prepared = clean(raw)
    cleaning_report = dict(prepared.attrs.get("cleaning_report", {}))
    data = engineer_features(prepared)
    log(
        f"Clean book: {len(data):,} accounts, "
        f"default rate {data['default_flag'].mean():.2%} "
        f"({int(data['default_flag'].sum()):,} defaults)"
    )

    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data["default_flag"],
        random_state=SEED,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y_train = train["default_flag"].to_numpy()
    y_test = test["default_flag"].to_numpy()
    log(f"Split: train {len(train):,} / holdout {len(test):,}, seed {SEED}")

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        log("Writing exploratory charts from the training sample")
        plot_default_rate_by_repayment(train, FIGURE_DIR / "01_default_rate_by_repayment.png")
        plot_default_rate_by_utilization(train, FIGURE_DIR / "02_default_rate_by_utilization.png")
        plot_limit_by_default(train, FIGURE_DIR / "03_limit_by_default.png")
        plot_correlation(train, FIGURE_DIR / "04_correlation.png")

    x_sc_train = train[SCORECARD_FEATURES]
    x_sc_test = test[SCORECARD_FEATURES]
    x_gb_train = train[GBM_FEATURES]
    x_gb_test = test[GBM_FEATURES]

    log(f"Cross-validating the scorecard ({N_SPLITS} folds, unweighted and balanced)")
    scorecard_cv = compare_family(
        "scorecard",
        cross_validate(scorecard_factory(None), x_sc_train, y_train),
        cross_validate(scorecard_factory("balanced"), x_sc_train, y_train),
    )
    log(
        f"  scorecard CV AUC {scorecard_cv['chosen_cv_auc']:.3f} "
        f"({scorecard_cv['weighting']})"
    )

    log(f"Cross-validating gradient boosting ({N_SPLITS} folds, unweighted and balanced)")
    boosting_cv = compare_family(
        "gradient_boosting",
        cross_validate(gbm_factory(False), x_gb_train, y_train),
        cross_validate(gbm_factory(True), x_gb_train, y_train),
    )
    log(
        f"  boosting CV AUC {boosting_cv['chosen_cv_auc']:.3f} "
        f"({boosting_cv['weighting']})"
    )

    champion = select_champion(scorecard_cv["chosen_cv_auc"], boosting_cv["chosen_cv_auc"])
    log(f"Champion selected on training CV: {champion}")

    scorecard = fit_scorecard(
        train,
        y_train,
        class_weight="balanced" if scorecard_cv["weighting"] == "balanced" else None,
    )
    boosting = gbm_factory(weighted=boosting_cv["weighting"] == "balanced")(x_gb_train, y_train)
    boosting.fit(x_gb_train, y_train)

    scorecard_iso = fit_isotonic(scorecard_cv["oof"], y_train)
    boosting_iso = fit_isotonic(boosting_cv["oof"], y_train)
    sc_raw = scorecard.predict_proba(x_sc_test)[:, 1]
    gb_raw = boosting.predict_proba(x_gb_test)[:, 1]
    sc_cal = apply_isotonic(scorecard_iso, sc_raw)
    gb_cal = apply_isotonic(boosting_iso, gb_raw)

    if champion == "gradient_boosting":
        raw_pd, cal_pd = gb_raw, gb_cal
        oof = boosting_cv["oof"]
    else:
        raw_pd, cal_pd = sc_raw, sc_cal
        oof = scorecard_cv["oof"]

    base_rate = float(y_train.mean())
    families = {
        "scorecard": _family_block(scorecard_cv, y_test, sc_raw, sc_cal, base_rate),
        "gradient_boosting": _family_block(boosting_cv, y_test, gb_raw, gb_cal, base_rate),
    }
    families["scorecard"]["kept_features"] = list(scorecard.kept_features_)
    families["scorecard"]["screened_out_low_iv"] = list(scorecard.screened_out_)
    families["scorecard"]["dropped_wrong_sign"] = list(scorecard.wrong_sign_)
    families["scorecard"]["intercept"] = scorecard.intercept_
    families["scorecard"]["base_points"] = scorecard.base_points_
    families["scorecard"]["coefficients"] = scorecard.coefs_
    families["scorecard"]["psi"] = population_stability_index(scorecard_cv["oof"], sc_raw)
    families["gradient_boosting"]["psi"] = population_stability_index(boosting_cv["oof"], gb_raw)

    baselines = {
        "repayment_status": ranking_metrics(y_test, test["pay_1"].to_numpy()),
        "utilization": ranking_metrics(y_test, test["util_1"].to_numpy()),
    }
    log(
        "Holdout AUC  "
        f"repayment {baselines['repayment_status']['auc']:.3f}  "
        f"scorecard {families['scorecard']['test_auc']:.3f}  "
        f"boosting {families['gradient_boosting']['test_auc']:.3f}"
    )

    deciles = decile_table(y_test, raw_pd, cal_pd)
    reliability_raw = reliability_table(y_test, raw_pd)
    reliability_cal = reliability_table(y_test, cal_pd)
    points = scorecard.predict_score(x_sc_test)

    ead = exposure_at_default(test["limit_bal"], test["bill_amt1"], ASSUMED_CCF)
    loss = expected_loss(cal_pd, ASSUMED_LGD, ead)
    scored = pd.DataFrame(
        {
            "account_id": test["account_id"].to_numpy(),
            "risk_grade": assign_risk_grades(raw_pd),
            "pd": cal_pd,
            "raw_pd": raw_pd,
            "default_flag": y_test,
            "ead": ead,
            "expected_loss": loss,
            "limit_bal": test["limit_bal"].to_numpy(),
            "bill_amt1": test["bill_amt1"].to_numpy(),
            "pay_1": test["pay_1"].to_numpy(),
        }
    )
    grades = grade_summary(scored)
    totals = portfolio_totals(scored)
    slices = pd.DataFrame(
        [
            segment_totals(scored, scored["pay_1"] <= 0, "Current or prepaid"),
            segment_totals(scored, scored["pay_1"] >= 1, "Already late"),
            segment_totals(scored, pd.Series(True, index=scored.index), "Full holdout"),
        ]
    )
    sensitivities = []
    for lgd in LGD_SENSITIVITIES:
        scaled = expected_loss(cal_pd, lgd, ead)
        sensitivities.append(
            {
                "lgd": lgd,
                "ccf": ASSUMED_CCF,
                "total_expected_loss": float(scaled.sum()),
                "el_rate": float(scaled.sum() / ead.sum()),
            }
        )

    fine_curve = cutoff_curve(
        y_test,
        raw_pd,
        loss,
        ead,
        calibrated_pd=cal_pd,
        grid=tuple(np.round(np.linspace(0.20, 1.00, 17), 2)),
    )
    policy_table = cutoff_curve(
        y_test,
        raw_pd,
        loss,
        ead,
        calibrated_pd=cal_pd,
        grid=APPROVAL_GRID,
    )
    policy = policy_row(policy_table, POLICY_APPROVAL_RATE)

    age_band = pd.cut(
        test["age"],
        bins=[-np.inf, 25, 35, 45, 55, np.inf],
        labels=["21–25", "26–35", "36–45", "46–55", "56+"],
    )
    fairness = {
        "sex": group_outcomes(y_test, raw_pd, test["sex_label"], POLICY_APPROVAL_RATE, cal_pd),
        "education": group_outcomes(y_test, raw_pd, test["education_label"], POLICY_APPROVAL_RATE, cal_pd),
        "marriage": group_outcomes(y_test, raw_pd, test["marriage_label"], POLICY_APPROVAL_RATE, cal_pd),
        "age": group_outcomes(y_test, raw_pd, age_band.astype(str), POLICY_APPROVAL_RATE, cal_pd),
    }
    fairness_ratios = {name: four_fifths_ratio(table) for name, table in fairness.items()}

    accepted = acceptance_mask(raw_pd, POLICY_APPROVAL_RATE)
    examples = _adverse_examples(test, scorecard, boosting, cal_pd, points, accepted)

    log("Computing SHAP values for the gradient boosting model")
    importance, explain_method, _, _ = importance_table(boosting, test.assign(default_flag=y_test))
    if explain_method == "shap":
        example_frame = test.iloc[[item["row"] for item in examples]]
        _, example_shap = shap_values_matrix(boosting, example_frame, max_rows=len(example_frame))
        for i, item in enumerate(examples):
            item["shap_reasons"] = shap_reasons(
                boosting.feature_names_,
                example_shap[i],
                test.iloc[item["row"]],
            )
    else:
        for item in examples:
            item["shap_reasons"] = []

    sql_results = run_queries(data, scored)
    eda = {
        "repayment": _band_rates(repayment_band(train["pay_1"]), train["default_flag"]),
        "utilization": _band_rates(utilization_band(train["util_1"]), train["default_flag"]),
    }
    capture = {
        "at_10pct": _capture(y_test, raw_pd, 0.10),
        "at_20pct": _capture(y_test, raw_pd, 0.20),
        "at_30pct": _capture(y_test, raw_pd, 0.30),
        "baseline_at_30pct": _capture(y_test, test["pay_1"].to_numpy(), 0.30),
    }

    metrics = {
        "dataset": {
            "source": "UCI Default of Credit Card Clients",
            "rows": int(len(data)),
            "defaults": int(data["default_flag"].sum()),
            "default_rate": float(data["default_flag"].mean()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
            "currency": "NTD",
            "cleaning": cleaning_report,
        },
        "protocol": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "champion_rule": (
                "Gradient boosting is the champion only if its training CV AUC "
                "exceeds the scorecard by at least 0.005. Otherwise the scorecard "
                "is the decision model."
            ),
            "weight_rule": (
                "Class weights are used only if they improve that family's "
                "training CV AUC by at least 0.005."
            ),
            "excluded_features": ["sex", "marriage"],
            "assumed_lgd": ASSUMED_LGD,
            "assumed_ccf": ASSUMED_CCF,
            "policy_approval_rate": POLICY_APPROVAL_RATE,
        },
        "eda": eda,
        "baselines": baselines,
        "models": families,
        "champion": champion,
        "capture": capture,
        "expected_loss": {
            "assumptions": {"lgd": ASSUMED_LGD, "ccf": ASSUMED_CCF},
            "holdout": totals,
            "slices": slices.to_dict(orient="records"),
            "lgd_sensitivity": sensitivities,
        },
        "policy": policy.to_dict(),
        "fairness": {
            "ratios_vs_highest_group": fairness_ratios,
            "note": (
                "Sex and marital status are not model inputs. Approval rates are "
                "still compared at the 70% cutoff because other features can proxy "
                "for them. The ratio of the lowest group approval rate to the highest "
                "is a screening statistic in the spirit of the four-fifths rule of "
                "thumb. It is not a legal conclusion under ECOA or Regulation B."
            ),
        },
        "explainability_method": explain_method,
        "top_boosting_features": importance.head(8).to_dict(orient="records"),
        "versions": _versions(),
    }

    if save:
        _persist(
            metrics,
            scorecard.iv_table_,
            scorecard.scorecard_table_,
            deciles,
            grades,
            policy_table,
            fine_curve,
            fairness,
            examples,
            sql_results,
            importance,
            slices,
            pd.DataFrame(sensitivities),
            reliability_raw,
            reliability_cal,
        )
        curves = {
            "Repayment status only": (y_test, test["pay_1"].to_numpy()),
            "Scorecard": (y_test, sc_raw),
            "Gradient boosting": (y_test, gb_raw),
        }
        plot_roc(curves, FIGURE_DIR / "05_roc_curves.png")
        plot_gains(curves, FIGURE_DIR / "06_gains_chart.png")
        plot_deciles(deciles, FIGURE_DIR / "07_decile_rates.png")
        plot_calibration(reliability_raw, reliability_cal, FIGURE_DIR / "08_calibration.png")
        plot_score_distribution(points, y_test, FIGURE_DIR / "09_score_distribution.png")
        plot_importance(importance, FIGURE_DIR / "10_shap_importance.png", explain_method)
        plot_cutoff(fine_curve, POLICY_APPROVAL_RATE, FIGURE_DIR / "11_cutoff_tradeoff.png")
        plot_expected_loss_by_grade(grades, FIGURE_DIR / "12_expected_loss_by_grade.png")
        plot_approval_by_group(
            fairness["sex"],
            FIGURE_DIR / "13_approval_by_sex.png",
            "Holdout approval rate by sex at the 70% cutoff",
            footnote=(
                "Sex is not a model input. Same score cutoff for every group. "
                "Coral marks a group under 80% of the highest approval rate."
            ),
        )
        plot_approval_by_group(
            fairness["education"],
            FIGURE_DIR / "14_approval_by_education.png",
            "Holdout approval rate by education at the 70% cutoff",
            footnote=(
                "Education is a model input. Other is a small bin, including undocumented codes. "
                "Coral marks a group under 80% of the highest approval rate."
            ),
        )
        log(f"Wrote figures, tables, and {METRICS_PATH.name}")

    return {
        "metrics": _json_ready(metrics),
        "cleaning_report": cleaning_report,
        "iv_table": scorecard.iv_table_,
        "scorecard_table": scorecard.scorecard_table_,
        "deciles": deciles,
        "grades": grades,
        "cutoff": policy_table,
        "fairness": fairness,
        "examples": examples,
        "sql": sql_results,
        "importance": importance,
        "slices": slices,
        "sensitivities": pd.DataFrame(sensitivities),
        "scorecard": scorecard,
        "boosting": boosting,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def _family_block(cv: dict, y_test, raw, calibrated, base_rate: float) -> dict:
    raw_rank = ranking_metrics(y_test, raw)
    cal_rank = ranking_metrics(y_test, calibrated)
    return {
        "weighting": cv["weighting"],
        "cv_auc": cv["chosen_cv_auc"],
        "cv_auc_std": cv["chosen_cv_auc_std"],
        "cv_ks": cv["chosen_cv_ks"],
        "cv_ks_std": cv["chosen_cv_ks_std"],
        "cv_auc_folds": cv["auc_folds"],
        "unweighted_cv_auc": cv["unweighted_cv_auc"],
        "balanced_cv_auc": cv["balanced_cv_auc"],
        "test_auc": raw_rank["auc"],
        "test_gini": raw_rank["gini"],
        "test_ks": raw_rank["ks"],
        "test_auc_calibrated": cal_rank["auc"],
        "test_gini_calibrated": cal_rank["gini"],
        "test_ks_calibrated": cal_rank["ks"],
        "brier_raw": brier(y_test, raw),
        "brier_calibrated": brier(y_test, calibrated),
        "brier_base_rate": brier(y_test, np.full(len(y_test), base_rate)),
        "ece_raw": expected_calibration_error(y_test, raw),
        "ece_calibrated": expected_calibration_error(y_test, calibrated),
    }


def _band_rates(band: pd.Series, target: pd.Series) -> list[dict]:
    frame = pd.DataFrame({"band": band, "y": target.to_numpy()})
    table = (
        frame.groupby("band", as_index=False)
        .agg(accounts=("y", "size"), default_rate=("y", "mean"))
        .sort_values("band")
    )
    return table.to_dict(orient="records")


def _capture(y, score, depth: float) -> float:
    population, captured = cumulative_gains(y, score)
    idx = max(0, int(round(depth * len(population))) - 1)
    return float(captured[idx])


def _adverse_examples(test, scorecard, boosting, cal_pd, points, accepted) -> list[dict]:
    declined = np.flatnonzero(~accepted)
    marginal = declined[np.argsort(cal_pd[declined], kind="mergesort")][:3]
    deepest = int(declined[np.argmax(cal_pd[declined])])
    chosen = []
    for idx in list(marginal) + [deepest]:
        idx = int(idx)
        if idx not in chosen:
            chosen.append(idx)
    examples = []
    for idx in chosen:
        row = test.iloc[[idx]]
        record = row.iloc[0]
        examples.append(
            {
                "row": idx,
                "account_id": int(record["account_id"]),
                "position": "marginal decline" if idx != deepest else "highest-risk decline",
                "calibrated_pd": float(cal_pd[idx]),
                "scorecard_points": float(points[idx]),
                "pay_1": int(record["pay_1"]),
                "util_1": None if pd.isna(record["util_1"]) else float(record["util_1"]),
                "limit_bal": float(record["limit_bal"]),
                "age": int(record["age"]),
                "education": str(record["education_label"]),
                "sex": str(record["sex_label"]),
                "defaulted": int(record["default_flag"]),
                "scorecard_reasons": scorecard_reasons(scorecard, row),
            }
        )
    # `boosting` is accepted so a future reason-code path can use the fitted model.
    _ = boosting
    return examples


def _versions() -> dict:
    names = ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "matplotlib"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def _persist(
    metrics,
    iv_table,
    scorecard_table,
    deciles,
    grades,
    policy_table,
    fine_curve,
    fairness,
    examples,
    sql_results,
    importance,
    slices,
    sensitivities,
    reliability_raw,
    reliability_cal,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "information_value": iv_table,
        "scorecard": scorecard_table,
        "deciles": deciles,
        "risk_grades": grades,
        "cutoff_policy": policy_table,
        "cutoff_curve": fine_curve,
        "expected_loss_slices": slices,
        "lgd_sensitivity": sensitivities,
        "shap_importance": importance,
        "calibration_raw": reliability_raw,
        "calibration_calibrated": reliability_cal,
    }
    for name, table in fairness.items():
        tables[f"fairness_{name}"] = table
    for name, table in sql_results.items():
        tables[f"sql_{name}"] = table
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)

    reason_rows = []
    for example in examples:
        for rank, reason in enumerate(example["scorecard_reasons"], start=1):
            reason_rows.append(
                {
                    "account_id": example["account_id"],
                    "position": example["position"],
                    "calibrated_pd": example["calibrated_pd"],
                    "scorecard_points": example["scorecard_points"],
                    "rank": rank,
                    "source": "scorecard",
                    "reason": reason["reason"],
                    "shortfall": reason["shortfall"],
                }
            )
        for rank, reason in enumerate(example.get("shap_reasons", []), start=1):
            reason_rows.append(
                {
                    "account_id": example["account_id"],
                    "position": example["position"],
                    "calibrated_pd": example["calibrated_pd"],
                    "scorecard_points": example["scorecard_points"],
                    "rank": rank,
                    "source": "shap",
                    "reason": reason["reason"],
                    "shortfall": reason["shap"],
                }
            )
    reason_table = pd.DataFrame(reason_rows)
    reason_table.to_csv(TABLE_DIR / "adverse_action_examples.csv", index=False)

    payload = _json_ready(metrics)
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    _write_excel(payload, tables, reason_table)


def _write_excel(metrics: dict, tables: dict, reasons: pd.DataFrame) -> None:
    summary_rows = [
        ("Dataset", metrics["dataset"]["source"]),
        ("Accounts", metrics["dataset"]["rows"]),
        ("Defaults", metrics["dataset"]["defaults"]),
        ("Portfolio default rate", metrics["dataset"]["default_rate"]),
        ("Holdout accounts", metrics["dataset"]["test_rows"]),
        ("Champion", metrics["champion"]),
        ("Scorecard holdout AUC", metrics["models"]["scorecard"]["test_auc"]),
        ("Scorecard holdout KS", metrics["models"]["scorecard"]["test_ks"]),
        ("Scorecard holdout Gini", metrics["models"]["scorecard"]["test_gini"]),
        ("Boosting holdout AUC", metrics["models"]["gradient_boosting"]["test_auc"]),
        ("Boosting holdout KS", metrics["models"]["gradient_boosting"]["test_ks"]),
        ("Boosting holdout Gini", metrics["models"]["gradient_boosting"]["test_gini"]),
        ("Repayment-status AUC", metrics["baselines"]["repayment_status"]["auc"]),
        ("Assumed LGD", metrics["expected_loss"]["assumptions"]["lgd"]),
        ("Assumed CCF on undrawn", metrics["expected_loss"]["assumptions"]["ccf"]),
        ("Holdout EAD", metrics["expected_loss"]["holdout"]["total_ead"]),
        ("Holdout expected loss", metrics["expected_loss"]["holdout"]["total_expected_loss"]),
        ("Holdout EL / EAD", metrics["expected_loss"]["holdout"]["el_rate"]),
        ("Policy approval rate", metrics["policy"]["approval_rate"]),
        ("Bad rate at policy", metrics["policy"]["observed_bad_rate"]),
        ("Defaults removed at policy", metrics["policy"]["defaults_excluded_share"]),
    ]
    summary = pd.DataFrame(summary_rows, columns=["item", "value"])
    sheets = {
        "Summary": summary,
        "Metrics": _metrics_sheet(metrics),
        "InformationValue": tables["information_value"],
        "Scorecard": tables["scorecard"],
        "Deciles": tables["deciles"],
        "RiskGrades": tables["risk_grades"],
        "Cutoff": tables["cutoff_policy"],
        "FairnessSex": tables["fairness_sex"],
        "FairnessEducation": tables["fairness_education"],
        "AdverseAction": reasons,
        "SQL_EL_by_grade": tables["sql_expected_loss_by_risk_grade"],
    }
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            _format_sheet(writer.book[name], frame)


def _metrics_sheet(metrics: dict) -> pd.DataFrame:
    rows = []
    for model in ("scorecard", "gradient_boosting"):
        block = metrics["models"][model]
        for key in (
            "weighting",
            "cv_auc",
            "cv_ks",
            "test_auc",
            "test_gini",
            "test_ks",
            "brier_raw",
            "brier_calibrated",
            "ece_raw",
            "ece_calibrated",
            "psi",
        ):
            rows.append({"model": model, "metric": key, "value": block[key]})
    return pd.DataFrame(rows)


def _format_sheet(worksheet, frame: pd.DataFrame) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="0B2545")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left")
    for index, column in enumerate(frame.columns, start=1):
        values = [str(column)] + ["" if pd.isna(v) else str(v) for v in frame[column].head(200)]
        width = min(max(len(value) for value in values) + 2, 48)
        worksheet.column_dimensions[get_column_letter(index)].width = width
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.freeze_panes = "A2"
    worksheet.sheet_view.showGridLines = False


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _logger(verbose: bool):
    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    return log
