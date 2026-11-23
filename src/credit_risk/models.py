"""Scorecard logistic regression and gradient boosting, with fold-wise validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from credit_risk.config import (
    CHAMPION_AUC_MARGIN,
    GBM_PARAMS,
    IV_MIN,
    N_SPLITS,
    PDO,
    SCORECARD_FEATURES,
    SEED,
    TARGET_ODDS,
    TARGET_SCORE,
    WEIGHT_AUC_MARGIN,
)
from credit_risk.metrics import ks_statistic, ranking_metrics, roc_auc
from credit_risk.woe import WoeTransformer


def points_factor(pdo: float = PDO) -> float:
    return pdo / np.log(2.0)


def points_offset(pdo: float = PDO, target_score: float = TARGET_SCORE, target_odds: float = TARGET_ODDS) -> float:
    return target_score - points_factor(pdo) * np.log(target_odds)


def probability_to_score(pd, pdo: float = PDO, target_score: float = TARGET_SCORE, target_odds: float = TARGET_ODDS) -> np.ndarray:
    """Higher score means lower risk. ``target_score`` sits at ``target_odds`` goods per bad."""
    p = np.clip(np.asarray(pd, dtype=float), 1e-6, 1 - 1e-6)
    odds_good = (1.0 - p) / p
    return points_offset(pdo, target_score, target_odds) + points_factor(pdo) * np.log(odds_good)


class ScorecardModel(BaseEstimator, ClassifierMixin):
    """WOE binning plus L2 logistic regression, scaled to a points scorecard.

    Characteristics with information value below ``iv_min`` are left out of
    the regression. After the first fit, characteristics whose coefficient
    has the wrong sign (positive, which would treat a safer WOE as riskier)
    are dropped and the regression is refit once. That keeps the multivariate
    score pointing the same way as the univariate evidence.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight=None, iv_min: float = IV_MIN):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.iv_min = iv_min

    def fit(self, X, y):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.woe_ = WoeTransformer(features=list(frame.columns))
        self.woe_.fit(frame, target)
        self.iv_table_ = self.woe_.information_value()
        eligible = [
            feature
            for feature, iv in self.woe_.iv_.items()
            if iv >= self.iv_min
        ]
        if not eligible:
            # Keep the strongest characteristic so the scorecard still scores.
            eligible = [max(self.woe_.iv_, key=self.woe_.iv_.get)]
        self.screened_out_ = [feature for feature in self.woe_.features_ if feature not in eligible]

        z = self.woe_.transform(frame)
        kept, logit = self._fit_with_sign_check(z, target, eligible)
        self.kept_features_ = kept
        self.logit_ = logit
        self.coefs_ = {feature: float(coef) for feature, coef in zip(kept, logit.coef_.ravel())}
        self.intercept_ = float(logit.intercept_[0])
        self.factor_ = points_factor()
        self.offset_ = points_offset()
        self.base_points_ = self.offset_ - self.factor_ * self.intercept_
        train_woe = z[kept]
        self.avg_woe_ = {feature: float(train_woe[feature].mean()) for feature in kept}
        self.avg_points_ = {
            feature: -self.factor_ * self.coefs_[feature] * self.avg_woe_[feature] for feature in kept
        }
        self.scorecard_table_ = self._scorecard_table()
        return self

    def _new_logit(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
        )

    def _fit_with_sign_check(self, z: pd.DataFrame, y: np.ndarray, eligible: list[str]):
        logit = self._new_logit()
        logit.fit(z[eligible], y)
        coefs = logit.coef_.ravel()
        wrong = [feature for feature, coef in zip(eligible, coefs) if coef > 0]
        kept = [feature for feature in eligible if feature not in wrong]
        self.wrong_sign_ = wrong
        if not kept:
            kept = list(eligible)
            self.wrong_sign_ = []
            return kept, logit
        if wrong:
            logit = self._new_logit()
            logit.fit(z[kept], y)
        return kept, logit

    def _scorecard_table(self) -> pd.DataFrame:
        rows = []
        for feature in self.woe_.features_:
            table = self.woe_.tables_[feature]
            coef = self.coefs_.get(feature)
            for record in table.itertuples(index=False):
                points = np.nan
                if coef is not None:
                    points = -self.factor_ * coef * float(record.woe)
                rows.append(
                    {
                        "feature": feature,
                        "bin": record.bin,
                        "accounts": int(record.n),
                        "defaults": int(record.n_bad),
                        "default_rate": float(record.n_bad / record.n) if record.n else np.nan,
                        "woe": float(record.woe),
                        "iv_contrib": float(record.iv_contrib),
                        "coefficient": coef if coef is not None else np.nan,
                        "points": points,
                        "in_model": feature in self.kept_features_,
                    }
                )
        return pd.DataFrame(rows)

    def transform_woe(self, X) -> pd.DataFrame:
        z = self.woe_.transform(X)
        return z[self.kept_features_]

    def predict_proba(self, X):
        z = self.transform_woe(X)
        return self.logit_.predict_proba(z)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_score(self, X) -> np.ndarray:
        """Points score. Higher is safer. Matches the PDO transform of the model PD."""
        return probability_to_score(self.predict_proba(X)[:, 1])

    def attribute_points(self, X) -> pd.DataFrame:
        z = self.transform_woe(X)
        data = {}
        for feature in self.kept_features_:
            data[feature] = -self.factor_ * self.coefs_[feature] * z[feature].to_numpy()
        points = pd.DataFrame(data, index=z.index)
        points["base_points"] = self.base_points_
        points["score"] = points.sum(axis=1)
        return points


class GradientBoostingModel(BaseEstimator, ClassifierMixin):
    """Median-imputed XGBoost classifier. Imputation is fit on the training rows only."""

    def __init__(
        self,
        n_estimators: int = GBM_PARAMS["n_estimators"],
        learning_rate: float = GBM_PARAMS["learning_rate"],
        max_depth: int = GBM_PARAMS["max_depth"],
        min_child_weight: float = GBM_PARAMS["min_child_weight"],
        subsample: float = GBM_PARAMS["subsample"],
        colsample_bytree: float = GBM_PARAMS["colsample_bytree"],
        reg_lambda: float = GBM_PARAMS["reg_lambda"],
        scale_pos_weight: float = 1.0,
        random_state: int = SEED,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_ = list(frame.columns)
        self.imputer_ = SimpleImputer(strategy="median")
        transformed = self.imputer_.fit_transform(frame[self.feature_names_])
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        self.model_.fit(transformed, np.asarray(y).astype(int))
        self.classes_ = np.array([0, 1])
        return self

    def _matrix(self, X) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
        return self.imputer_.transform(frame[self.feature_names_])

    def predict_proba(self, X):
        return self.model_.predict_proba(self._matrix(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def cross_validate(factory, X: pd.DataFrame, y, n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    """Out-of-fold probabilities and fold metrics.

    ``factory(train_frame, train_target)`` builds an unfitted estimator so a
    weighted model can set its weight from that fold only.
    """
    target = np.asarray(y).astype(int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(target), dtype=float)
    aucs, kss = [], []
    for train_idx, valid_idx in splitter.split(X, target):
        x_train = X.iloc[train_idx]
        y_train = target[train_idx]
        estimator = factory(x_train, y_train)
        estimator.fit(x_train, y_train)
        proba = estimator.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = proba
        aucs.append(roc_auc(target[valid_idx], proba))
        kss.append(ks_statistic(target[valid_idx], proba))
    return {
        "oof": oof,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "ks_mean": float(np.mean(kss)),
        "ks_std": float(np.std(kss, ddof=1)),
        "auc_folds": [float(v) for v in aucs],
        "ks_folds": [float(v) for v in kss],
    }


def fit_isotonic(oof_probability, y) -> IsotonicRegression:
    """Calibrator fit on out-of-fold scores, not on in-sample scores."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(oof_probability, dtype=float), np.asarray(y).astype(int))
    return calibrator


def apply_isotonic(calibrator: IsotonicRegression, probability) -> np.ndarray:
    return np.asarray(calibrator.predict(np.asarray(probability, dtype=float)), dtype=float)


def select_weighting(unweighted_auc: float, weighted_auc: float, margin: float = WEIGHT_AUC_MARGIN) -> str:
    if weighted_auc >= unweighted_auc + margin:
        return "balanced"
    return "unweighted"


def select_champion(scorecard_auc: float, gbm_auc: float, margin: float = CHAMPION_AUC_MARGIN) -> str:
    """Use boosting only when it beats the scorecard by more than ``margin`` AUC."""
    if gbm_auc >= scorecard_auc + margin:
        return "gradient_boosting"
    return "scorecard"


def scale_pos_weight(y) -> float:
    target = np.asarray(y).astype(int)
    positives = max(int(target.sum()), 1)
    return float((len(target) - positives) / positives)


def scorecard_factory(class_weight=None):
    def factory(_x, _y):
        return ScorecardModel(class_weight=class_weight)

    return factory


def gbm_factory(weighted: bool = False, **overrides):
    def factory(_x, y):
        params = dict(GBM_PARAMS)
        params.update(overrides)
        if weighted:
            params["scale_pos_weight"] = scale_pos_weight(y)
        else:
            params["scale_pos_weight"] = 1.0
        return GradientBoostingModel(**params)

    return factory


def fit_scorecard(X: pd.DataFrame, y, class_weight=None) -> ScorecardModel:
    model = ScorecardModel(class_weight=class_weight)
    model.fit(X[SCORECARD_FEATURES], y)
    return model


def compare_family(name: str, unweighted: dict, weighted: dict) -> dict:
    choice = select_weighting(unweighted["auc_mean"], weighted["auc_mean"])
    chosen = weighted if choice == "balanced" else unweighted
    return {
        "family": name,
        "weighting": choice,
        "unweighted_cv_auc": unweighted["auc_mean"],
        "unweighted_cv_auc_std": unweighted["auc_std"],
        "unweighted_cv_ks": unweighted["ks_mean"],
        "balanced_cv_auc": weighted["auc_mean"],
        "balanced_cv_auc_std": weighted["auc_std"],
        "balanced_cv_ks": weighted["ks_mean"],
        "chosen_cv_auc": chosen["auc_mean"],
        "chosen_cv_auc_std": chosen["auc_std"],
        "chosen_cv_ks": chosen["ks_mean"],
        "chosen_cv_ks_std": chosen["ks_std"],
        "auc_folds": chosen["auc_folds"],
        "oof": chosen["oof"],
    }


def holdout_ranking(y, raw_probability, calibrated_probability) -> dict:
    raw = ranking_metrics(y, raw_probability)
    calibrated = ranking_metrics(y, calibrated_probability)
    return {"raw": raw, "calibrated": calibrated}

