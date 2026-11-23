import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import GBM_FEATURES, SCORECARD_FEATURES, SEED
from credit_risk.fairness import four_fifths_ratio, group_outcomes, scorecard_reasons
from credit_risk.features import engineer_features
from credit_risk.metrics import roc_auc
from credit_risk.models import GradientBoostingModel, ScorecardModel, probability_to_score
from tests.synthetic import make_book


def _prepared(n=800, seed=3):
    return engineer_features(make_book(n, seed=seed))


def test_scorecard_score_matches_the_pdo_transform_of_pd():
    book = _prepared()
    model = ScorecardModel()
    model.fit(book[SCORECARD_FEATURES], book["default_flag"])
    probability = model.predict_proba(book[SCORECARD_FEATURES])[:, 1]
    from_points = model.attribute_points(book[SCORECARD_FEATURES])["score"].to_numpy()
    from_pd = probability_to_score(probability)
    np.testing.assert_allclose(from_points, from_pd, atol=1e-6)
    assert model.predict_score(book[SCORECARD_FEATURES]).shape == (len(book),)


def test_scorecard_beats_chance_and_reasons_are_ordered():
    book = _prepared(900, seed=4)
    train, test = train_test_split(book, test_size=0.3, stratify=book["default_flag"], random_state=SEED)
    model = ScorecardModel()
    model.fit(train[SCORECARD_FEATURES], train["default_flag"])
    proba = model.predict_proba(test[SCORECARD_FEATURES])[:, 1]
    assert roc_auc(test["default_flag"], proba) > 0.70
    riskiest = test.iloc[[int(np.argmax(proba))]]
    reasons = scorecard_reasons(model, riskiest, top_n=4)
    assert reasons
    shortfalls = [reason["shortfall"] for reason in reasons]
    assert shortfalls == sorted(shortfalls, reverse=True)
    assert all(reason["shortfall"] > 0 for reason in reasons)


def test_gradient_boosting_probabilities_sum_to_one():
    book = _prepared(500, seed=5)
    model = GradientBoostingModel(n_estimators=20, max_depth=2, min_child_weight=1, subsample=1.0, colsample_bytree=1.0)
    model.fit(book[GBM_FEATURES], book["default_flag"])
    proba = model.predict_proba(book[GBM_FEATURES])
    assert proba.shape == (len(book), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert roc_auc(book["default_flag"], proba[:, 1]) > 0.70


def test_four_fifths_ratio_on_a_known_cutoff():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    group = pd.Series(["Female", "Female", "Male", "Male"])
    table = group_outcomes(y, score, group, approval_rate=0.5)
    female = table.loc[table["group"] == "Female", "approval_rate"].item()
    male = table.loc[table["group"] == "Male", "approval_rate"].item()
    assert female == 1
    assert male == 0
    assert four_fifths_ratio(table) == 0
