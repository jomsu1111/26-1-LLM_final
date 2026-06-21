import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cost import ACTIONS, action_correct, action_total_tokens, utility
from src.features import feature_names, row_to_features
from src.oracle import oracle_action


@dataclass
class ControllerBundle:
    model: object
    controller_type: str
    classifier_type: str
    lambda_value: float
    cost_normalizer: float
    feature_names: List[str]
    label_counts: Dict[str, int]
    variant: str = "oracle_classifier"
    calibration_thresholds: Optional[Dict[str, float]] = None
    validation_utility: Optional[float] = None
    validation_action_counts: Optional[Dict[str, int]] = None
    cost_models: Optional[Dict[str, object]] = None

    def predict_actions(self, rows: List[Dict]) -> List[str]:
        x = np.asarray([row_to_features(row, self.controller_type) for row in rows], dtype=float)
        if self.variant == "value_predictor":
            return self._predict_value_actions(x)
        if self.calibration_thresholds:
            return predict_with_thresholds(self.model, x, self.calibration_thresholds)
        return list(self.model.predict(x))

    def _predict_value_actions(self, x: np.ndarray) -> List[str]:
        if not isinstance(self.model, dict) or not isinstance(self.cost_models, dict):
            raise ValueError("Value predictor bundle is missing action models.")
        scores = []
        for action in ACTIONS:
            correct_prob = predict_positive_probability(self.model[action], x)
            predicted_cost = np.maximum(0.0, np.asarray(self.cost_models[action].predict(x), dtype=float))
            scores.append(correct_prob - self.lambda_value * (predicted_cost / self.cost_normalizer))
        best_indices = np.argmax(np.vstack(scores).T, axis=1)
        return [ACTIONS[idx] for idx in best_indices]


def build_classifier(classifier_type: str, seed: int = 42):
    if classifier_type == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
    if classifier_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier_type: {classifier_type}")


def train_oracle_action_classifier(
    rows: List[Dict],
    controller_type: str,
    classifier_type: str,
    lambda_value: float,
    cost_normalizer: float,
    seed: int = 42,
    validation_rows: Optional[List[Dict]] = None,
    calibrate: bool = False,
) -> ControllerBundle:
    x = np.asarray([row_to_features(row, controller_type) for row in rows], dtype=float)
    y = np.asarray([oracle_action(row, lambda_value, cost_normalizer) for row in rows])
    label_counts = {action: int((y == action).sum()) for action in ACTIONS}
    if len(set(y.tolist())) == 1:
        model = ConstantActionClassifier(y[0])
    else:
        model = build_classifier(classifier_type, seed=seed)
        model.fit(x, y)
    calibration_thresholds = None
    validation_utility = None
    validation_action_counts = None
    if calibrate and validation_rows:
        calibration_thresholds, validation_utility, validation_action_counts = calibrate_classifier_thresholds(
            model=model,
            rows=validation_rows,
            controller_type=controller_type,
            lambda_value=lambda_value,
            cost_normalizer=cost_normalizer,
        )
    return ControllerBundle(
        model=model,
        controller_type=controller_type,
        classifier_type=classifier_type,
        lambda_value=lambda_value,
        cost_normalizer=cost_normalizer,
        feature_names=feature_names(controller_type),
        label_counts=label_counts,
        variant="oracle_classifier",
        calibration_thresholds=calibration_thresholds,
        validation_utility=validation_utility,
        validation_action_counts=validation_action_counts,
    )


class ConstantActionClassifier:
    def __init__(self, action: str):
        self.action = action

    def predict(self, x):
        return np.asarray([self.action] * len(x))

    def predict_proba(self, x):
        classes = np.asarray([self.action])
        probs = np.ones((len(x), 1), dtype=float)
        self.classes_ = classes
        return probs


class ConstantProbabilityModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.asarray([self.value] * len(x), dtype=float)

    def predict_proba(self, x):
        p = min(max(self.value, 0.0), 1.0)
        return np.asarray([[1.0 - p, p]] * len(x), dtype=float)


def train_value_predictor(
    rows: List[Dict],
    controller_type: str,
    lambda_value: float,
    cost_normalizer: float,
    seed: int = 42,
) -> ControllerBundle:
    x = np.asarray([row_to_features(row, controller_type) for row in rows], dtype=float)
    correct_models = {}
    cost_models = {}
    label_counts = {action: 0 for action in ACTIONS}
    for action in ACTIONS:
        y_correct = np.asarray([action_correct(row, action) for row in rows], dtype=int)
        label_counts[action] = int(y_correct.sum())
        if len(set(y_correct.tolist())) == 1:
            correct_models[action] = ConstantProbabilityModel(float(y_correct[0]))
        else:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
                ]
            )
            model.fit(x, y_correct)
            correct_models[action] = model

        y_cost = np.asarray([action_total_tokens(row, action) for row in rows], dtype=float)
        cost_model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=seed,
        )
        cost_model.fit(x, y_cost)
        cost_models[action] = cost_model

    return ControllerBundle(
        model=correct_models,
        controller_type=controller_type,
        classifier_type="value_predictor",
        lambda_value=lambda_value,
        cost_normalizer=cost_normalizer,
        feature_names=feature_names(controller_type),
        label_counts=label_counts,
        variant="value_predictor",
        cost_models=cost_models,
    )


def predict_positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        classes = getattr(model, "classes_", np.asarray([0, 1]))
        if 1 in classes:
            idx = list(classes).index(1)
            return np.asarray(probs[:, idx], dtype=float)
        return np.zeros(len(x), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def predict_with_thresholds(model: object, x: np.ndarray, thresholds: Dict[str, float]) -> List[str]:
    if not hasattr(model, "predict_proba"):
        return list(model.predict(x))
    probs = model.predict_proba(x)
    classes = list(getattr(model, "classes_", []))
    actions = []
    for row_probs in probs:
        prob_by_action = {action: 0.0 for action in ACTIONS}
        for cls, prob in zip(classes, row_probs):
            prob_by_action[str(cls)] = float(prob)
        chosen = max(ACTIONS, key=lambda action: prob_by_action[action])
        if chosen != "STOP" and prob_by_action[chosen] < thresholds.get(chosen, 0.0):
            chosen = "STOP"
        actions.append(chosen)
    return actions


def calibrate_classifier_thresholds(
    model: object,
    rows: List[Dict],
    controller_type: str,
    lambda_value: float,
    cost_normalizer: float,
) -> tuple[Dict[str, float], float, Dict[str, int]]:
    x = np.asarray([row_to_features(row, controller_type) for row in rows], dtype=float)
    grid = np.linspace(0.0, 0.95, 20)
    best_thresholds = {"VERIFY": 0.0, "SC-3": 0.0}
    best_utility = -float("inf")
    best_avg_cost = float("inf")
    best_counts = {action: 0 for action in ACTIONS}
    for verify_threshold in grid:
        for sc3_threshold in grid:
            thresholds = {"VERIFY": float(verify_threshold), "SC-3": float(sc3_threshold)}
            actions = predict_with_thresholds(model, x, thresholds)
            avg_utility = float(np.mean([utility(row, action, lambda_value, cost_normalizer) for row, action in zip(rows, actions)]))
            # Tie-break toward cheaper, more selective policies.
            avg_cost = float(np.mean([action_total_tokens(row, action) for row, action in zip(rows, actions)]))
            if avg_utility > best_utility + 1e-12 or (abs(avg_utility - best_utility) <= 1e-12 and avg_cost < best_avg_cost):
                best_thresholds = thresholds
                best_utility = avg_utility
                best_avg_cost = avg_cost
                best_counts = {action: int(actions.count(action)) for action in ACTIONS}
    return best_thresholds, best_utility, best_counts


def length_threshold_actions(rows: List[Dict]) -> List[str]:
    lengths = [float(row.get("question_token_length") or 0) for row in rows]
    threshold = float(np.median(lengths)) if lengths else 0.0
    return ["SC-3" if float(row.get("question_token_length") or 0) >= threshold else "STOP" for row in rows]


def random_actions(rows: List[Dict], seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    return list(rng.choice(ACTIONS, size=len(rows)))


def save_controller(bundle: ControllerBundle, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_controller(path: str) -> ControllerBundle:
    with open(path, "rb") as f:
        return pickle.load(f)
