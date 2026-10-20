from functools import lru_cache

import joblib
import numpy as np

from models.train_rul import (
    OUTPUT,
    TEST_FILE,
    make_features,
    read_cmapss,
)


# Keep the existing model unchanged.
MODEL_PATH = OUTPUT / "rul_baseline.joblib"

# Experimental five-member ensemble created in Step 40.
ENSEMBLE_PATH = OUTPUT / "rul_ensemble.joblib"


# Prototype review thresholds, not certified safety limits.
LOW_RUL_REVIEW_CYCLES = 30.0
HIGH_DISAGREEMENT_REVIEW_CYCLES = 8.0


@lru_cache(maxsize=1)
def load_rul_model():
    """Load the existing baseline model once."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline model not found: {MODEL_PATH}. "
            "Run python models\\train_rul.py first."
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_rul_ensemble():
    """Load the saved bootstrap ensemble once."""
    if not ENSEMBLE_PATH.is_file():
        raise FileNotFoundError(
            f"Ensemble not found: {ENSEMBLE_PATH}. "
            "Run python models\\train_rul_ensemble.py first."
        )

    bundle = joblib.load(ENSEMBLE_PATH)

    # Step 40 saves a dictionary containing the member models.
    if isinstance(bundle, dict):
        models = bundle.get("models")
    else:
        # Support a plain list of models as well.
        models = bundle

    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError(
            "The ensemble file does not contain a non-empty "
            "list of models under the 'models' key."
        )

    for index, model in enumerate(models, start=1):
        if not callable(getattr(model, "predict", None)):
            raise ValueError(
                f"Ensemble member {index} does not support predict()."
            )

    return list(models)


def get_latest_rul_predictions():
    """
    Predict remaining useful life for the latest observed cycle
    of every engine in NASA C-MAPSS FD001.

    The ensemble standard deviation represents model disagreement.
    It is not a calibrated confidence interval or failure probability.
    """
    if not TEST_FILE.is_file():
        raise FileNotFoundError(
            f"NASA test dataset not found: {TEST_FILE}"
        )

    baseline_model = load_rul_model()
    ensemble_models = load_rul_ensemble()

    test_df = read_cmapss(TEST_FILE)
    test_features = make_features(test_df)

    # Select the latest available cycle for each engine.
    last_indices = (
        test_df.groupby("unit", sort=True)["cycle"]
        .idxmax()
        .to_numpy()
    )

    latest_features = (
        test_features.loc[last_indices]
        .reset_index(drop=True)
    )

    latest_rows = (
        test_df.loc[last_indices, ["unit", "cycle"]]
        .reset_index(drop=True)
    )

    # Existing baseline predictions are retained for comparison.
    baseline_predictions = np.maximum(
        0.0,
        np.asarray(
            baseline_model.predict(latest_features),
            dtype=float,
        ),
    )

    # The trained target uses a 125-cycle RUL cap. Apply the same supported
    # output range before computing mean, disagreement, and member percentiles.
    RUL_TARGET_CAP_CYCLES = 125.0
    baseline_predictions = np.clip(
        np.asarray(baseline_predictions, dtype=float),
        0.0,
        RUL_TARGET_CAP_CYCLES,
    )

    # Shape: number_of_models x number_of_engines.
    member_predictions = np.vstack([
        np.clip(
            np.asarray(model.predict(latest_features), dtype=float),
            0.0,
            RUL_TARGET_CAP_CYCLES,
        )
        for model in ensemble_models
    ])

    # Ensemble estimate and model disagreement.
    ensemble_mean = np.mean(member_predictions, axis=0)
    ensemble_std = np.std(member_predictions, axis=0, ddof=0)

    # Descriptive spread of member predictions, not a calibrated interval.
    p10 = np.percentile(member_predictions, 10, axis=0)
    p90 = np.percentile(member_predictions, 90, axis=0)

    predictions = []

    for index in range(len(latest_rows)):
        rul = float(ensemble_mean[index])
        disagreement = float(ensemble_std[index])

        review_reasons = []

        if rul <= LOW_RUL_REVIEW_CYCLES:
            review_reasons.append("LOW_PREDICTED_RUL")

        if disagreement >= HIGH_DISAGREEMENT_REVIEW_CYCLES:
            review_reasons.append("HIGH_MODEL_DISAGREEMENT")

        evidence_status = (
            "REVIEW" if review_reasons else "PROVISIONAL"
        )

        predictions.append({
            "unit": int(latest_rows.loc[index, "unit"]),
            "last_observed_cycle": int(
                latest_rows.loc[index, "cycle"]
            ),

            # Keep this existing key for frontend compatibility.
            "predicted_rul_cycles": round(rul, 2),

            # Ensemble diagnostics.
            "baseline_rul_cycles": round(
                float(baseline_predictions[index]), 2
            ),
            "model_disagreement_cycles": round(disagreement, 2),
            "member_prediction_p10_cycles": round(
                float(p10[index]), 2
            ),
            "member_prediction_p90_cycles": round(
                float(p90[index]), 2
            ),
            "ensemble_size": len(ensemble_models),

            # Evidence gate output.
            "evidence_status": evidence_status,
            "review_reasons": review_reasons,
        })

    return predictions