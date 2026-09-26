import numpy as np
from sklearn.metrics import fbeta_score


def find_best_threshold(
    y_true,
    probabilities,
    start=0.10,
    end=0.90,
    step=0.01
):
    """
    Find the probability threshold that gives
    the highest F0.5 score.

    F0.5 gives more importance to precision than recall.
    """

    best_threshold = start
    best_f05 = -1.0

    thresholds = np.arange(
        start,
        end + step,
        step
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f05 = fbeta_score(
            y_true,
            predictions,
            beta=0.5,
            zero_division=0
        )

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    return best_threshold, best_f05


if __name__ == "__main__":

    # Example probabilities for testing
    y_true = np.array([
        1, 1, 0, 0, 1,
        0, 1, 0, 1, 0
    ])

    probabilities = np.array([
        0.91, 0.82, 0.21, 0.72, 0.88,
        0.15, 0.61, 0.32, 0.76, 0.18
    ])

    threshold, f05 = find_best_threshold(
        y_true,
        probabilities
    )

    print("Threshold Optimization")
    print("----------------------")
    print(f"Best Threshold : {threshold:.2f}")
    print(f"Best F0.5      : {f05:.4f}")