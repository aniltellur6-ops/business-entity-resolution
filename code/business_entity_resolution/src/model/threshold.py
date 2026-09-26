import numpy as np
from src.evaluation.f05_evaluator import calculate_macro_f05

def find_best_threshold(
    y_true,
    probabilities,
    s1_ids,
    start=0.10,
    end=0.90,
    step=0.01
):
    """
    Find the probability threshold that gives the highest macro F0.5 score.
    """
    best_threshold = start
    best_f05 = -1.0

    thresholds = np.arange(start, end + step, step)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        
        f05 = calculate_macro_f05(y_true, predictions, s1_ids)

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    return best_threshold, best_f05