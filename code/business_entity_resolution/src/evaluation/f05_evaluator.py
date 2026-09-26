from sklearn.metrics import precision_score, recall_score, fbeta_score


def calculate_f05(y_true, y_pred):
    """
    Calculate F0.5 score.

    F0.5 gives more importance to precision than recall.
    """
    return fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)


def evaluate_predictions(y_true, y_pred):
    """
    Calculate precision, recall and F0.5 score.
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f05 = calculate_f05(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f05
    }


if __name__ == "__main__":
    # Example data for testing
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 1, 0, 0, 0]

    results = evaluate_predictions(y_true, y_pred)

    print("Evaluation Results")
    print("------------------")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F0.5      : {results['f0.5']:.4f}")