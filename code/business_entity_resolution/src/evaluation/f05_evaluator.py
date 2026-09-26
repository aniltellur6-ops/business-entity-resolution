import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score

def calculate_macro_f05(y_true, y_pred, s1_ids):
    """
    Calculate the competition's per-entity macro-averaged F0.5 score.
    For each source1_entity_id, we calculate the F0.5 score.
    If an entity is a singleton (no true matches) and we predict 0 matches, F0.5 = 1.0.
    If we predict > 0 matches for a singleton, F0.5 = 0.0.
    """
    df = pd.DataFrame({
        's1_id': s1_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    entity_f05_scores = []
    
    for s1_id, group in df.groupby('s1_id'):
        y_t = group['y_true'].values
        y_p = group['y_pred'].values
        
        # Singleton logic
        if sum(y_t) == 0:
            if sum(y_p) == 0:
                entity_f05_scores.append(1.0)
            else:
                entity_f05_scores.append(0.0)
        else:
            # Note: zero_division=0 handles when y_p has all zeros but y_t has ones
            score = fbeta_score(y_t, y_p, beta=0.5, zero_division=0)
            entity_f05_scores.append(score)
            
    return sum(entity_f05_scores) / len(entity_f05_scores) if entity_f05_scores else 0.0

def evaluate_predictions(y_true, y_pred, s1_ids=None):
    """
    Calculate global precision, recall and the macro-averaged F0.5 score.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if s1_ids is not None:
        f05 = calculate_macro_f05(y_true, y_pred, s1_ids)
    else:
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f05
    }