import numpy as np

def predict_with_threshold(proba, classes, positive, threshold):
    classes = np.asarray(classes)
    proba = np.asarray(proba)
    positive_idx = np.flatnonzero(classes == positive)
    if len(positive_idx) == 0:
        raise ValueError("positive não está em classes")
    positive_idx = positive_idx[0]
    base_pred = classes[np.argmax(proba, axis=1)]
    pred = np.where(proba[:, positive_idx] >= threshold, positive, base_pred)
    return pred
