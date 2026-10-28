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

def choose_threshold(proba, y, positive, min_recall=0.9):
    classes = np.unique(y)
    grades = np.linspace(0.05, 0.95, 91)
    max_t = 0
    for t in grades:
        pred = predict_with_threshold(proba, classes, positive, t)
        recall = np.mean(pred[y == positive] == positive)
        if t > max_t and recall >= min_recall:
            max_t = t
    if max_t == 0:
        raise ValueError("Nenhum t atinge o recall pedido")
    return max_t