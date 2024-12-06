from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate(model, X, y):
    y_pred = model.predict(X)
    scores = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=1),
        "recall": recall_score(y, y_pred, pos_label=1),
        "f1": f1_score(y, y_pred),
    }
    cm = confusion_matrix(y, y_pred)
    return scores, cm


