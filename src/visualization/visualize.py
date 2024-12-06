import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


cm_colors = {"KNN": "Reds", "GaussianNB": "Blues", "Decision Tree": "Greens"}

def confusionMatrix(modelName, cm, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap=cm_colors[modelName])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
def plotROC(models, X_val, y_val, colors):
    for modelName, model in models.items():
        y_val_proba = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_val_proba)
        auc = roc_auc_score(y_val, y_val_proba)
        plt.plot(fpr, tpr, color=colors[modelName], label=f"{modelName} (AUC = {auc:.2f})")
    
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guessing")
    
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()