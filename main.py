from src.data.load_data import loadData
from src.data.preprocess import split
from src.models.train_model import trainKNN, trainGNB, trainDT
from src.models.evaluate_model import evaluate
from src.visualization.visualize import confusionMatrix, plotROC

# Load and preprocess data
X, y = loadData("data/raw/Healthcare-Diabetes.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)

# Train models
knn, best_k = trainKNN(X_train, y_train)
gnb = trainGNB(X_train, y_train)
dt = trainDT(X_train, y_train)

# Evaluate models
models = {"KNN": knn, "GaussianNB": gnb, "Decision Tree": dt}
colors = {"KNN": "red", "GaussianNB": "blue", "Decision Tree": "green"}

# Print confusion matrices
for name, model in models.items():
    scores, cm = evaluate(model, X_val, y_val)
    print(f"\n{name} Validation Scores: {scores}")
    confusionMatrix(name, cm, f"Validation Confusion Matrix - {name}")

# Print CM for best model
scores2, cm2 = evaluate(dt, X_test, y_test)
print(f"Decision Tree Test Scores: {scores2}")
confusionMatrix("Decision Tree", cm2, "Test Confusion Matrix - Decision Tree")

# Compare models with ROC curve
plotROC(models, X_val, y_val, colors)