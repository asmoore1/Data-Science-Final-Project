import pandas as pd

def loadData(filename):
    data = pd.read_csv(filename)
    X = data.drop(columns=["Outcome", "Id", "Pregnancies", "BloodPressure"])
    y = data["Outcome"]
    return X, y
