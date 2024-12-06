from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, random_state=42, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
