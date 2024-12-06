from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score


def trainKNN(X_train, y_train, k_range=range(3, 11)):
    best_k = 0
    best_recall = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        knn.fit(X_train, y_train)
        recall = recall_score(y_train, knn.predict(X_train))

        print(f"K: {k}, Training Recall: {recall:.4f}")

        if recall > best_recall:
            best_k = k
            best_recall = recall

    print(f"\nOptimal k: {best_k} with training recall: {best_recall * 100:.2f}%")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    return best_knn, best_k

def trainGNB(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def trainDT(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    return dt
