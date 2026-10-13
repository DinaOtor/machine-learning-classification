from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def run_part_c():


    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print("Iris Accuracy:", acc)

    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(OUTPUT_DIR / "cm_iris.png")
    plt.close()

    plt.figure(figsize=(12,6))
    plot_tree(model, feature_names=X.columns, class_names=iris.target_names, filled=True)
    plt.savefig(OUTPUT_DIR / "decision_tree_iris.png")
    plt.close()

   