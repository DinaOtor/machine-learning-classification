from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def run_part_ab():

   
    print("    Machine Learning Classification Project")
    print("\n" + "^" * 60)

    

    # Load dataset
    df = pd.read_csv(Path(__file__).resolve().parent / "playtennis.csv")

    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target = 'PlayTennis'

    print("\nDataset Loaded")
    print(f"Samples: {len(df)}")
    print(f"Features: {features}")

    # Encoding
    encoder = OrdinalEncoder()
    X = encoder.fit_transform(df[features])

    le = LabelEncoder()
    y = le.fit_transform(df[target])

    # DECISION TREE 
    print("\n Decision Tree Model")

    dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt.fit(X, y)

    y_pred_dt = dt.predict(X)
    acc_dt = accuracy_score(y, y_pred_dt)

    print(f"Accuracy: {acc_dt:.2%}")
    print(f"Tree Depth: {dt.get_depth()}")
    print(f"Number of Leaves: {dt.get_n_leaves()}")

    # Tree image
    plt.figure(figsize=(10, 6))
    plot_tree(dt, feature_names=features, class_names=le.classes_, filled=True)
    plt.title("Decision Tree - PlayTennis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree.png")
    plt.close()

    # Confusion matrix
    cm_dt = confusion_matrix(y, y_pred_dt)
    ConfusionMatrixDisplay(cm_dt, display_labels=le.classes_).plot()
    plt.title("Decision Tree Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "cm_dt.png")
    plt.close()

    # NAIVE BAYES 
    print("\n Naive Bayes Model")

    nb = CategoricalNB()
    nb.fit(X, y)

    y_pred_nb = nb.predict(X)
    acc_nb = accuracy_score(y, y_pred_nb)

    print(f"Accuracy: {acc_nb:.2%}")

    cm_nb = confusion_matrix(y, y_pred_nb)
    ConfusionMatrixDisplay(cm_nb, display_labels=le.classes_).plot()
    plt.title("Naive Bayes Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "cm_nb.png")
    plt.close()

    # COMPARISON 
    print("\nModel Comparison")
    print(f"Decision Tree: {acc_dt:.2%}")
    print(f"Naive Bayes  : {acc_nb:.2%}")

    plt.figure(figsize=(6, 4))
    plt.bar(["Decision Tree", "Naive Bayes"], [acc_dt, acc_nb])
    plt.ylim(0, 1.1)
    plt.title("Model Comparison")
    plt.savefig(OUTPUT_DIR / "comparison.png")
    plt.close()
    print("\n" + "^" * 60)


    