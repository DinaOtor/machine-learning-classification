# Machine Learning Classification

A Python classification project exploring **Decision Trees** and **Categorical Naive Bayes** using the PlayTennis dataset, plus a Decision Tree classifier for the Iris dataset.

## Features

- Load and encode categorical PlayTennis data.
- Train and compare Decision Tree and Naive Bayes classifiers.
- Display training-set accuracy and confusion matrices for PlayTennis.
- Train/test split on the Iris dataset and generate a classification report.
- Export plots and confusion matrices to the `outputs/` folder.

## Tech stack

Python, pandas, matplotlib, scikit-learn

## Getting started

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Generated figures will appear in `outputs/`.

## Files

- `main.py` - runs both experiment modules.
- `partA_B.py` - PlayTennis Decision Tree vs Naive Bayes.
- `partc.py` - Iris classification and evaluation.
- `playtennis.csv` - sample categorical dataset.
- `Report2.pdf` - accompanying course report.

## Evaluation note

The PlayTennis accuracies are measured on **training data**, not on an independent test set, so they should not be interpreted as generalization performance. The Iris experiment uses a separate stratified test split.

## Acknowledgments

University course project. Add the names and contributions of any project partners before publication, if applicable.
