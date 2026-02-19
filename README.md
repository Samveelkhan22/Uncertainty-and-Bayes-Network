# Uncertainty and Bayes Network

This repository contains a Jupyter notebook that explores fundamental concepts in probability and implements two variants of the Naive Bayes classifier. The notebook is structured into three independent problems:

1. **Basic Probability Calculations** – Using a joint probability distribution to compute marginal and conditional probabilities.
2. **Naive Bayes for Player Shortlisting** – Predicting whether a football player will be shortlisted based on league, position, preferred foot, and international cap status.
3. **Naive Bayes Text Classification** – Classifying newsgroup messages into categories using a bag‑of‑words model with Laplace smoothing.

---

## 📓 Notebook Contents

### Problem 1: Probability from a Joint Distribution
- Given a full joint probability table over the Boolean variables `Toothache`, `Cavity`, and `Catch`, the notebook computes:
  - Marginal probabilities: `P(Toothache)`, `P(¬Toothache)`
  - Conditional probabilities: `P(Catch | Cavity)`, `P(Cavity | Catch)`
  - Probabilities with conjunctions: `P(Catch | Cavity ∧ Toothache)`, `P(Cavity | Catch ∧ Toothache)`

### Problem 2: Naive Bayes for Football Player Data
- A small dataset of 14 players with features: `League`, `Position`, `Preferred Foot`, `Capped` (Yes/No) and target `Shortlisted` (True/False).
- The notebook builds a Naive Bayes classifier from scratch (without external ML libraries) to predict the shortlisting probability for two new players.
- Prior and conditional probabilities are estimated from the data; Laplace smoothing is applied to handle missing feature values.
- Output shows the normalized posterior probabilities and the final decision.

### Problem 3: Naive Bayes Text Classification
- A full implementation of a Naive Bayes text classifier that:
  - Preprocesses text (lowercase, tokenisation, removal of non‑word characters).
  - Learns word counts per class and builds a vocabulary.
  - Uses log‑probabilities and Laplace smoothing to avoid zero probabilities.
  - Predicts class labels for test data.
- The code expects two CSV files: `newsgroup_train.csv` and `newsgroup_test.csv`, each containing a `Text` column and a `Label` column.
- Evaluation metrics (accuracy, precision, recall, F1‑score) are calculated using `scikit‑learn`.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- Required packages: `pandas`, `numpy`, `scikit‑learn`
  

Install missing packages with:
```bash
pip install pandas numpy scikit-learn
