# Sports vs. Politics Text Classifier

A comparative study of machine learning techniques for binary text classification between **Sports** and **Politics** documents.

**Author:** Saurav Soni (B22AI035)  
**Course:** CSL 7640 — Natural Language Understanding  
**Institution:** IIT Jodhpur  

---

## Overview

This project implements and compares **9 classifier configurations** (3 feature representations × 3 ML techniques) for the task of classifying text documents as either Sports or Politics.

### Feature Representations
| Method | Description |
|--------|-------------|
| **Bag of Words (Unigram)** | Raw word frequency counts |
| **Bag of Words (Bigram)** | Unigram + bigram frequency counts |
| **TF-IDF** | Term frequency weighted by inverse document frequency |

### ML Techniques
| Classifier | Type |
|-----------|------|
| **Multinomial Naive Bayes** | Generative, probabilistic |
| **Logistic Regression** | Discriminative, linear |
| **Linear SVM** | Max-margin, linear |

---

## Results

Evaluated using **stratified 5-fold cross-validation** on a curated dataset of **178 sentences** (93 Sports, 85 Politics).

| Feature | Classifier | Accuracy | F1-Score |
|---------|-----------|----------|----------|
| BoW (Unigram) | **Logistic Regression** | **95.6%** | **95.9%** |
| BoW (Unigram) | Naive Bayes | 95.0% | 95.4% |
| BoW (Unigram) | Linear SVM | 93.3% | 93.6% |
| BoW (Bigram) | Logistic Regression | 94.4% | 94.9% |
| TF-IDF | Linear SVM | 94.4% | 94.9% |
| TF-IDF | Naive Bayes | 94.4% | 94.9% |

**Best model:** BoW (Unigram) + Logistic Regression at **95.6% accuracy**.

---

## Repository Structure

```
├── classifier.py        # Single-file classifier with embedded dataset
├── requirements.txt     # Python dependencies
├── B22AI035_prob4.pdf   # Detailed report (8 pages)
└── README.md            # This file
```

---

## Setup & Usage

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/sauravsoni-iitj/sports-politics-classifier.git
cd sports-politics-classifier
pip install -r requirements.txt
```

### Running the Classifier

```bash
python classifier.py
```

This will:
1. Load the dataset (178 sentences)
2. Run 5-fold cross-validation on all 9 configurations
3. Print a comparison table with accuracy, precision, recall, and F1
4. Train the best model on full data
5. Enter **interactive mode** where you can type sentences for prediction

### Example

```
Enter text: the team won the championship after a thrilling final
Prediction: SPORTS

Enter text: the parliament passed a new bill on healthcare reform
Prediction: POLITICS
```

---

## Dataset

The dataset consists of **178 manually curated sentences** covering diverse sub-topics:

- **Sports (93 sentences):** Cricket, Football, Basketball, Tennis, Olympics, and general sports
- **Politics (85 sentences):** Elections, Parliament, Policy & Governance, International Relations, Administration

Each sentence was written to be unambiguously classifiable. See `dataset.py` for the full dataset.

---

## Report

The detailed report (`B22AI035_prob4.pdf`) covers:
- Data collection methodology
- Dataset description and analysis
- Mathematical formulation of each technique
- Experimental setup and results
- Limitations and future work

---

## Key Findings

1. **Simple methods work well** — Unigram BoW outperforms bigrams and TF-IDF on this dataset
2. **Bigrams hurt on small data** — They inflate the feature space, causing Laplace smoothing dilution
3. **TF-IDF is not always better** — When class vocabularies are already distinct, reweighting adds little value
4. **All classifiers are competitive** — The accuracy gap between best and worst is only 2.3%

---

## License

This is part of an academic assignment for CSL 7640 at IIT Jodhpur.
