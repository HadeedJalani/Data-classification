<div align="center">

# 🌸 Data Classification Using AI

### Iris Flower Classification with Machine Learning

A clean, reproducible internship project demonstrating an end-to-end **supervised machine learning classification pipeline** using the classic Iris dataset, feature scaling, K-Nearest Neighbors, model selection, and evaluation.

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-5%20Passed-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-0A66C2?style=for-the-badge)

</div>

---

## 📌 About the Project

**Data Classification Using AI** is an internship project focused on understanding and implementing the fundamentals of supervised machine learning through a practical classification problem.

The project uses the well-known **Iris flower dataset** to classify flowers into three species based on four numerical measurements:

- 🌿 Sepal length
- 🌿 Sepal width
- 🌸 Petal length
- 🌸 Petal width

Rather than treating the task as a black-box prediction problem, the project demonstrates the complete machine learning workflow: **data validation → preprocessing → train/test splitting → feature scaling → model selection → training → prediction → evaluation**.

The implementation intentionally stays focused on the assignment requirements and avoids unnecessary frameworks, databases, APIs, or cloud services.

---

## 🎯 Project Objective

The main objective is to build a reliable classification model that can learn from labeled Iris flower examples and predict the species of previously unseen samples.

The project demonstrates several important machine learning concepts:

> **Data → Preprocessing → Training → Prediction → Evaluation**

It also demonstrates why evaluation should go beyond a single accuracy number by using a **confusion matrix, precision, recall, and F1 score**.

---

## 🧠 Machine Learning Approach

The project uses **K-Nearest Neighbors (KNN)** as the classification algorithm.

KNN predicts the class of a new observation by examining the labels of its nearest training examples. Because KNN is distance-based, feature scaling is an important part of the pipeline.

### Model pipeline

```text
                    IRIS DATASET
                         │
                         ▼
                Data Validation
                         │
                         ▼
             Stratified 80/20 Split
                  ┌──────┴──────┐
                  │             │
               TRAIN          TEST
                  │             │
                  ▼             │
            StandardScaler      │
                  │             │
                  ▼             │
             K Selection        │
                  │             │
                  ▼             │
               KNN Model        │
                  │             │
                  └──────┬──────┘
                         ▼
                      Predict
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Confusion Matrix         Macro F1
```

### Why StandardScaler?

The four Iris measurements are expressed on different numerical ranges. Standardization puts the features on a comparable scale so that distance calculations used by KNN are not dominated by a feature simply because of its numerical magnitude.

The scaler is placed **inside the scikit-learn Pipeline**, meaning it is fitted only using the training data. This prevents test-data leakage.

---

## 📊 Dataset

The Iris benchmark contains:

| Property | Value |
|---|---:|
| Total samples | **150** |
| Classes | **3** |
| Features | **4** |
| Samples per class | **50** |
| Training samples | **120** |
| Testing samples | **30** |

### Target classes

```text
setosa       50 samples
versicolor   50 samples
virginica    50 samples
```

---

## ⚙️ Model Selection

The assignment demonstrates **K = 5** for KNN, and the implementation supports that exact demonstration.

In addition, the project includes a small model-selection step that evaluates **K = 1 through K = 15** using:

- 5-fold cross-validation
- Macro F1 scoring
- Smallest-K tie breaking

The resulting selected value is **K = 5**, matching the assignment demonstration.

To reproduce the exact fixed-K demonstration:

```powershell
python -m src.main --no-tune-k --k 5
```

---

## 📈 Verified Results

The complete pipeline has been executed successfully.

| Metric | Result |
|---|---:|
| Accuracy | **93.33%** |
| Macro Precision | **94.44%** |
| Macro Recall | **93.33%** |
| Macro F1 | **93.27%** |
| Test samples | **30** |
| Correct predictions | **28 / 30** |
| Selected K | **5** |

### Confusion Matrix

```text
                 Predicted
              Setosa  Versicolor  Virginica

Setosa           10       0          0
Versicolor        0      10          0
Virginica         0       2          8
```

The classifier correctly recognizes all **10 Setosa** and **10 Versicolor** test samples. Two **Virginica** samples are classified as Versicolor, giving an overall accuracy of **93.33%**.

---

## 🧪 Testing

The repository includes an automated test suite covering the important parts of the pipeline.

```powershell
pytest -q
```

Verified result:

```text
5 passed
```

Tests cover:

- Dataset schema and dimensions
- Class distribution
- Stratified 80/20 split
- Model construction
- StandardScaler + KNN pipeline
- K selection
- End-to-end classification performance

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HadeedJalani/Data-classification.git
cd Data-classification
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the tests

```bash
pytest -q
```

### 5. Run the complete pipeline

```bash
python -m src.main
```

### 6. Run the exact K=5 assignment demonstration

```bash
python -m src.main --no-tune-k --k 5
```

---

## 📁 Project Structure

```text
Data-classification/
│
├── 📂 data/
│   └── iris.csv                    # Iris dataset
│
├── 📂 src/
│   ├── __init__.py
│   ├── config.py                   # Project configuration
│   ├── data.py                     # Loading, validation and splitting
│   ├── evaluate.py                 # Metrics and reports
│   ├── main.py                     # Main executable pipeline
│   ├── model.py                    # Scaling, KNN and K selection
│   └── visualize.py                # Evaluation visualizations
│
├── 📂 tests/
│   └── test_pipeline.py            # Automated tests
│
├── 📂 docs/
│   ├── demo.py
│   ├── PRESENTATION_OUTLINE.md
│   ├── SUBMISSION_CHECKLIST.md
│   └── TECHNICAL_REPORT.md
│
├── 📂 artifacts/                   # Generated locally at runtime
├── 📂 reports/                     # Generated locally at runtime
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Dataset loading and manipulation |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Scaling, KNN, model selection and evaluation |
| **Matplotlib** | Confusion matrix and K-selection visualizations |
| **Joblib** | Model persistence |
| **Pytest** | Automated testing |

---

## 🔍 Key Engineering Decisions

### Reproducibility

A fixed `random_state=42` is used for the train/test split so the experiment can be reproduced consistently.

### Stratification

The 80/20 split uses stratification to preserve the class distribution between training and testing data.

### Leakage prevention

`StandardScaler` is part of the scikit-learn Pipeline, so preprocessing is learned from training data rather than from the complete dataset.

### Practical scope

The project follows the supplied internship assignment closely. Computer vision and CNNs shown in the assignment as future directions are **not** unnecessarily added to this implementation.

---

## 📦 Generated Artifacts

After running the pipeline, the following outputs are generated locally:

```text
artifacts/
├── confusion_matrix.png
├── iris_knn_model.joblib
├── k_selection.png
└── metrics.json

reports/
└── classification_report.txt
```

These files are intentionally ignored by Git because they are reproducible outputs rather than source code.

---

## 🎓 Internship Project Context

This repository represents the implementation of **Project 2: Data Classification Using AI** from an Artificial Intelligence internship assignment.

The project focuses on demonstrating practical understanding of:

- Supervised learning
- Classification
- Data preprocessing
- Feature scaling
- Train/test methodology
- K-Nearest Neighbors
- Hyperparameter selection
- Confusion matrices
- Precision, recall and F1 score
- Reproducible ML engineering

The goal is not simply to produce a prediction, but to demonstrate the reasoning and engineering workflow behind a complete machine learning solution.

---

## 📜 License

This project is intended primarily for educational and internship demonstration purposes.

---

<div align="center">

## 👨‍💻 Author

### **Hadeed Jalani**

Artificial Intelligence / Machine Learning Project

⭐ If you found this project useful, consider giving the repository a star.

</div>
