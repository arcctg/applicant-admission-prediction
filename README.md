# Applicant Admission Prediction with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-Deep_Learning-FF6F00.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

## Project Overview

This project implements a **binary classification neural network** to predict university applicant admission outcomes based on a set of non-linear, multi-dimensional academic rules and quota constraints. The full machine learning pipeline is explored: from procedural synthetic data generation with injected anomalies, through exploratory data analysis, to training and comparing multiple neural network architectures and optimizers.

The core challenge is teaching a feedforward neural network to replicate a rule-based admission system involving hard score thresholds, a weighted rating formula, and a privilege-based quota — without the model ever explicitly seeing those rules. The project culminates in production-style inference on a new cohort of applicants and a structured validation against hand-crafted edge cases.

## Key Features

* **Synthetic Data Generation:** A dataset of **1500 applicants** is generated using scores drawn from `N(155, 20)` clipped to [100, 200]. Approximately 13% of applicants receive `privileged` status. **100 anomalous records** are deliberately injected using 8 distinct extreme score patterns (cycling by index), including: all-minimum scores (100–115 range), near-perfect scores with one failing subject (e.g., 190/180/100–115), a low Math score paired with high English and Ukrainian, perfect scores in one subject and minimum in another (e.g., 100/200/200), and all-maximum scores (200/200/200). These anomalies create realistic edge cases that stress-test the model's understanding of the admission rules.

* **Rule-Based Labeling:** Admission decisions are computed deterministically using a specific university policy:
  - A **weighted rating** formula: `rating = 0.4 × math + 0.3 × english + 0.3 × ukrainian`
  - **Hard minimum score thresholds** per subject (e.g., Math ≥ 140, English ≥ 120, Ukrainian ≥ 120 for non-privileged students)
  - A **quota system**: 350 total seats, of which 35 are reserved for privileged applicants (with a lower rating threshold of 144)

* **Exploratory Data Analysis (EDA):** Comprehensive visualizations including score distributions, correlation heatmaps, rating distribution, and admission rate breakdown by privileged status.

* **Data Preprocessing:** Anomaly detection and removal, `MinMaxScaler` normalization, and an 80/20 stratified train/test split.

* **Neural Network Architecture Comparison:** Three Keras Sequential models are trained and benchmarked:
  - **Model 1** — Minimal (4→1 neurons, Sigmoid): establishes a baseline; convergence is very slow (~76% accuracy after 100 epochs due to Sigmoid's vanishing gradient problem).
  - **Model 2** — Medium (16→8 neurons, ReLU): demonstrates the power of ReLU in overcoming stagnation; reaches **~97.3% accuracy** roughly 4× faster than Model 1.
  - **Model 3** — Deep with Dropout (32→16→8 neurons, ReLU + Dropout 0.2): adds regularization to prevent overfitting; validation loss consistently lower than training loss confirms successful generalization.

* **Optimizer Benchmarking:** All five required optimizers — **SGD, Adagrad, Adadelta, RMSProp, and Adam** — are applied to the best-performing architecture. Adam achieves the highest test accuracy (**98.33%**) and lowest loss, and is selected as the production model.

* **Model Evaluation:** The final Adam-optimized model is evaluated on the held-out test set with a full classification report and confusion matrix:
  - Accuracy: **98.33%** | Precision: **98.51%** | Recall: **94.29%** | F1-Score: **96.35%**
  - The model achieves perfect recall (1.00) on the "Rejected" class, ensuring no unqualified applicant bypasses the system.

* **Error Analysis:** False positives and false negatives are extracted and inspected to verify that classification errors are concentrated near decision boundaries — a sign of healthy, non-catastrophic failure modes.

* **Production Inference & Quota Deviation Check:** The best model is deployed on a **new cohort** of 1500 applicants (generated with a different random seed). The predicted admission count is compared against the rule-based ground truth; a deviation exceeding 10% of the total quota (35 seats) is flagged as "critical."

* **Anomalous Case Testing:** 10 hand-crafted edge cases — covering score extremes, hard-threshold violations, boundary ratings, and privilege interactions — are fed to the model to verify that it has genuinely internalized the complex, non-linear admission rules.

* **Excel Export:** The final list of admitted applicants (from the current-year dataset) is exported to `admitted_students.xlsx` for operational use.

## Technologies Used

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras (`Sequential`, `Dense`, `Dropout`)
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing & Metrics:** Scikit-Learn (`MinMaxScaler`, `train_test_split`, `classification_report`, `confusion_matrix`)
* **Data Visualization:** Matplotlib, Seaborn
* **Export:** OpenPyXL (via `pandas.DataFrame.to_excel`)
* **Environment:** Jupyter Notebook / Google Colab

## Dataset

The project uses a **fully synthetic dataset** — no external data source is required. All data is generated programmatically inside the notebook:

| Feature | Description | Range |
|---|---|---|
| `math` | Math exam score | 100 – 200 (with injected anomalies) |
| `english` | English exam score | 100 – 200 (with injected anomalies) |
| `ukrainian` | Ukrainian language exam score | 100 – 200 (with injected anomalies) |
| `privileged` | Privileged applicant status (binary) | 0 or 1 (~10% positive rate) |
| `admitted` | Target label (binary) | 0 = Rejected, 1 = Admitted |

Two separate datasets are generated:
- **Historical dataset** (`seed=42`): 1500 records used for training and testing.
- **Current-year dataset** (`seed=123`): 1500 records used exclusively for production inference.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arcctg/applicant-admission-prediction.git
   cd applicant-admission-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow openpyxl jupyter
   ```

4. **Create the data directory** (required for the production inference section):
   ```bash
   mkdir data
   ```

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   *Open `01_applicant_admission_prediction.ipynb` in your browser and run all cells sequentially.*

## Results and Insights

* **Data Handling:** The 100 injected anomalies (representing 8 distinct extreme score patterns) were deliberately kept in the dataset. This ensured that the model was trained and evaluated against realistic edge cases, proving its ability to handle boundary cases rather than simply removing them.

* **Architecture Matters:** The jump from Model 1 (Sigmoid, no hidden layers) to Model 2 (ReLU, two hidden layers) was the most impactful architectural decision, delivering a ~27 percentage point accuracy gain. The ReLU activation eliminates the vanishing gradient problem that caused Model 1 to stagnate for the first 50 epochs.

* **Dropout as Regularizer:** Model 3's training plots show the characteristic inverted gap — validation loss lower than training loss — confirming that Dropout(0.2) successfully prevented overfitting, even as network depth increased.

* **Optimizer Ranking:** `Adadelta` failed to converge within 100 epochs (~27% accuracy), while `Adam` and `RMSProp` both achieved strong results. Adam's adaptive learning rate made it the clear winner for this classification task.

* **Production Viability:** When applied to the new applicant cohort, the model predicted **308 admissions** versus the expected **350**, a deviation of -12% — flagged as "critical" by the operational threshold. This highlights the sensitivity of hard-quota systems to threshold approximations learned by soft classifiers.

* **Anomaly Robustness:** The model correctly handled 7 out of 10 hand-crafted edge cases. It successfully internalized "hard constraint" rules, such as rejecting applicants with a failing subject score despite perfect scores elsewhere (e.g., 200/200/100 → correctly "rejected" due to Ukrainian < 120). However, it struggled slightly with extreme borderline rating thresholds, occasionally misclassifying applicants who were right on the edge of the admission boundary.

## License

This project is open-source and available under the [MIT License](LICENSE).
