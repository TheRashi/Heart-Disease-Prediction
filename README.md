# 🫀 Heart Disease Prediction using Machine Learning

This project uses smart computer programs to guess if a person might have heart disease by looking at their health records. The goal is to help doctors catch heart problems early and make more accurate predictions using carefully tested methods and an easy-to-use website.

---

## 📌 Features

- Data cleaning and preprocessing
- Multiple machine learning algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - XGBoost
  - Neural Network (MLPClassifier)
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Visual analysis with graphs and comparison charts
- User-friendly Streamlit app for predictions
- Human-readable label mappings (e.g., `Sex: Male/Female` instead of `0/1`)

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset]
- **Note:** The dataset is already included in this repository as `heart.csv`
- **Attributes Include:**
  - Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Max Heart Rate, etc.
  - **Target variable:** Presence (`1`) or Absence (`0`) of heart disease

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook / Google Colab
- VS Code

---

## 🧪 Project Files

| File | Purpose |
|------|---------|
| `heart.csv` | Dataset used for training and prediction |
| `Heart Disease Model Evaluation.ipynb` | Colab/Jupyter Notebook for training, evaluating and comparing ML models to find the most effective one (KNN performs best) |
| `heart_app.py` | Streamlit web application for real-time user input and prediction |
| `evaluate_model_using_sample_input.py` | Command-line interface to test predictions directly in the terminal using predefined or manual input |

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/palharsh1103/Heart-Disease-Prediction-using-Machine-Learning-Models.git
cd Heart-Disease-Prediction-using-Machine-Learning-Models
