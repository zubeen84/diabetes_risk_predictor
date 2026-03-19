# 🩺 Diabetes Risk Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Live App
👉 [Click here to open the app](https://diabetesriskpredictor-b85jnngpyqkpvbkmeac2q6.streamlit.app/)

---

##  Overview

This app predicts the probability of a patient developing diabetes 
within five years using multiple machine learning models trained on 
the Pima Indians Diabetes Database.

Built as part of a portfolio demonstrating end-to-end machine learning: from data preprocessing and model training to live deployment.

---

##  Features

- 🔢 Enter 8 clinical measurements via interactive sliders
- 🤖 Choose from 6 trained ML models including Voting Ensemble
- 📊 Risk score shown as a percentage with colour coded risk levels
- 📈 Model performance comparison table in sidebar
- ⚠️ ISO 42001 aligned responsible AI disclaimer built in
- 📱 Mobile friendly layout

---

##  Models Available

| Model | Recall | Accuracy | Notes |
|-------|--------|----------|-------|
| Voting Ensemble ⭐ | 88.9% | 75% | Recommended |
| Logistic Regression | 70% | 74% | Most interpretable |
| Balanced Random Forest | 64% | 74% | Handles imbalance well |
| Random Forest | 57% | 76% | Best specificity |
| XGBoost | 58% | 73% | Handles complex patterns |
| SVC | 52% | 74% | Strong boundary detection |

---

##  Dataset

- **Source:** Pima Indians Diabetes Database - UCI ML Repository
- **Size:** 768 patients, 8 clinical features
- **Target:** Binary - Diabetic (1) or Non-Diabetic (0)
- **Class imbalance:** Handled using IQR, log transformation and SMOTE

---

##  Input Features

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| Diabetes Pedigree Function | Genetic risk score |
| Age | Age in years |

---

##  How to Run Locally
```bash
git clone https://github.com/zubeen84/diabetes-risk-predictor.git
cd diabetes-risk-predictor
pip install -r requirements.txt
streamlit run diabetes_app.py
```

---

##  Related Project

📓 [View the full data science notebook](https://github.com/zubeen84/Machine-Learning-Model-for-Diabetes-Prediction) :preprocessing, EDA, model training and evaluation

---

## ⚠️ Disclaimer

This app is for **informational and educational purposes only**.
It is not a substitute for professional medical advice, diagnosis or treatment.
Built in alignment with **ISO 42001 AI Management** principles.

---

## 👤 Author

**Zubeen Khalid**
MSc Applied Data Science
 [LinkedIn](https://www.linkedin.com/in/zubeenkhalid)
 [GitHub](https://github.com/zubeen84)
