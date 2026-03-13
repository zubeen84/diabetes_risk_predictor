import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")

models = {
    "Voting Ensemble (Recommended)": "ensemble_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "Balanced Random Forest": "brf_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "SVC": "svm_model.pkl"
}

model_info = {
    "Voting Ensemble (Recommended)": {
        "recall": "88.9%", "accuracy": "75%", "roc_auc": "0.833",
        "note": "Combines LR, RF and SVC. Best for catching diabetic cases. Recommended for clinical use."
    },
    "Logistic Regression": {
        "recall": "70%", "accuracy": "74%", "roc_auc": "0.833",
        "note": "Simple and interpretable. Highest ROC AUC. Good baseline model."
    },
    "Random Forest": {
        "recall": "57%", "accuracy": "76%", "roc_auc": "-",
        "note": "Best specificity (86%). Reliable at identifying non-diabetic cases."
    },
    "Balanced Random Forest": {
    "recall": "64%", "accuracy": "74%", "roc_auc": "-",
    "note": "Handles imbalanced data by undersampling majority class at each tree. More balanced recall than standard Random Forest."
    },
    "XGBoost": {
        "recall": "58%", "accuracy": "73%", "roc_auc": "-",
        "note": "Handles complex patterns well. Good overall performance."
    },
    "SVC": {
        "recall": "52%", "accuracy": "74%", "roc_auc": "-",
        "note": "Strong at finding boundaries between classes. Lower recall than ensemble."
    }
}

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("Diabetes Risk Predictor")
st.write("Adjust the patient values below and click Predict to assess diabetes risk.")
st.info("This tool is designed for use by healthcare professionals and researchers who have access to clinical patient measurements. If you are a patient please consult your doctor for a proper assessment.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    pregnancies = st.slider(
        "Pregnancies", 0, 17, 1,
        help="Number of times pregnant. Enter 0 if male or never pregnant.")
    age = st.slider(
        "Age", 1, 100, 30,
        help="Age of the patient in years.")
    bmi = st.slider(
        "BMI", 0.0, 70.0, 25.0,
        help="Body Mass Index. Healthy range is 18.5–24.9. Overweight is 25–29.9. Obese is 30+.")
    dpf = st.slider(
        "Diabetes Pedigree Function", 0.0, 2.5, 0.5,
        help="Genetic risk score calculated from family history of diabetes. Typical range 0.08–2.42. Use 0.5 if unknown.")

with col2:
    st.subheader("Clinical Measurements")
    glucose = st.slider(
        "Glucose Level", 0, 200, 100,
        help="Plasma glucose concentration (mg/dL) from 2-hour oral glucose tolerance test. Normal is below 140.")
    blood_pressure = st.slider(
        "Blood Pressure", 0, 140, 70,
        help="Diastolic blood pressure in mm Hg. Normal range is 60–80. Use 70 if unknown.")
    skin_thickness = st.slider(
        "Skin Thickness", 0, 100, 20,
        help="Triceps skin fold thickness in mm measured by clinician. Use 20 if unknown.")
    insulin = st.slider(
        "Insulin Level", 0, 900, 80,
        help="2-hour serum insulin (mu U/ml) from glucose tolerance test. Use 80 if unknown.")

st.divider()

if st.button("Predict Diabetes Risk", type="primary"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    selected_model_name = st.session_state.get(
        "selected_model", "Voting Ensemble (Recommended)")
    model = joblib.load(models[selected_model_name])

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    risk_percent = round(float(probability) * 100, 1)

    st.subheader("Prediction Result")
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Risk Score", f"{risk_percent}%")
    with col4:
        st.metric("Prediction", "Diabetic" if prediction == 1 else "Non-Diabetic")
    with col5:
        st.metric("Model Used", selected_model_name.split(" (")[0])

    st.progress(float(probability))

    if risk_percent <= 30:
        st.success(f"🟢 Low Risk ({risk_percent}%) - Low probability of diabetes.")
    elif risk_percent <= 50:
        st.warning(f"🟡 Moderate Risk ({risk_percent}%) - Some risk factors present. Monitor closely.")
    elif risk_percent <= 70:
        st.warning(f"🟠 High Risk ({risk_percent}%) - Significant risk factors. Recommend clinical review.")
    else:
        st.error(f"🔴 Very High Risk ({risk_percent}%) - High probability of diabetes. Immediate medical consultation recommended.")

    st.divider()
    st.subheader("Risk Score Guide")
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        st.info("🟢 0–30%\nLow Risk")
    with col7:
        st.warning("🟡 31–50%\nModerate Risk")
    with col8:
        st.warning("🟠 51–70%\nHigh Risk")
    with col9:
        st.error("🔴 71–100%\nVery High Risk")
    
    
    
    st.divider()
    st.subheader("What influenced this prediction?")
    factors = {
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Insulin": insulin,
        "Blood Pressure": blood_pressure
    }
    for factor, value in factors.items():
        st.write(f"**{factor}:** {value}")

with st.sidebar:
    st.header("🩺 About This App")
    st.divider()

    st.subheader("Select Model")
    selected = st.selectbox("Choose a model:", list(models.keys()))
    st.session_state["selected_model"] = selected

    info = model_info[selected]
    st.info(info["note"])

    st.divider()
    st.subheader("Model Performance")
    st.write(f"**Recall:** {info['recall']}")
    st.write(f"**Accuracy:** {info['accuracy']}")
    st.write(f"**ROC AUC:** {info['roc_auc']}")

    st.divider()
    st.subheader("All Models Comparison")
    st.write("| Model | Recall | Accuracy |")
    #st.write("|-------|--------|----------|")
    st.write("| Voting Ensemble | 88.9% | 75% |")
    st.write("| Logistic Regression | 70% | 74% |")
    st.write("| Balanced Random Forest | 64% | 74% |")
    st.write("| Random Forest | 57% | 76% |")
    st.write("| XGBoost | 58% | 73% |")
    st.write("| SVC | 52% | 74% |")

    st.divider()
    st.subheader("📋 Reference Values")
    st.write("If a measurement is unknown use these population averages:")
    st.write("**Glucose:** 100–120 mg/dL")
    st.write("**Blood Pressure:** 70 mm Hg")
    st.write("**Skin Thickness:** 20 mm")
    st.write("**Insulin:** 80 mu U/ml")
    st.write("**BMI:** 25.0")
    st.write("**DPF:** 0.5 (average genetic risk)")

    st.divider()
    st.subheader("🏛️ ISO 42001 AI Governance")
    st.write("This app is built in alignment with **ISO/IEC 42001:2023** — the international standard for AI Management Systems.")
    st.write("The following governance principles are applied:")
    st.write("✅ **Transparency** — model selection and performance metrics are fully visible to users")
    st.write("✅ **Accountability** — clear authorship and purpose stated throughout")
    st.write("✅ **Fairness** — class imbalance handled during training to reduce bias")
    st.write("✅ **Human Oversight** — tool explicitly requires clinical professional review")
    st.write("✅ **Limitations Disclosed** — scope of use clearly defined for intended users")

    st.divider()
    st.subheader("Dataset")
    st.write("Pima Indians Diabetes Database — 768 patients, 8 clinical features.")

    st.divider()
    st.warning("⚠️ For informational use only. Always consult a qualified medical professional before making clinical decisions. This tool does not replace professional medical advice.")

    st.divider()
    st.write("Built by **Zubeen Khalid**")
    st.write("MSc Applied Data Science")
    st.write("🏛️ ISO 42001 Certified AI Governance")