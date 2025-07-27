import streamlit as st
import pandas as pd
import joblib

# Load the trained model and all encoders 
model = joblib.load("best_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

education_encoder = joblib.load("education_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="🙎‍♂️", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Determine if an employee's salary is over 50K or 50K or less based on their input features")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education level:", education_encoder.classes_)
occupation = st.sidebar.selectbox("Job Role", occupation_encoder.classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Encode categorical inputs before prediction
education_encoded = education_encoder.transform([education])[0]
occupation_encoded = occupation_encoder.transform([occupation])[0]

input_df = pd.DataFrame({
    'age': [age],
    'education': [education_encoded],
    'occupation': [occupation_encoded],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### Input Data (encoded)")
st.write(input_df)

if st.button("Predict Salary Class"):
    try:
        prediction_encoded = model.predict(input_df)
        prediction_decoded = label_encoder.inverse_transform(prediction_encoded)
        st.success(f"🎯 Prediction: {prediction_decoded[0]}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            st.info(f"Prediction Confidence: {max(proba[0]):.2%}")

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")


uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        # Encode categorical columns in batch_data too
        batch_data['education'] = education_encoder.transform(batch_data['education'])
        batch_data['occupation'] = occupation_encoder.transform(batch_data['occupation'])

        st.write("Uploaded data preview (encoded):", batch_data.head())

        batch_preds_encoded = model.predict(batch_data)
        batch_preds_decoded = label_encoder.inverse_transform(batch_preds_encoded)
        batch_data['PredictedClass'] = batch_preds_decoded

        st.write("Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Download Predictions CSV",
            csv,
            file_name='predicted_classes.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Error processing batch file: {str(e)}")
