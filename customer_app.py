import pickle
import streamlit as st
import numpy as np

# Load the model
with open("customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app setup
st.title("Customer Churn Prediction App")
st.write("This application predicts the likelihood of a customer churning based on their profile and account details.")

# Input fields for the 9 features used in the model
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.slider("Tenure (Years)", 0, 10)
balance = st.number_input("Balance ($)", min_value=0.0, step=100.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=5, step=1)
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, step=1000.0)
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode categorical input (Gender, Credit Card, Active Member)
gender_encoded = {"Male": 0, "Female": 1}
credit_card_encoded = {"Yes": 1, "No": 0}
active_member_encoded = {"Yes": 1, "No": 0}

gender_numeric = gender_encoded[gender]
credit_card_numeric = credit_card_encoded[credit_card]
active_member_numeric = active_member_encoded[active_member]

# Prepare input data
input_data = np.array([[
    credit_score, age, tenure, balance, products_number, 
    credit_card_numeric, active_member_numeric, estimated_salary, gender_numeric
]])

# Debug print statements for diagnosis
st.write("Input Data:", input_data)

# Predict button
if st.button("Predict Churn Probability"):
    prediction = model.predict(input_data)
    st.write("Raw Prediction:", prediction)  # Debug line

    # Display the result
    result = "Likely to Churn" if prediction[0] == 1 else "Unlikely to Churn"
    st.write("Prediction:", result)

    # Display probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]
        st.write("Probabilities:", probability)  # Debug line
        st.write(f"Churn Probability: {probability[1]:.2f}")

