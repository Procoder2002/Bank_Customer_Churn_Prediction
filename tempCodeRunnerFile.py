result = "Likely to Churn" if prediction[0] == 1 else "Unlikely to Churn"
    st.write("Prediction:", result)
