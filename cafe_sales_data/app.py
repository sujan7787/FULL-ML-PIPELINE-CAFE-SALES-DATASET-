import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("tuned_random_forest_regressor.pkl")

st.title("Coffee Shop Spending Predictor")

st.write("Enter customer order details")

# INPUTS
quantity = st.number_input("Quantity", min_value=1, max_value=10)

price = st.number_input("Price Per Unit", min_value=1.0)

credit_card = st.checkbox("Payment Method: Credit Card")
digital_wallet = st.checkbox("Payment Method: Digital Wallet")

takeaway = st.checkbox("Takeaway")

coffee = st.checkbox("Coffee")
cookie = st.checkbox("Cookie")
juice = st.checkbox("Juice")
salad = st.checkbox("Salad")
sandwich = st.checkbox("Sandwich")
smoothie = st.checkbox("Smoothie")
tea = st.checkbox("Tea")

# PREDICT BUTTON
if st.button("Predict Total Spent"):

    input_data = pd.DataFrame([[

        quantity,
        price,
        int(credit_card),
        int(digital_wallet),
        int(takeaway),
        int(coffee),
        int(cookie),
        int(juice),
        int(salad),
        int(sandwich),
        int(smoothie),
        int(tea)

    ]],

    columns=[
        "Quantity",
        "Price Per Unit",
        "Payment Method_Credit Card",
        "Payment Method_Digital Wallet",
        "Location_Takeaway",
        "item_Coffee",
        "item_Cookie",
        "item_Juice",
        "item_Salad",
        "item_Sandwich",
        "item_Smoothie",
        "item_Tea"
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Total Spent: ${prediction[0]:.2f}")