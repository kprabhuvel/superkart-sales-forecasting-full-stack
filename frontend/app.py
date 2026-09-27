import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="SuperKart Sales Forecasting", page_icon="🛒", layout="wide")

st.title("🛒 SuperKart Sales Forecasting System")
st.markdown("### Predict sales revenue for products across different stores")

# Backend API URL (replace with your Github Codespace API URL)
BACKEND_URL = st.text_input("Backend API URL", value="https://github-codespace-superkart-api-url/")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0, step=0.1)
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=0.1)
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        store_establishment_year = st.number_input("Store Establishment Year", min_value=1985, max_value=2010, value=2000, step=1)
    
    with col2:
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar", "reg"])
        product_type = st.selectbox("Product Type", ["Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Meat", "Soft Drinks", "Hard Drinks", "Breads", "Breakfast", "Household", "Seafood", "Starchy Foods", "Others"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    
    submitted = st.form_submit_button("Predict Sales")
    
    if submitted:
        data = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Establishment_Year": store_establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type
        }
        
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                predicted_sales = result.get("prediction", 0)
                st.success(f"Predicted Sales Revenue: ${predicted_sales:,.2f}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

# Batch prediction section
st.markdown("---")
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    if st.button("Predict Batch"):
        records = df.to_dict('records')
        try:
            response = requests.post(f"{BACKEND_URL}/predict_batch", json={"records": records}, timeout=30)
            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                df['Predicted_Sales'] = [p['prediction'] for p in predictions]
                st.success(f"Predictions completed for {len(predictions)} records")
                st.dataframe(df)
                st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
