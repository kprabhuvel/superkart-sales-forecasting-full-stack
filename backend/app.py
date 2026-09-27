
# Flask API
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask("SuperKart Sales Forecaster")

# Load the model
model = joblib.load("superkart_model.joblib")

@app.route('/')
def home():
    return jsonify({
        "message": "SuperKart Sales Forecasting API",
        "status": "running",
        "endpoints": {
            "/v1/predict": "POST - Make single prediction",
            "/v1/predictbatch": "POST - Make batch predictions"
        }
    })

@app.post('/v1/predict')
def predict():
    try:
        data = request.get_json()
        
        # Extract features from request
        features = {
            'Product_Weight': [data.get('Product_Weight')],
            'Product_Sugar_Content': [data.get('Product_Sugar_Content')],
            'Product_Allocated_Area': [data.get('Product_Allocated_Area')],
            'Product_Type': [data.get('Product_Type')],
            'Product_MRP': [data.get('Product_MRP')],
            'Store_Establishment_Year': [data.get('Store_Establishment_Year')],
            'Store_Size': [data.get('Store_Size')],
            'Store_Location_City_Type': [data.get('Store_Location_City_Type')],
            'Store_Type': [data.get('Store_Type')]
        }
        
        # Create DataFrame
        df = pd.DataFrame(features)
        
        # Calculate derived features
        current_year = 2010
        df['Store_Age'] = current_year - df['Store_Establishment_Year']
        df['Price_Per_Weight'] = df['Product_MRP'] / df['Product_Weight']
        df['MRP_Area_Interaction'] = df['Product_MRP'] * df['Product_Allocated_Area']
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return jsonify({
            "prediction": float(prediction),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.post('/v1/predictbatch')
def predict_batch():
    try:
        file = request.files['file']
                
        # Convert to DataFrame
        df = pd.read_csv(file)
        
        # Calculate derived features
        current_year = 2010
        df['Store_Age'] = current_year - df['Store_Establishment_Year']
        df['Price_Per_Weight'] = df['Product_MRP'] / df['Product_Weight']
        df['MRP_Area_Interaction'] = df['Product_MRP'] * df['Product_Allocated_Area']
        
        # Make predictions
        predictions = model.predict(df)
        
        results = [
            {"prediction": float(pred), "status": "success"}
            for pred in predictions
        ]
        
        return jsonify({
            "predictions": results,
            "count": len(results),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
