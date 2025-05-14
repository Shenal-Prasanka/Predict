from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model at startup
model = joblib.load('model/optimized_moisture_predictor.joblib')

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from form
        data = request.get_json()
        brix = float(data['brix'])
        ph = float(data['ph'])
        acidity = float(data['acidity'])
        
        # Make prediction
        prediction = model.predict([[brix, ph, acidity]])[0]
        
        # Return result
        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 2),
            'input': {
                'BRIX': brix,
                'PH': ph,
                'ACIDITY': acidity
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)